from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
import numpy as np
from .metrics_calculator import MetricsCalculator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from collections import Counter
from wordcloud import WordCloud



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.util import ngrams

from django.http import JsonResponse
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib, os
from sklearn.ensemble import RandomForestClassifier

global df, X, Y_dict




    
def plot_target_distributions(Y_dict):

    # Convert to DataFrame
    y_df = pd.DataFrame(Y_dict)

    # Loop through each target column
    for count,col in enumerate(y_df.columns):

        # Print total number of rows
        total_rows = len(y_df[col])
        print(f"{col} → Total Rows: {total_rows}")

        # Plot count distribution
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=y_df[col])

        
        # Add count labels on each bar
        
        for p in ax.patches:
            
            height = p.get_height()
            ax.annotate(
                f'{height}',                     # text
                (p.get_x() + p.get_width()/2, height),  # position
                ha='center', va='bottom', 
                fontsize=10, fontweight='bold'
            )

        plt.title(f'Class Distribution: {col}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        # plt.show()
        # os.makedirs("outputs/plots",exist_ok=True)
        plt.savefig(f"ChurnAnalysis/static/preprocess/image{count}.png")
        plt.close()

    
def preprocess_data(df, save_path=None, target_cols=None):

    global label_encoders    
    # global df

    label_encoders = {}  # dictionary to hold encoders for each target column

    if save_path and os.path.exists(save_path):
        print(f"Loading existing preprocessed file: {save_path}")
        df = pd.read_csv(save_path)
    else:
        print("Preprocessing data" + (f" and saving to: {save_path}" if save_path else " (no saving)"))
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
            return ' '.join(tokens)

        # Separate target columns
        target_df = None
        if target_cols:
            existing_targets = [col for col in target_cols if col in df.columns]
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

        # Process text columns
        text_columns = df.select_dtypes(include='object').columns
        for col in text_columns:
            df[f'processed_{col}'] = df[col].apply(clean_text)

        # Drop original text columns
        df.drop(columns=text_columns, inplace=True)

        # Reattach target columns
        if target_df is not None:
            for col in target_df.columns:
                df[col] = target_df[col]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        # Save only if path is specified
        if save_path:
            df.to_csv(save_path,index=False)

    # Select processed and numerical columns
    processed_text_cols = [col for col in df.columns if col.startswith('processed_')]
    non_text_cols = [col for col in df.columns if col not in processed_text_cols + (target_cols if target_cols else [])]

    # Join processed text columns into one string
    X_text = df[processed_text_cols].astype(str).agg(' '.join, axis=1)

    # Combine with numerical columns if any
    X_numeric = df[non_text_cols].values if non_text_cols else None
    if X_numeric is not None and len(X_numeric) > 0:
        X = [f"{text} {' '.join(map(str, numeric))}" for text, numeric in zip(X_text, X_numeric)]
    else:
        X = X_text.tolist()

    # Encode multiple target columns
    Y_dict = {}
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le

    return X, Y_dict

def Preprocess_Dataset_button():
    global X, Y_dict,df
    global metrics_calculator_dict,target_cols
    MODEL_DIR = "model"
    target_cols = ["churn_risk_score", "complaint_status"]
    X, Y_dict = preprocess_data(
        df,
        save_path="model/cleaned_data.csv",
        target_cols=target_cols
    )

    metrics_calculator_dict = {}
    lab_file=[]
    
    for col, le in label_encoders.items():

        labels = list(le.classes_)    # class names list

        # Print readable summary
        for idx, class_name in enumerate(labels):
            lab_file.append(f"  {idx}: {class_name}\n")


        label_file = os.path.join(MODEL_DIR, f"labels_{col}.npy")
        np.save(label_file, np.array(labels), allow_pickle=True)


        metrics_calculator_dict[col] = MetricsCalculator(labels)


    # Plot distributions
    plot_target_distributions(Y_dict)




def upload_dataset(request):
    print("upload button hit")
    context = {
        "show_block": False,  
        "file_name": None,
        "file_path": None,
        "toplist": None
    }
    
    global df


    if request.method == "POST" and request.FILES.get("dataset"):
        dataset_file = request.FILES["dataset"]
        

        fs = FileSystemStorage(location="uploads/")
        filename = fs.save(None,dataset_file)
        file_path = fs.path(filename)
        df=pd.read_csv(file_path)
        toplist=df.head()
        
       


        context["show_block"] = True  
        context["file_name"] = filename
        context["file_path"] = file_path
        context["toplist"] = toplist
        context["Success_msg"] ="Dataset uploaded successfully. Ready for next actions."
        
    elif request.headers.get("Content-Type") == "application/json":
        data = json.loads(request.body)
        action = data.get("action")
        if action=="processing_btn":
            print("preprocess button hit")
            Preprocess_Dataset_button()
            plot_dir = os.path.join(
        str(os.getcwd()),
        "ChurnAnalysis",
        "static",
        "preprocess"
        )
            sub_folder="preprocess"
            
            
        elif action=="eda_btn":
            print("eda button hit")
            eda_nlp_analysis()
            plot_dir = os.path.join(
        str(os.getcwd()),
        "ChurnAnalysis",
        "static",
        "eda"
        )
            sub_folder="eda"

        if os.path.exists(plot_dir):
            plots = [
                f"{sub_folder}/{file}"
                for file in os.listdir(plot_dir)
                if file.endswith(".png")
            ]
        else:
            plots = []
        
        return JsonResponse({
                "status": "success",
                "message": "Processing successful!",
                "plots" : plots
                
            })
    #   elif request.headers.get("Content-Type") == "application/json":    
    # # elif "application/json" in request.headers.get("Content-Type", ""):
    #     print("eda button hit")
    #     eda_nlp_analysis()
    #     plot_dir = os.path.join(
    #     str(os.getcwd()),
    #     "ChurnAnalysis",
    #     "static",
    #     "ChurnAnalysis"
    # )

    #     if os.path.exists(plot_dir):
    #         plots = [
    #             f"ChurnAnalysis/{file}"
    #             for file in os.listdir(plot_dir)
    #             if file.endswith(".png")
    #         ]
    #     else:
    #         plots = []
        
    #     return JsonResponse({
    #             "status": "success",
    #             "message": "EDA successful!",
    #             "plots" : plots
                
    #         })


        

    return render(request, "ChurnAnalysis/upload.html", context)

def eda_nlp_analysis():
    global X
    
    text=[]
    X_text=X
    num_words=100
    top_n_words=20

    text.append("Generating NLP EDA Visualizations..."+"\n\n")

    # Flatten all tokens from all texts
    all_tokens = [word for doc in X_text for word in word_tokenize(doc)]

    # --- 1. WordCloud ---
    word_freq = Counter(all_tokens)
    wc = WordCloud(width=800, height=400, max_words=num_words, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top {num_words} Words - WordCloud")
    plt.savefig(f"ChurnAnalysis/static/eda/WordCloud.png")
    plt.close()

    # --- 2. Top-N Frequent Words ---
    common_words = word_freq.most_common(top_n_words)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Top {top_n_words} Most Frequent Words")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.savefig(f"ChurnAnalysis/static/eda/topNfrequentWords.png")
    plt.close()

    # --- 3. Document Length Histogram ---
    doc_lengths = [len(word_tokenize(doc)) for doc in X_text]
    plt.figure(figsize=(10, 5))
    sns.histplot(doc_lengths, bins=20, kde=True, color='teal')
    plt.title("Distribution of Document Lengths (in words)")
    plt.xlabel("Number of Words per Document")
    plt.ylabel("Frequency")
    plt.savefig(f"ChurnAnalysis/static/eda/DocLenHistogram.png")
    plt.close()

    # --- 4. POS Tag Frequency ---
    all_pos = [tag for _, tag in pos_tag(all_tokens)]
    pos_counts = Counter(all_pos).most_common()
    pos_tags, pos_freqs = zip(*pos_counts)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(pos_tags), y=list(pos_freqs), palette="coolwarm")
    plt.title("Part of Speech (POS) Tag Frequency")
    plt.xlabel("POS Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.savefig(f"ChurnAnalysis/static/eda/POSTagFrequency.png")
    plt.close()

    # --- 5. Bigram Frequency Plot ---
    bigrams = list(ngrams(all_tokens, 2))
    bigram_freq = Counter(bigrams).most_common(top_n_words)
    bigram_labels = [' '.join(b) for b, _ in bigram_freq]
    bigram_counts = [count for _, count in bigram_freq]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=bigram_counts, y=bigram_labels, palette="magma")
    plt.title(f"Top {top_n_words} Bigrams")
    plt.xlabel("Count")
    plt.ylabel("Bigram")
    plt.savefig(f"ChurnAnalysis/static/eda/BigramFreqPlot.png")
    plt.close()



def home(request):
    return render(request,"ChurnAnalysis/home.html")
    