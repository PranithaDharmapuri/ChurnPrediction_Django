from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
import numpy as np
from .metrics_calculator import MetricsCalculator
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


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
        plt.savefig(f"ChurnAnalysis/static/ChurnAnalysis/image{count}.png")
        plt.close()

    
def preprocess_data(df, save_path=None, target_cols=None):

    
    global label_encoders
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
        context["toplist"] = [toplist ,"Dataset uploaded successfully. Ready for next actions."]
        
    elif request.headers.get("Content-Type") == "application/json":
        Preprocess_Dataset_button()
        plot_dir = os.path.join(
        str(os.getcwd()),
        "ChurnAnalysis",
        "static",
        "ChurnAnalysis"
    )

        if os.path.exists(plot_dir):
            plots = [
                f"ChurnAnalysis/{file}"
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
        


        

    return render(request, "ChurnAnalysis/upload.html", context)



def home(request):
    return render(request,"ChurnAnalysis/home.html")
    