from hmmlearn import hmm
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
import numpy as np

X = [['walk', 'in', 'the', 'park'], ['eat', 'apple'], ['eat', 'apple', 'in', 'the', 'morning']]
y = [['V', 'P', 'D', 'N'], ['V', 'N'], ['V', 'N', 'P', 'D', 'N']]

words = list(set(w for seq in X for w in seq))
tags = list(set(t for seq in y for t in seq))
w2i = {w: i for i, w in enumerate(words)}
t2i = {t: i for i, t in enumerate(tags)}
i2t = {i: t for t, i in t2i.items()}

X_flat = np.array([w2i[w] for seq in X for w in seq]).reshape(-1, 1)
lengths = [len(seq) for seq in X]
y_flat = [t2i[t] for seq in y for t in seq]

model = hmm.MultinomialHMM(n_components=len(tags), random_state=42)
model.fit(X_flat, lengths)
pred = model.predict(X_flat)
print("HMM:\n", classification_report(y_flat, pred, target_names=tags))

def to_features(seq): return [{'w': w} for w in seq]
X_feat = [to_features(seq) for seq in X]
crf = CRF()
crf.fit(X_feat, y)
pred_crf = crf.predict(X_feat)
print("CRF:\n", classification_report([t for seq in y for t in seq], [t for seq in pred_crf for t in seq]))