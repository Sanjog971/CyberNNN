import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

rng = np.random.default_rng(42)

n_benign = 1900
n_attack = 100
benign = pd.DataFrame({
    "bytes_sent": rng.normal(3000, 800, n_benign),
    "packets": rng.normal(20, 5, n_benign),
    "label": 0,
})
attack = pd.DataFrame({
    "bytes_sent": rng.normal(2800, 800, n_attack),
    "packets": rng.normal(22, 5, n_attack),
    "label": 1,
})
df = pd.concat([benign, attack], ignore_index=True)

X = df[["bytes_sent", "packets"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, pred), 3))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 3))
print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 3))
print("F1       :", round(f1_score(y_test, pred, zero_division=0), 3))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, pred))
print("Attacks in test set:", int(y_test.sum()))