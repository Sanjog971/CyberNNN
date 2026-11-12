import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    for split, X_part, y_part in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(X_part)
        print(
            f"{name:20s} {split:5s} "
            f"precision={precision_score(y_part, pred, zero_division=0):.2f} "
            f"recall={recall_score(y_part, pred, zero_division=0):.2f} "
            f"F1={f1_score(y_part, pred, zero_division=0):.2f}"
        ) 