import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)

n = 1000
benign = pd.DataFrame({
    "bytes_sent": rng.normal(3000, 800, n),
    "packets": rng.normal(20, 5, n),
    "label": 0,
})
attack = pd.DataFrame({
    "bytes_sent": rng.normal(2800, 800, n),
    "packets": rng.normal(22, 5, n),
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

predictions = model.predict(X_test)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Test accuracy:", accuracy_score(y_test, predictions))