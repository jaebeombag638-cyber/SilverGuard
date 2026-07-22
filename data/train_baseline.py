"""
baseline 모델(RandomForest, XGBoost) 학습 + 평가.
data/features.csv -> stratified 70/15/15 split -> train -> val 평가.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("features.csv")

y = (df["label"] == "phishing").astype(int)
X = df.drop(columns=["url", "label", "source"])
X = pd.get_dummies(X, columns=["tld"], prefix="tld")

# stratified 70/15/15: 먼저 70/30으로 나누고, 그 30을 다시 반반
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, random_state=42, eval_metric="logloss"
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(f"\n=== {name} (validation) ===")
    print(classification_report(y_val, pred, target_names=["benign", "phishing"], digits=4))
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_val, pred))
