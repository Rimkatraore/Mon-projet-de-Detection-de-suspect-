import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

MODEL_PATH      = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
TEST_PATH       = "data/processed/test_data.csv"
METRICS_PATH    = "reports/metrics.json"

print("Chargement du modele et des donnees de test...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

test_data = pd.read_csv(TEST_PATH)
X_test = vectorizer.transform(test_data["text"])
y_test = test_data["label"]

y_pred = model.predict(X_test)

metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall":    round(recall_score(y_test, y_pred), 4),
    "f1_score":  round(f1_score(y_test, y_pred), 4)
}

print("Resultats :")
for k, v in metrics.items():
    print(f"  {k} : {v:.4f}")

print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))

os.makedirs("reports", exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Metriques sauvegardees :", METRICS_PATH)
