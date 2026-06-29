
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# Configuration
MODEL_PATH      = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
TEST_PATH       = "data/processed/test_data.csv"
METRICS_PATH    = "reports/metrics.json"

print("Chargement du modèle et des données de test...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

test_data = pd.read_csv(TEST_PATH)
X_test = vectorizer.transform(test_data["text"])
y_test = test_data["label"]

# Prédictions
y_pred = model.predict(X_test)

# Métriques
metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall":    round(recall_score(y_test, y_pred), 4),
    "f1_score":  round(f1_score(y_test, y_pred), 4)
}

print("
=== Résultats ===")
for k, v in metrics.items():
    print(f"  {k:12s} : {v:.4f}")

print("
=== Rapport détaillé ===")
print(classification_report(y_test, y_pred, target_names=["Normal", "Suspect"]))

# Sauvegarde des métriques
os.makedirs("reports", exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métriques sauvegardées : {METRICS_PATH}")
