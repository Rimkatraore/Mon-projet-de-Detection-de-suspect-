
import pandas as pd
import pickle
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Configuration
INPUT_PATH   = "data/processed/tweets_clean.csv"
MODEL_PATH   = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
PARAMS_PATH  = "params.yaml"
TEXT_COL     = "cleaned_text"
LABEL_COL    = "label"

print("Chargement des données nettoyées...")
df = pd.read_csv(INPUT_PATH)
print(f"Shape : {df.shape}")

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL],
    test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

# Vectorisation TF-IDF
print("Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Entraînement avec class_weight balanced
print("Entraînement du modèle...")
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# Sauvegarde
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

# Sauvegarde des indices de test pour l'évaluation
test_data = pd.DataFrame({"text": X_test, "label": y_test})
test_data.to_csv("data/processed/test_data.csv", index=False)

print(f"✅ Modèle sauvegardé : {MODEL_PATH}")
print(f"✅ Vectorizer sauvegardé : {VECTORIZER_PATH}")
