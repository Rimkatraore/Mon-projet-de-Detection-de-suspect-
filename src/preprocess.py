
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Téléchargement des ressources NLTK
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Configuration
INPUT_PATH  = "data/raw/tweets_suspect.csv"
OUTPUT_PATH = "data/processed/tweets_clean.csv"
TEXT_COL    = "message"
LABEL_COL   = "label"

STOP_WORDS = set(stopwords.words("english")) | set(stopwords.words("french"))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", "", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)

print("Chargement des données...")
df = pd.read_csv(INPUT_PATH)
print(f"Dataset chargé : {df.shape}")

df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)
df["cleaned_text"] = df[TEXT_COL].apply(clean_tweet)
df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Données nettoyées sauvegardées : {OUTPUT_PATH}")
print(f"   Shape finale : {df.shape}")
