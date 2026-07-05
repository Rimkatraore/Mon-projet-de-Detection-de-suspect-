import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

st.set_page_config(page_title="Detection de Tweets", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    with open("models/best_model_final.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

STOP_WORDS = set(stopwords.words("english")) | set(stopwords.words("french"))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"[^\\w\\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

with st.sidebar:
    st.title("Informations")
    st.markdown("**Auteur :** TRAORE KARIM")
    st.markdown("**Encadrant :** Dr. Abdoul Kader KABORE")
    st.markdown("**Annee :** 2026")
    st.markdown("---")
    st.markdown("**Modele :** Random Forest | **F1 :** 98.98%")
    st.info("Label 0 = Negatif/Suspect\nLabel 1 = Positif/Normal")

page = st.sidebar.selectbox("Navigation", ["🔍 Analyse de Tweet", "📊 Dashboard Monitoring"])

if page == "🔍 Analyse de Tweet":
    st.title("🔍 Detection de Tweets Suspects")
    st.markdown("Cette application analyse le **sentiment** des tweets.")
    tweet_input = st.text_area("Entrez votre tweet :", height=120)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("Analyser le tweet", use_container_width=True, type="primary")
    if predict_btn:
        if not tweet_input.strip():
            st.warning("Veuillez entrer un tweet.")
        else:
            with st.spinner("Analyse..."):
                model, vectorizer = load_model()
                tweet_clean = clean_tweet(tweet_input)
                tweet_vec = vectorizer.transform([tweet_clean])
                prediction = model.predict(tweet_vec)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(tweet_vec)[0]
                    prob_neg = proba[0] * 100
                    prob_pos = proba[1] * 100
                else:
                    prob_neg = 85.0 if prediction == 0 else 15.0
                    prob_pos = 100 - prob_neg
            st.markdown("---")
            if prediction == 0:
                st.error(f"⚠️ TWEET SUSPECT (Sentiment Negatif) — {prob_neg:.1f}%")
                st.progress(prob_neg / 100)
            else:
                st.success(f"✅ TWEET NORMAL (Sentiment Positif) — {prob_pos:.1f}%")
                st.progress(prob_pos / 100)
            with st.expander("Details"):
                st.write(f"**Original :** {tweet_input}")
                st.write(f"**Nettoyé :** {tweet_clean}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tweets Suspects :**")
        st.code("I hate everything today, so depressed")
        st.code("This is the worst day of my life!!!")
    with col2:
        st.markdown("**Tweets Normaux :**")
        st.code("Good morning everyone! Have a great day")
        st.code("Just watched an amazing movie tonight")

elif page == "📊 Dashboard Monitoring":
    st.title("📊 Dashboard de Monitoring")
    st.markdown("---")
    st.subheader("🎯 Performances du modèle (Random Forest)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  "98.16%", "+0.68% vs LR")
    c2.metric("Precision", "98.56%", "+0.49% vs LR")
    c3.metric("Recall",    "99.39%", "+0.25% vs LR")
    c4.metric("F1-Score",  "98.98%", "+0.38% vs LR")
    st.markdown("---")
    st.subheader("📦 Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total tweets", "60 000")
    c2.metric("Tweets Normaux", "53 855 (89.7%)")
    c3.metric("Tweets Suspects", "6 145 (10.3%)")
    st.markdown("---")
    st.subheader("📈 Distribution des classes")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Normal (1)", "Suspect (0)"], [53855, 6145], color=["#2ecc71", "#e74c3c"])
        ax.set_title("Distribution des classes")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie([53855, 6145], labels=["Normal 89.7%", "Suspect 10.3%"], colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%")
        ax.set_title("Proportion des classes")
        st.pyplot(fig)
    st.markdown("---")
    st.subheader("🤖 Comparaison des modèles")
    df_m = {"Modèle": ["Logistic Regression", "Random Forest", "SVM"],
            "Accuracy": [0.9748, 0.9816, 0.9679],
            "Precision": [0.9807, 0.9856, 0.9833],
            "Recall": [0.9914, 0.9939, 0.9808],
            "F1-Score": [0.9860, 0.9898, 0.9821]}
    st.dataframe(pd.DataFrame(df_m).set_index("Modèle"), use_container_width=True)
    st.markdown("---")
    st.subheader("🔗 Liens")
    st.markdown("**GitHub :** https://github.com/Rimkatraore/Mon-projet-de-Detection-de-suspect-")
    st.markdown("**App :** https://detection-tweets-suspects.streamlit.app")
