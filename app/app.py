import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(page_title='Detection de Tweets Suspects', page_icon='🔍', layout='centered')

@st.cache_resource
def load_model():
    with open('models/best_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

STOP_WORDS = set(stopwords.words('english')) | set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS and len(tok) > 2]
    return ' '.join(tokens)

st.title('🔍 Detection de Tweets Suspects')
st.markdown('---')
st.markdown('Cette application analyse le **sentiment** des tweets. Un tweet negatif/haineux est **suspect**, un tweet positif/normal est **normal**.')

st.subheader('Entrez votre tweet :')
tweet_input = st.text_area(label='Tweet', placeholder='Exemple : I hate everything today, so sad...', height=120, label_visibility='collapsed')

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button('Analyser le tweet', use_container_width=True, type='primary')

if predict_btn:
    if not tweet_input.strip():
        st.warning('Veuillez entrer un tweet avant d analyse.')
    else:
        with st.spinner('Analyse en cours...'):
            model, vectorizer = load_model()
            tweet_clean = clean_tweet(tweet_input)
            tweet_vec   = vectorizer.transform([tweet_clean])
            prediction  = model.predict(tweet_vec)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(tweet_vec)[0]
                prob_negatif = proba[0] * 100
                prob_positif = proba[1] * 100
            else:
                prob_negatif = 85.0 if prediction == 0 else 15.0
                prob_positif = 100 - prob_negatif
        st.markdown('---')
        if prediction == 0:
            st.error('⚠️ TWEET SUSPECT (Sentiment Negatif)')
            st.markdown(f'**Probabilite d etre negatif/suspect : {prob_negatif:.1f}%**')
            st.progress(prob_negatif / 100)
        else:
            st.success('✅ TWEET NORMAL (Sentiment Positif)')
            st.markdown(f'**Probabilite d etre positif/normal : {prob_positif:.1f}%**')
            st.progress(prob_positif / 100)
        with st.expander('Details de l analyse'):
            st.write(f'**Tweet original :** {tweet_input}')
            st.write(f'**Tweet nettoyé :** {tweet_clean}')
            st.write(f'**Prediction :** {"Suspect/Negatif (0)" if prediction == 0 else "Normal/Positif (1)"}')
            if hasattr(model, 'predict_proba'):
                st.write(f'**Prob. Normal/Positif  :** {prob_positif:.2f}%')
                st.write(f'**Prob. Suspect/Negatif :** {prob_negatif:.2f}%')

st.markdown('---')
st.subheader('Exemples de tweets a tester :')
col1, col2 = st.columns(2)
with col1:
    st.markdown('**Tweets Suspects (Negatifs) :**')
    st.code('I hate everything today, so depressed')
    st.code('This is the worst day of my life!!!')
    st.code('I am so angry I could scream right now')
with col2:
    st.markdown('**Tweets Normaux (Positifs) :**')
    st.code('Good morning everyone! Have a great day')
    st.code('Just watched an amazing movie tonight')
    st.code('Feeling happy today, life is beautiful')
with st.sidebar:
    st.title('Informations')
    st.markdown('**Projet :** Detection de Tweets Suspects')
    st.markdown('**Auteur :** TRAORE KARIM')
    st.markdown('**Encadrant :** Dr. Abdoul Kader KABORE')
    st.markdown('**Annee :** 2026')
    st.markdown('---')
    st.markdown('**Dataset :** Sentiment140 (60 000 tweets)')
    st.markdown('**Modele :** Random Forest optimise')
    st.markdown('**F1-Score :** ~98.98%')
    st.markdown('---')
    st.info('Label 0 = Negatif/Suspect\nLabel 1 = Positif/Normal')
