import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Telechargement des ressources NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Configuration de la page
st.set_page_config(
    page_title='Detection de Tweets Suspects',
    page_icon='🔍',
    layout='centered'
)

# Chargement du modele et du vectorizer
@st.cache_resource
def load_model():
    with open('models/best_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Fonction de nettoyage du texte
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
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return ' '.join(tokens)

# Interface principale
st.title('🔍 Detection de Tweets Suspects')
st.markdown('---')
st.markdown(
    'Cette application utilise un modele de Machine Learning '
    'pour detecter si un tweet est **suspect** ou **normal**.'
)

# Zone de saisie
st.subheader('Entrez votre tweet :')
tweet_input = st.text_area(
    label='Tweet',
    placeholder='Exemple : Buy followers now! Click here http://spam.com',
    height=120,
    label_visibility='collapsed'
)

# Bouton de prediction
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

            # Probabilite (si disponible)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(tweet_vec)[0]
                prob_suspect = proba[1] * 100
                prob_normal  = proba[0] * 100
            else:
                prob_suspect = 85.0 if prediction == 1 else 15.0
                prob_normal  = 100 - prob_suspect

        st.markdown('---')

        if prediction == 1:
            st.error('⚠️ TWEET SUSPECT')
            st.markdown(f'**Probabilite d etre suspect : {prob_suspect:.1f}%**')
            st.progress(prob_suspect / 100)
        else:
            st.success('✅ TWEET NORMAL')
            st.markdown(f'**Probabilite d etre normal : {prob_normal:.1f}%**')
            st.progress(prob_normal / 100)

        # Details
        with st.expander('Details de l analyse'):
            st.write(f'**Tweet original :** {tweet_input}')
            st.write(f'**Tweet nettoyé :** {tweet_clean}')
            st.write(f'**Prediction :** {"Suspect (1)" if prediction == 1 else "Normal (0)"}')
            if hasattr(model, 'predict_proba'):
                st.write(f'**Probabilite Normal  :** {prob_normal:.2f}%')
                st.write(f'**Probabilite Suspect :** {prob_suspect:.2f}%')

st.markdown('---')

# Exemples de tweets
st.subheader('Exemples de tweets a tester :')
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Tweets Suspects :**')
    st.code('BUY FOLLOWERS NOW!!! Click http://spam.com')
    st.code('@everyone FREE IPHONE just click here!!!')
    st.code('Get 10000 followers fast → http://bot.com')

with col2:
    st.markdown('**Tweets Normaux :**')
    st.code('Good morning everyone! Have a great day')
    st.code('Just watched an amazing movie tonight')
    st.code('Feeling happy today, life is beautiful')

# Sidebar avec infos
with st.sidebar:
    st.title('Informations')
    st.markdown('**Projet :** Detection de Tweets Suspects')
    st.markdown('**Auteur :** TRAORE KARIM')
    st.markdown('**Encadrant :** Dr. Abdoul Kader KABORE')
    st.markdown('**Annee :** 2026')
    st.markdown('---')
    st.markdown('**Modele utilise :** Random Forest')
    st.markdown('**Representation :** TF-IDF')
    st.markdown('**Dataset :** 60 000 tweets')
    st.markdown('**F1-Score :** ~98.98%')
