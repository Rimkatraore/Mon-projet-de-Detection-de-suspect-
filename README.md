# Detection de Tweets Suspects

## Description
Projet de Machine Learning pour la classification automatique de tweets suspects.
Ce projet identifie si un tweet est suspect (bot, spam, contenu trompeur) ou normal.

## Auteur
- **Nom :** TRAORE KARIM
- **Encadrant :** Dr. Abdoul Kader KABORE
- **Annee :** 2026

## Dataset
- **Source :** Google Drive (fourni par l encadrant)
- **Taille :** 60 000 tweets
- **Classes :** Normal (0) = 10.3%, Suspect (1) = 89.7%

## Structure du projet
```
projet/
|-- data/
|   |-- raw/                  # Dataset original
|   `-- processed/            # Donnees nettoyees
|-- models/                   # Modeles sauvegardes
|-- reports/
|   `-- figures/              # Graphiques generes
|-- src/
|   |-- preprocess.py         # Script de nettoyage
|   |-- train.py              # Script d entrainement
|   `-- evaluate.py           # Script d evaluation
|-- app/
|   `-- app.py                # Application Streamlit
|-- notebooks/
|   `-- projet_detection_tweets_suspects.ipynb
|-- dvc.yaml                  # Pipeline DVC
|-- params.yaml               # Hyperparametres
|-- requirements.txt          # Dependances
`-- README.md
```

## Installation
```bash
git clone <url_du_repo>
cd projet-detection-tweets
pip install -r requirements.txt
dvc pull
```

## Utilisation

### Lancer le pipeline DVC
```bash
dvc repro
```

### Lancer l application Streamlit
```bash
streamlit run app/app.py
```

## Resultats

| Modele | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 97.48% | 98.07% | 99.14% | 98.60% |
| Random Forest | 98.16% | 98.56% | 99.39% | 98.98% |
| SVM (LinearSVC) | 96.79% | 98.33% | 98.08% | 98.21% |

**Meilleur modele : Random Forest (F1 = 98.98%)**

## Pipeline DVC
```
tweets_suspect.csv
       |
  preprocess.py  -> tweets_clean.csv
       |
    train.py     -> model.pkl + vectorizer.pkl
       |
  evaluate.py    -> metrics.json
```

## Technologies utilisees
- Python 3.14
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, imbalanced-learn
- DVC (Data Version Control)
- Git
- Streamlit
- NLTK

## Reproductibilite
```bash
dvc pull    # recupere les donnees
dvc repro   # relance tout le pipeline
```
