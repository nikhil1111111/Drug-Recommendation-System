
import os
import joblib
import pandas as pd
import re
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Load model and vectorizer
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'data/custom_dataset.csv'

vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title='PDDRS', page_icon='👨‍⚕️', layout='wide')
st.markdown("""
    <p style="font-family:sans-serif; color:#D30000; font-size:24px;">
    <marquee>Warning: For Educational purposes only. Not recommended for actual use.!</marquee>
    </p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    color:#1D8096;
    margin-left:7%;
}
</style>
""", unsafe_allow_html=True)

st.title("Patient Diagnosis and Drug Recommendation System 💉")
st.header("The system can detect the following diseases and recommend top drugs.")
conditions_links = {
    'Acne': "https://www.niams.nih.gov/health-topics/acne",
    'ADHD': "https://www.cdc.gov/ncbddd/adhd/facts.html",
    'Depression': "https://www.who.int/news-room/fact-sheets/detail/depression",
    'Diabetes, Type 2': "https://diabetes.org/diabetes/type-2",
    'Migraine': "https://medlineplus.gov/ency/article/000709.htm",
    'Pneumonia': "https://nhlbi.nih.gov/health/pneumonia"
}

for condition, link in conditions_links.items():
    st.markdown(f'<p class="big-font">• <a href="{link}" target="_blank">{condition}</a></p>', unsafe_allow_html=True)

st.header("Enter Patient Condition:")
raw_text = st.text_area('Describe the symptoms and condition of the patient:', height=50, help="Please provide a detailed description of the symptoms.")

def predict(raw_text):
    global predicted_cond
    global top_drugs
    if raw_text.strip() != "":
        clean_text = cleanText(raw_text)
        tfidf_vect = vectorizer.transform([clean_text])
        prediction = model.predict(tfidf_vect)
        predicted_cond = prediction[0]
        df = pd.read_csv(DATA_PATH)
        top_drugs = top_drugs_extractor(predicted_cond, df)

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stop]
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmitize_words)

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 90)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(4).tolist()
    return list(set(drug_lst))

predict_button = st.button("Predict")

if predict_button:
    if raw_text.strip() == "":
        st.warning("Please enter the patient's condition and symptoms to make a prediction.")
    else:
        predict(raw_text)
        st.header('Condition Predicted')
        st.subheader(predicted_cond)
        st.header('Top Recommended Drugs')
        for drug in top_drugs:
            st.subheader(drug)
            st.markdown(f"[Learn more about {drug}](https://www.drugs.com/search.php?searchterm={drug.replace(' ', '+')})", unsafe_allow_html=True)

st.markdown("""
<style>
    .reportview-container .main .block-container{
        max-width: 80%;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)
