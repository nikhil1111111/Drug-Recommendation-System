---
title: Drugrecommendationsystem
emoji: 🔥
colorFrom: yellow
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
duplicated_from: shibinashraf/drugrecommendationsystem
---

# Patient Diagnosis and Drug Recommendation System (PDDRS)

## Overview
The **Patient Diagnosis and Drug Recommendation System (PDDRS)** is a web-based application built with **Streamlit** that allows users to input a patient's symptoms and receive predictions about possible conditions, along with recommended drugs for the predicted condition. The system utilizes **machine learning models** and a **TF-IDF vectorizer** to analyze the input text, predict the most likely medical condition, and recommend drugs from a preloaded dataset.

> **Note:** This tool is for educational purposes only and should not be used for actual medical diagnosis or treatment.

## Features
- **Text-based Condition Prediction**: Users can input detailed descriptions of symptoms to receive a predicted condition.
- **Drug Recommendations**: Based on the predicted condition, the system suggests the top-rated drugs from the dataset.
- **Interactive Interface**: The user-friendly interface allows easy input of symptoms and provides predictions with detailed links for more information on conditions and drugs.
- **Educational Links**: Direct links to trusted medical websites for more information on the predicted conditions and recommended drugs.

## Technologies Used
- **Streamlit**: Used to build the web application and provide the interactive user interface.
- **Natural Language Processing (NLP)**:
  - **nltk**: Used for stopword removal and word lemmatization.
  - **BeautifulSoup**: Used to clean raw text inputs by removing HTML tags.
- **Machine Learning**:
  - **joblib**: Used to load the pre-trained machine learning model and TF-IDF vectorizer.
  - **scikit-learn**: The machine learning model was trained using this library.
- **Pandas**: Used for data manipulation and filtering of the custom drug dataset.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nikhil1111111/Drug-Recommendation-System.git 
   cd Online-Doc
