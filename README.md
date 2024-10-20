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
   git clone https://github.com/nikhil1111111/Online-Doc.git 
   cd Online-Doc
   ```

2. **Install Dependencies**:
   Ensure that you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Resources**:
   The system relies on NLTK's WordNet Lemmatizer and Stopwords. Ensure that these are downloaded by adding the following to your script or executing them manually:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

4. **Run the Application**:
   Launch the Streamlit app by running the following command:
   ```bash
   streamlit run app.py
   sdk_version: 1.17.0
   ```

5. **Access the Web App**:
   Open your browser and go to the local address displayed in your terminal (typically http://localhost:8501).

## How It Works

### Input:
- The user enters a text description of the patient’s symptoms in the provided text box.

### Prediction:
- The app uses a pre-trained **TF-IDF Vectorizer** to transform the input into numerical data.
- The **Machine Learning model** (loaded from `model/passmodel.pkl`) predicts the most likely medical condition based on the input text.

### Drug Recommendations:
- The app checks the **custom dataset** (`data/custom_dataset.csv`) for drugs associated with the predicted condition.
- Drugs with high ratings and usefulness scores are recommended, and links to detailed drug information are provided.

### Display:
- The predicted condition is displayed to the user, followed by the top recommended drugs.
- Each drug comes with a link for more information, sourced from trusted medical websites like **Drugs.com**.

## Files and Directories
- **`app.py`**: The main script that runs the Streamlit web application.
- **`model/`**: Contains the pre-trained machine learning model (`passmodel.pkl`) and the TF-IDF vectorizer (`tfidfvectorizer.pkl`).
- **`data/`**: Contains the custom dataset (`custom_dataset.csv`) used for drug recommendations.
- **`requirements.txt`**: Lists all the Python dependencies needed to run the project.

## Usage Notes
- This system is designed for **educational purposes only** and should not be relied upon for actual diagnosis or medical advice.
- All predictions and drug recommendations are based on a limited dataset, and the app is not a substitute for professional healthcare advice.

## Future Enhancements
- Expand the dataset with more medical conditions and drug reviews.
- Integrate a more comprehensive NLP pipeline for improved prediction accuracy.
- Add more detailed and interactive visuals for drug recommendations and condition insights.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed for your educational projects.

## Acknowledgements
- Data sourced from public drug review datasets.
- NLP tools and machine learning libraries provided by the open-source community.


---
