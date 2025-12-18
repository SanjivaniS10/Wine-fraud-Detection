🍷 Wine Fraud Detection Using Machine Learning (SVM)

A Machine Learning project that detects whether wine is legit or fraudulent based on various chemical properties.
This project uses Support Vector Machine (SVM) for classification and includes a Streamlit web application for user interaction.

📌 Project Overview

The goal of this project is to build a classification model that predicts wine authenticity by analyzing chemical parameters such as acidity, chlorides, sulphates, pH, and more.

The project includes:

Data preprocessing

Exploratory data analysis

Model training using SVM

Saving the trained model with pickle

Deploying an interactive Streamlit web app

📁 Project Structure
wine_fraud_detection/
│
├── data/
│   └── wine.csv
│
├── notebooks/
│   └── wine_fraud_detection.ipynb
│
├── app/
│   ├── app.py
│   └── model.pkl
│
├── requirements.txt
└── README.md

🚀 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Streamlit

Pickle

🔍 Machine Learning Workflow
1️⃣ Data Loading & Cleaning

Handle missing values

Standardize column names

Remove duplicates

Treat outliers

2️⃣ Exploratory Data Analysis

Distribution of chemical features

Correlation heatmap

Boxplots to check outliers

3️⃣ Feature Scaling

StandardScaler applied to numerical columns

4️⃣ Model Building (SVM)

Train-test split

Fit SVM model

Evaluate using:

Accuracy

Confusion matrix

Classification report

5️⃣ Model Saving

The trained SVM model is saved as:

model.pkl

6️⃣ Streamlit Web App

Users can input wine chemical values and get prediction results:

“Legit Wine”

“Fraud Wine”

🧪 How to Run the Project Locally
✔️ Step 1: Clone the Repository
git clone https://github.com/yourusername/wine_fraud_detection.git
cd wine_fraud_detection

✔️ Step 2: Install Dependencies
pip install -r requirements.txt

✔️ Step 3: Run Streamlit App
streamlit run app/app.py

🗂 Sample app.py Snippet
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🍷 Wine Fraud Detection App")

# Example inputs
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0)

if st.button("Predict"):
    features = np.array([[fixed_acidity, volatile_acidity]])
    result = model.predict(features)[0]
    if result == 1:
        st.success("Legit Wine")
    else:
        st.error("Fraud Wine")

📄 requirements.txt

Example packages:
![first page](https://github.com/SanjivaniS10/Wine-fraud-Detection/blob/main/snap01.png)
![second Page](https://github.com/SanjivaniS10/Wine-fraud-Detection/blob/main/Snsp%2002.png)




streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
pickle-mixin

📊 Results

The SVM model achieved:

High accuracy on test data

Good precision and recall

Reliable prediction performance

🎯 Project Aim

To develop a practical and simple ML-powered tool that helps detect fraudulent wine based on its chemical composition.

👩‍💻 Author

Your Name
Data Science & Machine Learning Enthusiast

⭐ Contribute

Pull requests are welcome!
If you’d like a more advanced UI or model improvement, feel free to suggest.
