# 📉 Customer Churn Prediction App

A simple interactive Streamlit web app that predicts whether a telecom customer will churn based on their profile.

## 🚀 Live App

👉 [Click here to use the app](https://telco-customer-churn--prediction.streamlit.app/)


## 📦 Features

- Input customer data via dropdowns and sliders
- Predict churn likelihood using a trained machine learning model (XGBoost)
- Visual confidence score with progress bar
- Clean and responsive UI

## 🔧 Tech Stack

- Python 🐍
- Streamlit 🚀
- scikit-learn 🤖
- XGBoost 🌲
- Pandas / NumPy 📊

## 🧠 How It Works

The model is trained on the [Telco Customer Churn dataset]([https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv]), using various features like contract type, monthly charges, and internet service.

It outputs:
- **Churn prediction (Yes/No)**
- **Churn probability (%)**

## 📁 Project Structure
customer-churn--prediction/
├── app.py # Streamlit web app
├── churn_model_pipeline.pkl # Trained ML pipeline
├── requirements.txt # Python dependencies
└── README.md # Project description
└── train_model.py

