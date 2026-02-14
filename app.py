import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("2025AA05655 - Income Classification - ML Assignment 2")

label_encoders = joblib.load("model/label_encoders.pkl")
log_model = joblib.load("model/log_model.pkl")
dt_model = joblib.load("model/dt_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")
nb_model = joblib.load("model/nb_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")
xgb_model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    X = data.drop("income", axis=1)
    y = data["income"]

    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)

    for column in data.select_dtypes(include='object').columns:
        if column in label_encoders:
            le = label_encoders[column]
            data[column] = le.transform(data[column])

    X = data.drop("income", axis=1)
    y = data["income"]

    model = models[model_choice]
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = None

    st.write("### Evaluation Metrics")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"AUC: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)
