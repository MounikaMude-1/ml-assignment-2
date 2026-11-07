import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Wine Quality Classification", layout="wide")

st.title("📊 Wine Quality ML Classification Benchmark")
st.markdown("Upload test dataset (CSV) containing feature variables and the `target` column to evaluate pre-trained models.")

# Sidebar Controls
st.sidebar.header("Configurations")

# Feature 1: CSV File Upload
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

# Feature 2: Model Selection Dropdown
model_choice = st.sidebar.selectbox(
    "Select ML Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        
        if 'target' not in df_test.columns:
            st.error("Error: The uploaded CSV must contain a target column named 'target'.")
        else:
            X_test = df_test.drop(columns=['target'])
            y_test = df_test['target']
            
            # Load pre-trained scaler and chosen model
            scaler = joblib.load("model/scaler.pkl")
            model_file_map = {
                "Logistic Regression": "model/logistic_regression.pkl",
                "Decision Tree": "model/decision_tree.pkl",
                "KNN": "model/knn.pkl",
                "Naive Bayes": "model/naive_bayes.pkl",
                "Support Vector Machine": "model/support_vector_machine.pkl"
            }
            
            model = joblib.load(model_file_map[model_choice])
            
            # Scale input features for distance/linear models
            if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            # Model Predictions (Without retraining!)
            y_pred = model.predict(X_eval)
            y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader(f"Evaluation Results for: {model_choice}")
            
            # Feature 3: Display Metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            col_left, col_right = st.columns(2)

            # Feature 4: Display Confusion Matrix
            with col_left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            # Display Classification Report
            with col_right:
                st.subheader("Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_dict).transpose())

    except Exception as e:
        st.error(f"Error evaluating dataset: {e}")
else:
    st.info("👈 Please upload your `test_data.csv` in the sidebar to view evaluation metrics.")
