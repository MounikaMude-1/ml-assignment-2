# Machine Learning Classifier Evaluation & Streamlit Deployment

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five distinct machine learning classification algorithms on a tabular dataset to classify red wine quality effectively. The project includes building an end-to-end interactive Streamlit web frontend and deploying it to Streamlit Community Cloud.

## b. Dataset Description
- **Source:** UCI Red Wine Quality Dataset (via OpenML)
- **Instances:** 1,599 total rows (320 test samples saved in `test_data.csv`)
- **Features:** 11 continuous features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- **Target:** Binary classification (`1` for Quality >= 6, `0` for Quality < 6)

## c. GitHub Repository Link
`(https://github.com/MounikaMude-1/ml-assignment-2)`

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7375 | 0.8123 | 0.7412 | 0.7778 | 0.7590 | 0.4682 |
| **Decision Tree** | 0.7188 | 0.7654 | 0.7303 | 0.7486 | 0.7393 | 0.4321 |
| **KNN** | 0.7250 | 0.7812 | 0.7345 | 0.7601 | 0.7471 | 0.4445 |
| **Naive Bayes** | 0.7281 | 0.8041 | 0.7456 | 0.7486 | 0.7471 | 0.4523 |
| **Support Vector Machine** | 0.7594 | 0.8351 | 0.7668 | 0.7919 | 0.7792 | 0.5132 |

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a robust baseline with linear decision boundaries. Handles scaled feature sets well. |
| **Decision Tree** | Offers high interpretability but tends to slightly overfit decision boundaries on continuous feature spaces. |
| **KNN** | Performance heavily relies on feature scaling and local density, resulting in moderate boundary separation. |
| **Naive Bayes** | Computes rapidly and performs surprisingly well despite assuming feature independence. |
| **Support Vector Machine** | **Best Overall Performer.** Cleanly separates complex non-linear feature distributions using the RBF kernel. |
| **Overall Winner** | **Support Vector Machine (SVM)** — Achieved highest Accuracy (0.7594), AUC (0.8351), and MCC (0.5132). |
