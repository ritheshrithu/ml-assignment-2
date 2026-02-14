# ML Assignment 2 – Income Classification

## a) Problem Statement

The goal of this project is to build and compare different machine learning classification models to predict whether a person earns more than $50K per year. This is a binary classification problem where the output is either <=50K or >50K.

This project covers the complete machine learning workflow including:
- Data preprocessing  
- Training multiple models  
- Evaluating model performance  
- Comparing results  
- Deploying the final solution using Streamlit  

---

## b) Dataset Description

The Adult Income dataset is taken from the UCI Machine Learning Repository. It contains demographic and work-related information such as age, education, occupation, marital status, workclass, hours-per-week, and more.

- **Number of Instances:** ~48,000  
- **Number of Features:** 14  
- **Target Variable:** Income (<=50K or >50K)  
- **Problem Type:** Binary Classification  

The dataset meets the assignment requirement of having more than 12 features and more than 500 records.

---

## c) Models Used and Evaluation Metrics

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using these performance metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.814283 | 0.841913 | 0.698166 | 0.441570 | 0.540984 | 0.449689 |
| Decision Tree | 0.807208 | 0.742946 | 0.610080 | 0.615522 | 0.612789 | 0.484452 |
| kNN | 0.818925 | 0.846963 | 0.654082 | 0.571811 | 0.610186 | 0.494839 |
| Naive Bayes | 0.783109 | 0.823390 | 0.628676 | 0.305085 | 0.410811 | 0.326139 |
| Random Forest (Ensemble) | 0.847225 | 0.898880 | 0.733189 | 0.603033 | 0.661772 | 0.568801 |
| XGBoost (Ensemble) | 0.865355 | 0.923469 | 0.770042 | 0.651204 | 0.705655 | 0.622813 |

---

## Model Performance Observations

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Logistic Regression achieved good overall accuracy but lower recall. This means it predicts positive cases carefully but misses some actual high-income individuals. Since it is a linear model, it may not capture complex patterns in the data. |
| Decision Tree | The Decision Tree model gave balanced precision and recall but lower AUC compared to ensemble models. Single trees can sometimes overfit and may not generalize well. |
| kNN | The kNN model performed slightly better than Logistic Regression in terms of F1 and MCC score. This suggests that local similarity between data points helps in classification. |
| Naive Bayes | Naive Bayes showed the weakest performance, especially in recall. This indicates that the assumption of independent features does not hold strongly for this dataset. |
| Random Forest (Ensemble) | Random Forest improved performance compared to a single Decision Tree. Ensemble learning helped reduce overfitting and improved overall accuracy and AUC. |
| XGBoost (Ensemble) | XGBoost achieved the best performance across all metrics. The boosting technique helps in capturing complex relationships and improving prediction accuracy. It is the best-performing model in this project. |

---

## Streamlit Deployment

The trained models were deployed using Streamlit Community Cloud.

**Live App:**  
https://ml-assignment-2-h54pbzslfyq2mwtchrqyka.streamlit.app/

The web application allows:
- Uploading a test dataset (CSV file)  
- Selecting a model from a dropdown  
- Viewing evaluation metrics  
- Displaying the confusion matrix  

---

## Final Notes

- All six models were implemented on the same dataset.  
- All required evaluation metrics were calculated.  
- The Streamlit application meets all assignment requirements.  
