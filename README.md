# Home Credit Default Risk Prediction

This project uses machine learning to predict the likelihood of a customer defaulting on a loan. Built using the [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk) dataset, the model helps financial institutions make responsible and fair lending decisions for people with limited credit history.

---

## Problem Statement

Home Credit provides loans to individuals with little or no credit history. The task is to predict whether a customer will repay their loan based on their demographic, financial, and social data.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Identified missing values and feature distributions.
- Analyzed class imbalance in the `TARGET` column (majority: non-defaulters).
- Explored correlations between numerical features.
- Visualized target distribution and categorical variables.

### 2. Data Preprocessing
- Missing Values: Imputed numeric columns with median and categorical ones with mode.
- Categorical Encoding: Label encoded high-cardinality features; one-hot encoded smaller ones.
- Feature Scaling: Used `StandardScaler` for models sensitive to magnitude.

### 3. Model Training & Evaluation
Tested multiple models:
- Logistic Regression: Baseline model, improved using:
  - `class_weight='balanced'`
  - SMOTE (Synthetic Minority Oversampling)
- Random Forest: Powerful ensemble model but struggled with class imbalance.
- LightGBM: Fast and efficient boosting model.

Metrics used:
- Accuracy
- Confusion Matrix
- Precision, Recall, and F1-Score for each class

### Final Model
The final choice was Logistic Regression with SMOTE and class weighting, offering the best balance between fairness and recall for detecting defaulters.

---

## Performance Summary

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | ~72%     |
| Recall (Defaulters) | Improved significantly with SMOTE |
| Fairness Check      | Conducted across gender categories |

---

## Fairness and Ethics

Bias in credit scoring can lead to real-world discrimination. This project includes a fairness audit based on gender (`F`, `M`, `XNA`):

| Gender | Total Samples | Actual Defaulters | Predicted Defaulters | Prediction Rate |
|--------|----------------|-------------------|----------------------|-----------------|
| F      | 40561          | 6.99% (2836)      | 24.87% (10089)       | Overpredicted   |
| M      | 20940          | 10.17% (2129)     | 39.41% (8252)        | Overpredicted   |
| XNA    | 2              | 0.00% (0)         | 50.00% (1)           | Unreliable      |

Note: Gender-based discrepancies highlight the importance of further fairness corrections and using sensitive features responsibly.

---

## How to Run

1. Clone this repository or open it in Google Colab.
2. Upload the `application_train.csv` dataset.
3. Install dependencies:
   ```bash
   pip install pandas scikit-learn seaborn imbalanced-learn lightgbm

## Future Work
- Hyperparameter tuning for LightGBM.
- Feature engineering on credit history and document flags.
- Advanced fairness techniques (e.g. adversarial debiasing).
- Explainability using SHAP or LIME.

---

## Skills Demonstrated
- End-to-end machine learning pipeline
- Handling imbalanced datasets with SMOTE
- Fairness-aware machine learning practices
- Classification model evaluation and interpretation

---

## Dataset
- Source: Kaggle Home Credit Competition
- Size: ~300,000 applications with 100+ features


