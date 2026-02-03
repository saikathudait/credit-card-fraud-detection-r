# Credit Card Fraud Detection Using Classification Techniques in R

## Project Overview
This project investigates credit card fraud detection using classification-based machine learning techniques implemented in R. Due to the highly imbalanced nature of fraud datasets, the study focuses on robust preprocessing, class imbalance handling, and performance evaluation using ROC–AUC and recall-based metrics.

## Application Area
Credit card fraud detection is a critical challenge for financial institutions, as fraudulent transactions cause financial losses, operational disruption, and reputational damage. Machine learning models are applied to identify rare fraud patterns within large volumes of transaction data.

## Dataset
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Source: Kaggle – Credit Card Fraud Detection Dataset  
- Total transactions: 284,807  
- Fraud cases: 492 (0.17%)  
- Features:  
  - V1–V28 (PCA-transformed variables)  
  - Time (seconds since first transaction)  
  - Amount (transaction value)  
  - Class (NonFraud, Fraud)

## Data Preprocessing
- Conversion of target variable to factor
- Z-score scaling of Time and Amount
- Stratified 70:30 train-test split
- Down-sampling applied to the training data to address class imbalance

## Exploratory Data Analysis
- Class distribution visualisation
- Transaction amount analysis by class
- Time vs Amount scatter plots
- Correlation matrix of numeric features

## Models Implemented
- Logistic Regression
- Decision Tree (rpart)
- Random Forest
- Support Vector Machine (Radial Kernel)

All models were trained using repeated k-fold cross-validation with ROC as the optimisation metric.

## Model Evaluation
Performance was assessed on the test dataset using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC Curve
- Area Under the Curve (AUC)

Logistic Regression achieved the highest AUC, while Random Forest demonstrated a strong balance between recall and overall performance.

## Repository Structure
credit-card-fraud-detection-r/
│
├── credit-card-fraud-detection.R
├── creditcard.csv
├── figures/
│ ├── class_distribution.png
│ ├── amount_distribution.png
│ ├── amount_boxplot.png
│ ├── time_vs_amount_sample.png
│ ├── correlation_matrix.png
│ └── roc_comparison.png
├── model_results_summary.csv
├── model_logistic_regression.rds
├── model_decision_tree.rds
├── model_random_forest.rds
├── model_svm_radial.rds
└── README.md


## Tools and Libraries
- tidyverse
- caret
- pROC
- corrplot
- e1071
- randomForest
- rpart

## How to Run
1. Clone the repository  
2. Open the R script in RStudio  
3. Ensure the dataset is in the working directory  
4. Run the script sequentially to reproduce results

## Key Learning Outcomes
- Handling imbalanced datasets in fraud detection
- Applying multiple classification algorithms in R
- Evaluating models using ROC–AUC rather than accuracy
- Translating machine learning outputs into business-relevant insights


