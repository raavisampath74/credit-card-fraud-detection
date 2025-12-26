Project Overview
Credit card fraud is a major issue in the financial sector, causing significant losses every year. The objective of this project is to detect fraudulent credit card transactions using machine learning algorithms and a neural network model, while handling highly imbalanced data efficiently.
This project analyzes transaction data, performs exploratory data analysis (EDA), applies data preprocessing techniques, and compares the performance of multiple classification models using appropriate evaluation metrics.

Objectives
Analyze credit card transaction data
Handle severe class imbalance
Apply data preprocessing and feature scaling
Train and evaluate multiple machine learning models
Implement a neural network for fraud detection
Compare models using performance metrics and visualizations

Dataset Description
Dataset Name: Credit Card Fraud Detection
Source: Kaggle
Total Transactions: 284,807
Fraudulent Transactions: 492 (≈ 0.17%)

Features:
V1 to V28: PCA-transformed anonymized features
Time: Time elapsed since first transaction
Amount: Transaction amount
Class: Target variable (0 = Legitimate, 1 = Fraud)
The dataset is highly imbalanced, which makes accuracy alone an unreliable metric.

Exploratory Data Analysis:
Class distribution analysis
Transaction amount distribution
Time-based transaction analysis
Boxplots for outlier detection
Correlation heatmap
Feature correlation with target variable
These analyses helped in understanding data imbalance, identifying important features, and guiding preprocessing decisions.

Data Preprocessing:
Removal of missing values
Removal of duplicate records
Feature scaling (Time and Amount)
Handling class imbalance using fast downsampling
Train–test split with stratification
Instead of SMOTE, downsampling was used to improve computational efficiency while preserving fraud patterns.

Models Implemented:
Machine Learning Models
Logistic Regression
Decision Tree
Random Forest
Naive Bayes

Deep Learning Model:
Artificial Neural Network (ANN)

Model Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
ROC-AUC

Visualizations:
Bar charts comparing model performance
ROC curve comparison for all models
Confusion matrix heatmaps
Correlation heatmaps
These visualizations help in clearly comparing the strengths and weaknesses of each model.

Results Summary:
Logistic Regression achieved high recall but lower precision
Decision Tree showed balanced performance
Random Forest provided strong overall performance
Neural Network achieved the highest F1-Score and strong ROC-AUC

Conclusion:
This project demonstrates that machine learning models can effectively detect credit card fraud even in highly imbalanced datasets. Tree-based models and neural networks performed better in capturing complex fraud patterns. Proper data preprocessing and evaluation metrics are crucial for reliable fraud detection systems.

Technologies Used:

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
TensorFlow / Keras (Google Colab)