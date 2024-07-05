# Sentiment Analysis and Recommendation Index Prediction for Women's Clothing

## 1.0 Abstract
This study integrates sentiment polarity with preprocessed customer reviews to predict recommendation indices for women's clothing in e-commerce stores. Sentiment polarity was extracted using VADER. Reviews were vectorized using TF-IDF and analyzed with machine learning models (Random Forest, XGBoost) and deep learning models (LSTM, BERT). Extensive fine-tuning was applied to optimize performance. XGBoost and LSTM achieved the highest accuracy (88%) and ROC-AUC (91%).

## 2.0 Introduction
Online shopping has surged, with global e-commerce spending reaching $6.31 trillion in 2023. Traditional e-commerce recommendation systems often fail to incorporate user reviews and sentiments, resulting in recommendations that may not align with user satisfaction. This project aims to improve recommendation quality by predicting a product's recommendation index from customer reviews using sentiment analysis and machine learning techniques.

## 3.0 Project Plan

| Phase                    | Timeline | Milestones                                                                  |
|--------------------------|----------|-----------------------------------------------------------------------------|
| Data Collection          | Week 1   | Collect dataset from Kaggle and assess its structure                        |
| Data Preprocessing       | Week 2-3 | Clean and preprocess data, handle missing values, normalize text            |
| Exploratory Data Analysis| Week 4   | Generate word clouds and count plots to visualize data                      |
| Feature Engineering      | Week 5   | Convert text to TF-IDF matrix, compute sentiment scores                     |
| Baseline Model           | Week 6   | Train and evaluate Gaussian Naive Bayes model                               |
| Traditional ML Models    | Week 7-8 | Train and evaluate Random Forest and XGBoost models                         |
| Deep Learning Models     | Week 9-10| Train and evaluate LSTM and BERT models                                     |
| Model Fine-Tuning        | Week 11  | Apply hyperparameter tuning, data augmentation, regularization               |
| Final Evaluation         | Week 12  | Compare models, finalize best-performing models                             |
| Visualization & Reporting| Week 13  | Generate plots, confusion matrix, and final report                          |

## 4.0 Methodology

### Dataset
- **Source:** Kaggle
- **Size:** 3 MB
- **Composition:** 23,486 e-commerce reviews with 10 columns
- **Imbalance:** Addressed using SMOTE and Class Weight Strategy

### Data Preparation
- **Loading Data:** Removal of unnecessary columns and filling missing data
- **Text Preprocessing:** Cleaning, normalizing, lemmatizing, and removing stopwords and punctuation

### Exploratory Data Analysis
- **Visualizations:** Word clouds and count plots for recommendation indices

### Baseline Model
- **Model:** Gaussian Naive Bayes
- **Evaluation Metrics:** Accuracy, ROC-AUC, Precision, F1-Score

### Traditional Machine Learning Methods
- **Selected Methods:** Random Forest, XGBoost
- **Justifications:** Robustness, computational efficiency, handling imbalanced data

### Deep Learning Methods
- **Selected Methods:** LSTM, BERT
- **Justifications:** Handling long-range dependencies, bidirectional context understanding

### Implementation
- **Libraries:** Pandas, Matplotlib, Seaborn, NLTK, TextBlob, Spacy, Scikit-learn, Tensorflow, Keras, Transformers
- **Feature Engineering:** TF-IDF matrix and sentiment scores
- **Model Building and Training:** Random Forest, XGBoost, LSTM, BERT
- **Model Evaluation:** Classification reports, confusion matrix, ROC-AUC scores

## 5.0 Results

### Visualizations
- **Word Clouds:** Most frequent terms within the reviews
- **Count Plot:** Distribution of recommendation indices

### Model Evaluation
- **Random Forest:**
  - **Best Configuration:** Grid search optimization
  - **Metrics:** Precision (0.86), Recall (0.87), F1-Score (0.86), Accuracy (0.87), ROC-AUC (0.90)

- **XGBoost:**
  - **Best Configuration:** Grid search optimal parameters
  - **Metrics:** Precision (0.87), Recall (0.88), F1-Score (0.87), Accuracy (0.88), ROC-AUC (0.91)

- **LSTM:**
  - **Best Configuration:** Batch normalization and multiple layers
  - **Metrics:** ROC AUC Score (0.91), Accuracy (0.88)

- **BERT:**
  - **Best Configuration:** Regularization techniques (L2 = 0.01, Dropout = 0.3)
  - **Metrics:** Accuracy (0.82), ROC AUC (0.87)

## 6.0 Conclusion
XGBoost and LSTM models provided the best performance with an accuracy and ROC-AUC score of 88% and 91%, respectively. This project demonstrated the effectiveness of integrating sentiment polarity with customer reviews for enhancing recommendation systems in e-commerce.
