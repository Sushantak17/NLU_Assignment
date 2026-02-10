# Problem 4 – Sports vs Politics Text Classification

## Overview
This project implements a machine learning–based text classification system that automatically categorizes news documents into **Sports** or **Politics**.  
Multiple feature representation techniques and machine learning models were evaluated and compared quantitatively.

---

## Dataset
The dataset used in this project was derived from the publicly available **Kaggle News Category Dataset**.  
Articles belonging to the **Sports** and **Politics** categories were filtered using a preprocessing script and converted into plain text format for experimentation.

- **Classes**:
  - Sports
  - Politics
- **Documents**: Approximately 2000 total (about 1000 per class)
- **Files**:
  - `data/sports.txt`
  - `data/politics.txt`
- Each line in the file represents one document.
- The preprocessing script used for dataset preparation is:
  - `prepare_dataset.py`


---

## Feature Representations
The following feature extraction methods were evaluated:

- Bag of Words (BoW)
- TF-IDF (Unigrams)
- TF-IDF + Bigrams (n-grams)

---

## Machine Learning Models
The following models were implemented and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear SVM)

---

## Experimental Results

| Model | Feature Representation | Accuracy | Precision | Recall | F1 Score |
|------|------------------------|----------|-----------|--------|----------|
| Naive Bayes | BoW | 0.96 | 0.96 | 0.96 | 0.96 |
| Logistic Regression | BoW | 0.94 | 0.94 | 0.94 | 0.94 |
| SVM | BoW | 0.94 | 0.95 | 0.94 | 0.94 |
| Naive Bayes | TF-IDF | 0.95 | 0.96 | 0.95 | 0.95 |
| Logistic Regression | TF-IDF | 0.95 | 0.95 | 0.95 | 0.95 |
| SVM | TF-IDF | 0.96 | 0.96 | 0.96 | 0.96 |
| Naive Bayes | TF-IDF + Bigrams | 0.95 | 0.96 | 0.95 | 0.95 |
| Logistic Regression | TF-IDF + Bigrams | 0.95 | 0.96 | 0.95 | 0.95 |
| SVM | TF-IDF + Bigrams | 0.96 | 0.96 | 0.96 | 0.96 |


A visual comparison of model accuracy is available in results/accuracy_comparison.png
