# Problem 4 – Sports vs Politics Text Classification

## Overview
This project implements a machine learning–based text classification system that automatically categorizes news documents into **Sports** or **Politics**.  
Multiple feature representation techniques and machine learning models were evaluated and compared quantitatively.

---

## Dataset
A custom dataset was prepared consisting of short news-style paragraphs:

- **Classes**:
  - Sports
  - Politics
- **Documents**: 60 total (30 per class)
- **Files**:
  - `data/sports.txt`
  - `data/politics.txt`
- Each line in the file represents one document.

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
| Naive Bayes | BoW | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | BoW | 0.92 | 0.93 | 0.92 | 0.91 |
| SVM | BoW | 1.00 | 1.00 | 1.00 | 1.00 |
| Naive Bayes | TF-IDF | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | TF-IDF | 0.92 | 0.93 | 0.92 | 0.92 |
| SVM | TF-IDF | 1.00 | 1.00 | 1.00 | 1.00 |
| Naive Bayes | TF-IDF + Bigrams | 0.92 | 0.93 | 0.92 | 0.92 |
| Logistic Regression | TF-IDF + Bigrams | 0.83 | 0.89 | 0.83 | 0.84 |
| SVM | TF-IDF + Bigrams | 0.92 | 0.93 | 0.92 | 0.90 |

A visual comparison of model accuracy is available in results/accuracy_comparison.png
