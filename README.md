# Sports vs Politics Text Classification

## Project Overview
This project focuses on building a machine-learning-based text classification system that automatically categorizes news documents into **Sports** or **Politics** domains. The objective is to study how different feature representations and classification algorithms influence performance on a binary text classification task.

The project compares multiple **feature extraction techniques** and **machine learning models**, followed by quantitative evaluation using standard performance metrics.

---

## Dataset
A custom dataset was created for this task consisting of short news-style paragraphs collected and paraphrased from common sports and political themes.

- Total documents: **60**
- Classes:
  - **Sports** (30 documents)
  - **Politics** (30 documents)
- Each document contains **2–4 sentences**
- Dataset files:
  - `data/sports.txt`
  - `data/politics.txt`

Each line in the file represents one document.

---

## Feature Representations
The following text representation techniques were evaluated:

- Bag of Words (BoW)
- TF-IDF (Unigrams)
- TF-IDF with Bigrams

---

## Machine Learning Models
Three classification algorithms were implemented and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (Linear SVM)

---

## Experimental Setup
- Dataset split: **80% training / 20% testing**
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## Results Summary
The experiments show that both **Naive Bayes** and **SVM** perform strongly across most feature representations, while **Logistic Regression** shows slightly lower performance when feature dimensionality increases (e.g., bigrams).

A visualization comparing accuracy across models and feature representations is available in:

