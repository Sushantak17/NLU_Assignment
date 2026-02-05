import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Loading documents from sports and politics files
def load_data(sports_file, politics_file):
    texts = []
    labels = []

    with open(sports_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(line.strip())
                labels.append("sport")

    with open(politics_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(line.strip())
                labels.append("politics")

    return texts, labels


# ---------------- FEATURE REPRESENTATIONS ----------------
def get_vectorizers():
    return {
        "BoW": CountVectorizer(),
        "TF-IDF": TfidfVectorizer(),
        "TF-IDF + Bigrams": TfidfVectorizer(ngram_range=(1, 2))
    }


# ---------------- MODELS ----------------
def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }


# ---------------- MODEL EVALUATION ----------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted"
    )

    return accuracy, precision, recall, f1


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sports_path = os.path.join(BASE_DIR, "data", "sports.txt")
    politics_path = os.path.join(BASE_DIR, "data", "politics.txt")

    texts, labels = load_data(sports_path, politics_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vectorizers = get_vectorizers()
    models = get_models()

    accuracy_results = {key: [] for key in vectorizers}

    print("\n=== Experimental Results ===\n")

    for vec_name, vectorizer in vectorizers.items():
        print(f"\nFeature Representation: {vec_name}")

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for model_name, model in models.items():
            acc, prec, rec, f1 = evaluate_model(
                model, X_train_vec, X_test_vec, y_train, y_test
            )

            print(
                f"{model_name}: "
                f"Accuracy={acc:.2f}, "
                f"Precision={prec:.2f}, "
                f"Recall={rec:.2f}, "
                f"F1={f1:.2f}"
            )

            accuracy_results[vec_name].append(acc)

    # ---------------- PLOTTING ----------------
    model_names = list(models.keys())
    x = np.arange(len(model_names))
    width = 0.25

    plt.figure()

    plt.bar(x - width, accuracy_results["BoW"], width, label="BoW")
    plt.bar(x, accuracy_results["TF-IDF"], width, label="TF-IDF")
    plt.bar(x + width, accuracy_results["TF-IDF + Bigrams"], width, label="TF-IDF + Bigrams")

    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Models")
    plt.xticks(x, model_names)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
    plt.show()
