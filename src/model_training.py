import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

#Clean transaction description text so it is easier for the model to process.
def clean_description(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


#Load labelled transaction data and train and evaluate supervise ml classification model and save results
def train_transaction_classifier():
    # Load the training dataset
    df = pd.read_csv("data/training_transactions.csv")

    # Check that the expected columns exist
    required_columns = ["date", "description", "amount", "category"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Remove rows with missing descriptions or categories
    df = df.dropna(subset=["description", "category"])

    # Create a cleaned version of the transaction description
    df["clean_description"] = df["description"].apply(clean_description)

    # Define input feature and target label
    x = df["clean_description"]
    y = df["category"]

    # Split dataset into training and testing subsets
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create a machine learning pipeline:TF-IDF converts text into numerical features.Logistic Regression performs supervised classification.
    models= {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "TF-IDF + Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", CalibratedClassifierCV(LinearSVC()))
        ])
    }

    metrics = []
    classification_reports = {}
    trained_models = {}

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(x_train, y_train)

        y_pred = model_pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_rows": len(x_train),
            "testing_rows": len(x_test)
        })

        classification_reports[model_name] = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        trained_models[model_name] = model_pipeline

    # Create output folders if they do not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df .to_csv("outputs/model_metrics.csv", index=False)

    #Select the best model using F1-Score
    best_model_name = metrics_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]["model"]

    best_model = trained_models[best_model_name]

    #Save the best trained model for app use
    joblib.dump(best_model, "models/transaction_classifier.pkl")

    #Save detailed classification reports for all models
    with open("outputs/classification_report.txt", "w") as file:
        for model_name, report in classification_reports.items():
            file.write(f"{model_name}\n")
            file.write("=" * len(model_name))
            file.write("\n")
            file.write(report)
            file.write("\n\n")

    # Print results to terminal
    print("Model training complete.")

    print("\nEvaluation metrics:")
    print(metrics_df)

    print(f"\nBest model saved for app use: {best_model_name}")
    print("Model saved to:")
    print("models/transaction_classifier.pkl")

    print("\nClassification reports saved to:")
    print("outputs/classification_report.txt")

    print("\nMetrics saved to:")
    print("outputs/model_metrics.csv")

if __name__ == "__main__":
    train_transaction_classifier()