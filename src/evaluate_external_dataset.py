import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.categorisation import load_transaction_model, categorise_transactions


VALID_PROJECT_CATEGORIES = [
    "bills",
    "eating_out",
    "entertainment",
    "groceries",
    "health_fitness",
    "income",
    "shopping",
    "subscriptions",
    "transport"
]

#Evaluates the trained transaction categorisation model on an external labelled dataset.

def evaluate_external_dataset(
    input_path= r"C:\Users\ore_a\PycharmProjects\personal-finance-assistant\data\external_transactions_labelled.csv.csv",
    predictions_output_path="outputs/external_predictions.csv",
    metrics_output_path="outputs/external_evaluation_metrics.csv",
    report_output_path="outputs/external_classification_report.txt"
):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"External dataset not found at {input_path}")

    os.makedirs("outputs", exist_ok=True)

    model = load_transaction_model()
    external_df = pd.read_csv(input_path)

    required_columns = ["date", "description", "amount", "category"]

    for column in required_columns:
        if column not in external_df.columns:
            raise ValueError(
                f"Missing required column: {column}. "
                f"Current columns are: {list(external_df.columns)}"
            )

    # Standardise mapped category labels
    external_df["category"] = (
        external_df["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove rows with blank, invalid or excluded categories
    external_df = external_df[
        external_df["category"].isin(VALID_PROJECT_CATEGORIES)
    ].copy()

    if len(external_df) == 0:
        raise ValueError("No valid mapped categories found in the external dataset.")

    # Predict categories using the trained model
    predictions_df = categorise_transactions(external_df, model)

    y_true = predictions_df["category"]
    y_pred = predictions_df["predicted_category"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    predictions_df["correct_prediction"] = (
        predictions_df["category"] == predictions_df["predicted_category"]
    )

    metrics_df = pd.DataFrame([
        {
            "dataset": "external_kaggle_dataset",
            "model": "saved_best_model",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "rows_evaluated": len(predictions_df)
        }
    ])

    report = classification_report(y_true, y_pred, zero_division=0)

    predictions_df.to_csv(predictions_output_path, index=False)
    metrics_df.to_csv(metrics_output_path, index=False)

    with open(report_output_path, "w") as file:
        file.write(report)

    print("External dataset evaluation complete.")

    print("\nEvaluation metrics:")
    print(metrics_df)

    print("\nClassification report:")
    print(report)

    print("\nPrediction accuracy counts:")
    print(predictions_df["correct_prediction"].value_counts())

    incorrect_predictions = predictions_df[
        predictions_df["correct_prediction"] == False
    ]

    print("\nIncorrect predictions:")

    if len(incorrect_predictions) == 0:
        print("No incorrect predictions found.")
    else:
        columns_to_show = [
            "date",
            "description",
            "amount",
            "category",
            "predicted_category",
            "prediction_confidence"
        ]

        if "source_category" in incorrect_predictions.columns:
            columns_to_show.insert(4, "source_category")

        print(incorrect_predictions[columns_to_show])

    print(f"\nPredictions saved to: {predictions_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Classification report saved to: {report_output_path}")


if __name__ == "__main__":
    evaluate_external_dataset()