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

#Evaluates the trained transaction categorisation model on a separate labelled challenge dataset.
"""

    This dataset is not used for training. It is used to test whether the
    model can generalise to transaction descriptions that are different from
    the synthetic training data.
    """

def evaluate_challenge_set(
    input_path="data/challenge_transactions.csv",
    predictions_output_path="outputs/challenge_predictions.csv",
    metrics_output_path="outputs/challenge_evaluation_metrics.csv",
    report_output_path="outputs/challenge_classification_report.txt"
):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Challenge dataset not found at {input_path}")

    os.makedirs("outputs", exist_ok=True)

    # Load trained model and challenge dataset
    model = load_transaction_model()
    challenge_df = pd.read_csv(input_path)

    required_columns = ["date", "description", "amount", "category"]

    for column in required_columns:
        if column not in challenge_df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Predict categories
    predictions_df = categorise_transactions(challenge_df, model)

    # Compare true labels against predicted labels
    y_true = predictions_df["category"]
    y_pred = predictions_df["predicted_category"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics_df = pd.DataFrame([
        {
            "dataset": "challenge_transactions",
            "model": "saved_best_model",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "rows_evaluated": len(predictions_df)
        }
    ])

    # Add an error flag for analysis
    predictions_df["correct_prediction"] = (
        predictions_df["category"] == predictions_df["predicted_category"]
    )

    # Save outputs
    predictions_df.to_csv(predictions_output_path, index=False)
    metrics_df.to_csv(metrics_output_path, index=False)

    report = classification_report(y_true, y_pred, zero_division=0)

    with open(report_output_path, "w") as file:
        file.write(report)

    # Print results
    print("Challenge set evaluation complete.")

    print("\nEvaluation metrics:")
    print(metrics_df)

    print("\nClassification report:")
    print(report)

    print("\nPrediction accuracy counts:")
    print(predictions_df["correct_prediction"].value_counts())

    print("\nIncorrect predictions:")
    incorrect_predictions = predictions_df[predictions_df["correct_prediction"] == False]

    if len(incorrect_predictions) == 0:
        print("No incorrect predictions found.")
    else:
        print(
            incorrect_predictions[
                [
                    "date",
                    "description",
                    "amount",
                    "category",
                    "predicted_category",
                    "prediction_confidence"
                ]
            ]
        )

    print(f"\nPredictions saved to: {predictions_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Classification report saved to: {report_output_path}")


if __name__ == "__main__":
    evaluate_challenge_set()