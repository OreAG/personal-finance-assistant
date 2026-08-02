import os
import re
import joblib
import pandas as pd

#Cleans transaction description text before prediction
def clean_description(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Loads the trained transaction categorisation model
def load_transaction_model(model_path="models/transaction_classifier.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run src/model_training.py first to train and save the model."
        )

    return joblib.load(model_path)

#takes a transaction df and predicts categories using the model
def categorise_transactions(df, model):
    if "description" not in df.columns:
        raise ValueError("The uploaded transaction file must contain a 'description' column.")

    categorised_df = df.copy()
    categorised_df["clean_description"] = categorised_df["description"].apply(clean_description)
    categorised_df["predicted_category"] = model.predict(categorised_df["clean_description"])

    if hasattr(model, "predict_proba"):
        prediction_probabilities = model.predict_proba(categorised_df["clean_description"])
        categorised_df["prediction_confidence"] = prediction_probabilities.max(axis=1)
    else:
        categorised_df["prediction_confidence"] = None

    return categorised_df

#Loads the sample transaction file, categorises the transactions and saves categorised output.
def categorise_sample_file(
    input_path="data/sample_transactions.csv",
    output_path="outputs/categorised_transactions.csv"
):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}.")

    os.makedirs("outputs", exist_ok=True)

    model = load_transaction_model()
    df = pd.read_csv(input_path)

    categorised_df = categorise_transactions(df, model)
    categorised_df.to_csv(output_path, index=False)

    print("Sample transactions categorised successfully.")
    print(f"Rows categorised: {len(categorised_df)}")
    print("\nFirst five categorised rows:")
    print(categorised_df.head())

    print("\nPredicted category counts:")
    print(categorised_df["predicted_category"].value_counts())

    print(f"\nCategorised file saved to: {output_path}")


if __name__ == "__main__":
    categorise_sample_file()