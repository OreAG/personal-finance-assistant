import pandas as pd

from src.categorisation import clean_description, categorise_transactions


class DummyModel:
    def predict(self, descriptions):
        return ["shopping" for _ in descriptions]


def test_clean_description_removes_special_characters():
    result = clean_description("Amazon Marketplace *1234 LONDON!!")

    assert result == "amazon marketplace 1234 london"


def test_categorise_transactions_adds_predicted_category():
    sample_df = pd.DataFrame({
        "date": ["2026-01-01"],
        "description": ["Amazon Marketplace"],
        "amount": [-35.99]
    })

    model = DummyModel()
    result_df = categorise_transactions(sample_df, model)

    assert "predicted_category" in result_df.columns
    assert result_df.loc[0, "predicted_category"] == "shopping"