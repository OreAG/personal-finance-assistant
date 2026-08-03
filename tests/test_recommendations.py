import pandas as pd

from src.recommendations import (
    prepare_transaction_data,
    calculate_spending_by_category,
    generate_recommendations
)


def test_prepare_transaction_data_adds_spend_amount():
    sample_df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Tesco", "Salary Payment"],
        "amount": [-25.50, 2000.00],
        "predicted_category": ["groceries", "income"]
    })

    result_df = prepare_transaction_data(sample_df)

    assert "spend_amount" in result_df.columns
    assert result_df.loc[0, "spend_amount"] == 25.50
    assert result_df.loc[1, "spend_amount"] == 0


def test_calculate_spending_by_category_excludes_income():
    sample_df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "description": ["Tesco", "Salary Payment"],
        "amount": [-25.50, 2000.00],
        "predicted_category": ["groceries", "income"]
    })

    prepared_df = prepare_transaction_data(sample_df)
    result_df = calculate_spending_by_category(prepared_df)

    assert "income" not in result_df["predicted_category"].values
    assert result_df.loc[0, "predicted_category"] == "groceries"
    assert result_df.loc[0, "spend_amount"] == 25.50


def test_generate_recommendations_identifies_subscription_review():
    sample_df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "description": ["Netflix", "Spotify", "Salary Payment"],
        "amount": [-25.00, -30.00, 2000.00],
        "predicted_category": ["subscriptions", "subscriptions", "income"]
    })

    recommendations_df = generate_recommendations(sample_df)

    assert "Subscription review" in recommendations_df["recommendation_type"].values