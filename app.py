import streamlit as st
import pandas as pd
import plotly.express as px

from src.categorisation import load_transaction_model, categorise_transactions
from src.recommendations import (
    prepare_transaction_data,
    calculate_spending_by_category,
    generate_recommendations
)


st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💷",
    layout="wide"
)


st.title("Personal Finance Assistant")
st.write(
    "Upload a transaction CSV file to categorise spending, view category-level summaries, "
    "and receive transparent rule-based financial recommendations."
)


uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:
    transactions_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded transactions")
    st.dataframe(transactions_df)

    required_columns = ["date", "description", "amount"]

    missing_columns = [
        column for column in required_columns
        if column not in transactions_df.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded file is missing the following required columns: "
            + ", ".join(missing_columns)
        )
    else:
        model = load_transaction_model()
        categorised_df = categorise_transactions(transactions_df, model)

        st.subheader("Categorised transactions")
        st.dataframe(categorised_df)

        prepared_df = prepare_transaction_data(categorised_df)
        category_summary = calculate_spending_by_category(prepared_df)

        st.subheader("Spending by predicted category")
        st.dataframe(category_summary)

        if not category_summary.empty:
            spending_chart = px.bar(
                category_summary,
                x="predicted_category",
                y="spend_amount",
                title="Total spending by predicted category",
                labels={
                    "predicted_category": "Predicted category",
                    "spend_amount": "Spend amount (£)"
                }
            )

            st.plotly_chart(spending_chart, use_container_width=True)

        recommendations_df = generate_recommendations(categorised_df)

        st.subheader("Transparent recommendations")
        st.write(
            "Each recommendation includes the rule trigger and suggested action, "
            "so the user can understand why it was generated."
        )

        st.dataframe(recommendations_df)

        st.download_button(
            label="Download categorised transactions",
            data=categorised_df.to_csv(index=False),
            file_name="categorised_transactions.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download recommendations",
            data=recommendations_df.to_csv(index=False),
            file_name="recommendations.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Upload a CSV file with the columns: date, description and amount. "
        "You can test the app using data/sample_transactions.csv."
    )