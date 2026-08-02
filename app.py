import os
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

        categorised_display_df = categorised_df.copy()

        if "prediction_confidence" in categorised_display_df.columns:
            categorised_display_df["prediction_confidence"] = categorised_display_df[
                "prediction_confidence"
            ].apply(lambda value: f"{value:.0%}" if pd.notna(value) else "N/A")

        st.dataframe(categorised_display_df)

        st.subheader("Low-confidence predictions for review")

        confidence_threshold_percent = st.slider(
            "Select confidence threshold",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            format="%d%%"
        )

        confidence_threshold = confidence_threshold_percent/100

        if "prediction_confidence" in categorised_df.columns:
            low_confidence_df = categorised_df[
                categorised_df["prediction_confidence"] < confidence_threshold
            ]
            st.write(
                f"Transactions below {confidence_threshold_percent}% confidence: "
                f"{len(low_confidence_df)}"
            )

            if len(low_confidence_df) >0:
                low_confidence_display_df = low_confidence_df.copy()

                low_confidence_display_df["prediction_confidence"] = low_confidence_display_df[
                    "prediction_confidence"
                ].apply(
                    lambda value: f"{value:.0%}" if pd.notna(value) else "N/A"
                )

                st.dataframe(low_confidence_display_df)


                st.download_button(
                    label="Download low-confidence transactions",
                    data=low_confidence_df.to_csv(index=False),
                    file_name="low_confidence_transactions.csv",
                    mime="text/csv"
                )
            else:
                st.success("No low-confidence transactions detected at the selected threshold.")
        else:
            st.warning("Prediction confidence is not available for this model.")

        st.subheader("Review and correct categories")

        st.write(
            "The predicted categories can be reviewed and corrected before the spending "
            "summary and recommendations are generated. This supports the feedback loop "
            "in the system design."
        )

        category_options = [
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

        review_df = categorised_df[
            ["date", "description", "amount", "predicted_category", "prediction_confidence"]
        ].copy()

        review_df["prediction_confidence"] = review_df["prediction_confidence"].apply(
            lambda value: f"{value:.0%}" if pd.notna(value) else "N/A"
        )

        review_df["final_category"] = categorised_df["predicted_category"]

        edited_review_df = st.data_editor(
            review_df,
            column_config={
                "final_category": st.column_config.SelectboxColumn(
                    "Final category",
                    options=category_options,
                    required=True
                )
            },
            disabled=[
                "date",
                "description",
                "amount",
                "predicted_category",
                "prediction_confidence"
            ],
            use_container_width=True,
            key="category_review_editor"
        )

        final_df = categorised_df.copy()
        final_df["final_category"] = edited_review_df["final_category"].values
        final_df["predicted_category"] = final_df["final_category"]

        prepared_df = prepare_transaction_data(final_df)
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

        recommendations_df = generate_recommendations(final_df)

        st.subheader("Recommendations")
        st.write(
            "Each recommendation includes the rule trigger and suggested action, "
            "so the user can understand why it was generated."
        )

        st.dataframe(recommendations_df)

        st.subheader("Model evaluation")

        metrics_path = "outputs/model_metrics.csv"
        report_path = "outputs/classification_report.txt"

        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            st.write("Summary of the trained transaction categorisation model:")
            st.dataframe(metrics_df)
        else:
            st.warning("Model metrics file not found. Run src/model_training.py to generate metrics.")

        if os.path.exists(report_path):
            with open(report_path, "r") as file:
                classification_report_text = file.read()

            st.text("Detailed classification report:")
            st.code(classification_report_text)
        else:
            st.warning("Classification report file not found. Run src/model_training.py to generate the report.")

        st.download_button(
            label="Download categorised transactions",
            data=final_df.to_csv(index=False),
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