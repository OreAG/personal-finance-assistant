import os
import pandas as pd

#Function to prep categorised transaction data for recommendation generation.
def prepare_transaction_data(df):
    required_columns = ["date", "description", "amount", "predicted_category"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    prepared_df = df.copy()
    prepared_df["date"] = pd.to_datetime(prepared_df["date"], errors="coerce")
    prepared_df["spend_amount"] = prepared_df["amount"].apply(lambda value: abs(value) if value < 0 else 0)

    return prepared_df

#Calculate total spending by predicted transaction category
def calculate_spending_by_category(df):
    spending_df = df[df["predicted_category"] != "income"]

    category_summary = (
        spending_df.groupby("predicted_category")["spend_amount"]
        .sum()
        .reset_index()
        .sort_values(by="spend_amount", ascending=False)
    )

    return category_summary

#generates simple transparent financial recommendations based on category-level spending patterns
def generate_recommendations(df):
    prepared_df = prepare_transaction_data(df)
    category_summary = calculate_spending_by_category(prepared_df)

    total_spending = category_summary["spend_amount"].sum()
    recommendations = []

    if total_spending == 0:
        recommendations.append({
            "recommendation_type": "General",
            "recommendation": "No spending transactions were detected in the uploaded data.",
            "trigger": "The total spending amount was calculated as £0.",
            "action": "Check that the uploaded file contains outgoing transactions."
        })
        return pd.DataFrame(recommendations)

    for _, row in category_summary.iterrows():
        category = row["predicted_category"]
        category_spend = row["spend_amount"]
        category_share = category_spend / total_spending

        if category_share >= 0.30:
            recommendations.append({
                "recommendation_type": "High category spending",
                "recommendation": f"Spending on {category.replace('_', ' ')} is a large share of total spending.",
                "trigger": f"{category.replace('_', ' ').title()} accounts for {category_share:.1%} of total spending.",
                "action": "Review recent transactions in this category and consider setting a monthly limit."
            })

        if category == "eating_out" and category_share >= 0.15:
            recommendations.append({
                "recommendation_type": "Eating out review",
                "recommendation": "Eating out appears to be a notable area of discretionary spending.",
                "trigger": f"Eating out accounts for {category_share:.1%} of total spending.",
                "action": "Consider reducing the number of takeaway or restaurant purchases next month."
            })

        if category == "entertainment" and category_share >= 0.15:
            recommendations.append({
                "recommendation_type": "Entertainment review",
                "recommendation": "Entertainment spending is relatively high for this period.",
                "trigger": f"Entertainment accounts for {category_share:.1%} of total spending.",
                "action": "Review whether any entertainment purchases can be reduced or planned in advance."
            })

        if category == "subscriptions" and category_spend >= 40:
            recommendations.append({
                "recommendation_type": "Subscription review",
                "recommendation": "Subscription spending may be worth reviewing.",
                "trigger": f"Subscription spending totals £{category_spend:.2f} in the uploaded period.",
                "action": "Check for subscriptions that are unused, duplicated, or no longer needed."
            })

        if category == "transport" and category_share >= 0.20:
            recommendations.append({
                "recommendation_type": "Transport spending review",
                "recommendation": "Transport is one of the larger spending categories.",
                "trigger": f"Transport accounts for {category_share:.1%} of total spending.",
                "action": "Review frequent travel costs and consider whether any journeys could be planned differently."
            })

    income_total = prepared_df.loc[prepared_df["predicted_category"] == "income", "amount"].sum()

    if income_total > 0:
        spending_ratio = total_spending / income_total

        if spending_ratio >= 0.80:
            recommendations.append({
                "recommendation_type": "Budget pressure",
                "recommendation": "Spending is high relative to income for this period.",
                "trigger": f"Detected spending is {spending_ratio:.1%} of detected income.",
                "action": "Consider reviewing the highest spending categories and setting a savings target."
            })
        elif spending_ratio <= 0.50:
            recommendations.append({
                "recommendation_type": "Savings opportunity",
                "recommendation": "There may be an opportunity to increase savings for this period.",
                "trigger": f"Detected spending is {spending_ratio:.1%} of detected income.",
                "action": "Consider moving part of the remaining income into savings or an emergency fund."
            })

    if len(recommendations) == 0:
        recommendations.append({
            "recommendation_type": "No major spending concern",
            "recommendation": "No major spending issues were detected using the current rule set.",
            "trigger": "No category exceeded the predefined spending thresholds.",
            "action": "Continue monitoring spending patterns over future periods."
        })

    return pd.DataFrame(recommendations)

#Loads categorised transactions, generates recommendations, and saves the output
def generate_recommendations_from_file(
    input_path="outputs/categorised_transactions.csv",
    output_path="outputs/recommendations.csv"
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found at {input_path}. "
            "Run src/categorisation.py first."
        )

    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(input_path)
    recommendations_df = generate_recommendations(df)
    recommendations_df.to_csv(output_path, index=False)

    print("Recommendations generated successfully.")
    print(f"Recommendations created: {len(recommendations_df)}")
    print("\nRecommendations:")
    print(recommendations_df)

    print(f"\nRecommendations saved to: {output_path}")

if __name__ == "__main__":
    generate_recommendations_from_file()

