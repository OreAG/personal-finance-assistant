import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path

# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
predictions_path = PROJECT_ROOT / "outputs" / "external_predictions.csv"
output_path = PROJECT_ROOT / "outputs" / "external_confusion_matrix.png"

# Load predictions
df = pd.read_csv(predictions_path)

# Check expected columns exist
required_columns = {"category", "predicted_category"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Define category order
categories = [
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

# Create confusion matrix
cm = confusion_matrix(
    df["category"],
    df["predicted_category"],
    labels=categories
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=categories
)

display.plot(ax=ax, xticks_rotation=45, values_format="d")

ax.set_title("External Kaggle Dataset Confusion Matrix")
ax.set_xlabel("Predicted category")
ax.set_ylabel("True category")

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Confusion matrix saved to: {output_path}")