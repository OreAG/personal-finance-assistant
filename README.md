# ML-Enhanced Personal Finance Assistant

This repository contains the source code for an MSc Data Science project: **ML-Enhanced Personal Finance Assistant: Transaction Categorisation and Transparent Rule-Based Budgeting Prompts**.

The prototype supports budgeting and expense tracking by combining supervised transaction categorisation with transparent rule-based budgeting prompts. Users can upload transaction data, view predicted categories, review model confidence scores, manually correct categories, view spending summaries and download outputs.

## Project overview

The prototype was developed as a functional proof of concept rather than a production-ready financial application. It does not connect to live bank accounts, use open banking APIs or provide regulated financial advice.

The system includes:

- synthetic transaction data generation
- supervised transaction categorisation
- TF-IDF feature extraction
- Logistic Regression and calibrated Linear SVM model comparison
- Streamlit user interface
- manual category correction
- rule-based budgeting prompts
- internal, challenge and external dataset evaluation
- automated tests using pytest

## Repository link

https://github.com/OreAG/personal-finance-assistant

## Project structure

```text
personal-finance-assistant/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── outputs/
├── src/
│   ├── categorisation.py
│   ├── generate_training_data.py
│   ├── model_training.py
│   ├── recommendations.py
│   ├── evaluate_challenge_set.py
│   ├── evaluate_external_dataset.py
│   └── create_external_confusion_matrix.py
└── tests/
    ├── conftest.py
    ├── test_categorisation.py
    └── test_recommendations.py
```

## Requirements

The project was developed using Python. The main dependencies are listed in `requirements.txt` and include:

- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- joblib
- pytest

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/OreAG/personal-finance-assistant.git
cd personal-finance-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected CSV input format

The Streamlit application expects an uploaded CSV file with at least the following columns:

| Column | Description |
|---|---|
| `date` | Transaction date |
| `description` | Transaction description used for categorisation |
| `amount` | Transaction amount |

Spending transactions should be represented as negative values and income transactions as positive values.

Example:

```csv
date,description,amount
2026-01-05,TESCO EXPRESS,-24.50
2026-01-07,TRAINLINE TICKET,-18.90
2026-01-31,EMPLOYER PAYROLL,2500.00
```

## Generating synthetic training data

To generate the synthetic transaction dataset, run:

```bash
python -m src.generate_training_data
```

This creates synthetic labelled transaction records across nine project categories:

- bills
- eating_out
- entertainment
- groceries
- health_fitness
- income
- shopping
- subscriptions
- transport

## Training the model

To train the transaction categorisation model, run:

```bash
python -m src.model_training
```

The training script compares two supervised machine learning pipelines:

1. TF-IDF + Logistic Regression
2. TF-IDF + Calibrated Linear Support Vector Machine

The synthetic labelled dataset is split using an 80/20 stratified train/test split with `random_state=42`.

The selected model is saved as:

```text
models/transaction_classifier.pkl
```

If the model file is not present, run the training script before launching the Streamlit application.

## Running the Streamlit application

To start the app, run:

```bash
streamlit run app.py
```

The application allows users to:

- upload a CSV transaction file
- view predicted transaction categories
- inspect model confidence scores
- review low-confidence predictions
- manually correct categories
- view spending summaries
- generate rule-based budgeting prompts
- download categorised transactions and recommendation outputs

## Running evaluation scripts

To evaluate the saved model on the challenge dataset:

```bash
python -m src.evaluate_challenge_set
```

To evaluate the saved model on the external labelled Kaggle dataset:

```bash
python -m src.evaluate_external_dataset
```

To generate the external dataset confusion matrix:

```bash
python -m src.create_external_confusion_matrix
```

Evaluation outputs are saved in the `outputs/` folder.

## Running automated tests

To run the automated tests:

```bash
pytest
```

The tests check core categorisation and recommendation functions, including:

- transaction description cleaning
- categorisation output structure
- spending amount calculation
- income exclusion from spending summaries
- subscription recommendation triggering

## External evaluation dataset

The external evaluation used the Kaggle dataset **Personal Finance** by Lauhith / entrepreneurlife, downloaded on 03 August 2026. The dataset was used only for external evaluation and was not included in model training.

The original dataset categories were manually mapped into the project’s nine-category taxonomy before evaluation. This mapping introduced subjectivity and is discussed as a limitation in the project report.

## Known limitations

This prototype has several limitations:

- It was built as an MSc prototype, not a production-ready financial application.
- It does not connect to live bank accounts or open banking APIs.
- Uploaded transaction files may still contain sensitive financial information.
- Model performance is limited outside the synthetic training distribution.
- Confidence scores are used as review heuristics rather than independently validated probability estimates.
- Recommendation thresholds are heuristic and were not validated through user testing or financial expert review.
- Manual corrections are used within the current workflow and exported outputs; they are not automatically stored for future model retraining.
- The system always predicts one of the predefined categories, even where a transaction may not fit the taxonomy well.

## Academic context

This project was completed as part of an MSc Data Science final project. The report evaluates the prototype across internal synthetic testing, challenge dataset testing and external labelled dataset testing. The external evaluation showed lower generalisation performance, which is discussed as a key limitation of training with synthetic transaction data.