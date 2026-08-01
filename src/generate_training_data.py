import random
from datetime import datetime, timedelta

import pandas as pd


random.seed(42)

category_merchants = {
    "groceries": [
        "TESCO STORE LONDON", "SAINSBURYS LOCAL", "ASDA GROCERIES",
        "LIDL GB", "WAITROSE SUPERMARKET", "MORRISONS", "ALDI STORE"
    ],
    "transport": [
        "TFL TRAVEL CHARGE", "UBER TRIP HELP.UBER.COM", "NATIONAL RAIL TICKET",
        "SHELL PETROL STATION", "BP FUEL", "TRAINLINE", "ZIPCAR UK"
    ],
    "subscriptions": [
        "NETFLIX.COM", "SPOTIFY UK", "AMAZON PRIME UK", "DISNEY PLUS",
        "APPLE ICLOUD", "MICROSOFT 365", "AUDIBLE UK"
    ],
    "bills": [
        "EDF ENERGY BILL", "THAMES WATER", "BRITISH GAS",
        "VIRGIN MEDIA BILL", "COUNCIL TAX PAYMENT", "EE MOBILE",
        "O2 MOBILE BILL"
    ],
    "eating_out": [
        "PRET A MANGER", "NANDOS LONDON", "DELIVEROO ORDER",
        "UBER EATS", "STARBUCKS COFFEE", "CAFFE NERO", "WAGAMAMA"
    ],
    "shopping": [
        "H&M CLOTHING", "ZARA ONLINE", "AMAZON MARKETPLACE",
        "BOOTS UK", "JOHN LEWIS ONLINE", "ARGOS", "NEXT RETAIL"
    ],
    "health_fitness": [
        "PUREGYM MEMBERSHIP", "BOOTS PHARMACY", "HOLLAND AND BARRETT",
        "DENTAL PRACTICE", "PHARMACY PRESCRIPTION", "VISION EXPRESS",
        "SPORTS MASSAGE CLINIC"
    ],
    "entertainment": [
        "ODEON CINEMA", "TICKETMASTER UK", "STEAM GAMES",
        "SKY STORE MOVIE", "BOWLING LONDON", "THEATRE TICKETS",
        "GO APE ACTIVITY"
    ],
    "income": [
        "SALARY PAYMENT", "FREELANCE PAYMENT", "INTEREST PAID",
        "REFUND AMAZON", "CASHBACK REWARD", "BONUS PAYMENT",
        "BANK TRANSFER RECEIVED"
    ]
}


amount_ranges = {
    "groceries": (8, 95),
    "transport": (3, 75),
    "subscriptions": (2, 30),
    "bills": (25, 180),
    "eating_out": (4, 65),
    "shopping": (10, 160),
    "health_fitness": (5, 120),
    "entertainment": (5, 110),
    "income": (5, 3000)
}

locations = [
    "LONDON",
    "MANCHESTER",
    "BIRMINGHAM",
    "ONLINE",
    "UK",
    "OXFORD ST",
    "CAMDEN",
    "VICTORIA"
]

def generate_random_date(start_date, end_date):
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    return start_date + timedelta(days=random_days)

#Adding realistic variation to transaction descriptions like reference numbers, locations, symbols, mixed formatting and inconsistent spacing
def add_noise(description):
    noisy_description = description

    if random.random() < 0.65:
        noisy_description += f" {random.choice(locations)}"

    if random.random() < 0.55:
        noisy_description += f" REF {random.randint(1000, 999999)}"

    if random.random() < 0.35:
        noisy_description = noisy_description.replace(" ", "  ")

    if random.random() < 0.25:
        noisy_description = noisy_description + " CARD PAYMENT"

    if random.random() < 0.20:
        noisy_description = noisy_description.replace(" ", " * ")

    if random.random() < 0.20:
        noisy_description = noisy_description.lower()

    return noisy_description


def create_transaction(category, start_date, end_date):
    merchant = random.choice(category_merchants[category])
    min_amount, max_amount = amount_ranges[category]

    amount = round(random.uniform(min_amount, max_amount), 2)

    if category != "income":
        amount = -amount

    transaction_date = generate_random_date(start_date, end_date)

    return {
        "date": transaction_date.strftime("%Y-%m-%d"),
        "description": merchant,
        "amount": amount,
        "category": category
    }

def generate_dataset(rows_per_category=100):
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 6, 30)

    transactions = []

    for category in category_merchants:
        for _ in range(rows_per_category):
            transaction = create_transaction(category, start_date, end_date)
            transactions.append(transaction)

    random.shuffle(transactions)

    df = pd.DataFrame(transactions)
    df.to_csv("data/training_transactions.csv", index=False)

    print("Synthetic training dataset created.")
    print(f"Rows created: {len(df)}")
    print("\nCategory counts:")
    print(df["category"].value_counts())
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    generate_dataset(rows_per_category=100)