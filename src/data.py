import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mushroom.csv"

COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]


def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

    y = df["class"].map({"e": 0, "p": 1})
    X = df.drop(columns=["class"])

    X = pd.get_dummies(X)

    return X, y


def get_dataset():
    return load_data()