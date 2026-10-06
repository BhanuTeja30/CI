import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data import get_dataset


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train():
    X, y = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }

    joblib.dump(model, MODEL_DIR / "model.joblib")

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model trained successfully")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train()