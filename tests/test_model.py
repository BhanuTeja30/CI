import joblib
from src.data import get_dataset


def test_model_prediction():
    model = joblib.load("models/model.joblib")
    X, y = get_dataset()

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert all(prediction in [0, 1] for prediction in predictions)