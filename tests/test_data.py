from src.data import get_dataset


def test_dataset_loads():
    X, y = get_dataset()

    assert X.shape[0] > 0
    assert X.shape[1] > 0
    assert len(y) == len(X)


def test_target_values():
    X, y = get_dataset()

    assert set(y.dropna().unique()).issubset({0, 1})