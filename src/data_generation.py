import pandas as pd
from sklearn.datasets import make_classification


def generate_dataset(
    n_samples=100000,
    n_features=20,
    n_informative=10,
    random_state=42,
):
    """
    Generate a synthetic classification dataset for distributed ML experiments.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        random_state=random_state,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df
