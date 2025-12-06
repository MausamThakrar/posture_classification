import pandas as pd
from sklearn.cluster import DBSCAN


def remove_outliers_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 3) -> pd.DataFrame:
    """
    Remove outliers using DBSCAN clustering on feature space.
    Assumes target column is named 'label'.
    """
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataframe.")

    features = df.drop(columns=["label"])
    labels = df["label"]

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(features)

    mask = db.labels_ != -1  # keep non-outliers
    cleaned = df[mask].reset_index(drop=True)
    return cleaned
