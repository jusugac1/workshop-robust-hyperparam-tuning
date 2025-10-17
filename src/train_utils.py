import pandas as pd
from typing import Literal


def retrieve_data_w_features(
    df: pd.DataFrame,
    features_to_drop: list[str],
    split: Literal["train_set", "test_set", "val_set", "big_train_set"],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retrieve the features and target variable from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing features and target variable.
        features_to_drop (list[str]): List of feature names to drop from the DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing the features after dropping specified columns.
        pd.Series: Series containing the target variable.
    """

    df_filtered = df[df[split] == 1].copy(deep=True)
    # drop requested features and the target column from X (ignore missing columns)
    X = df_filtered.drop(columns=features_to_drop + ["PremTot"], errors="raise")
    y = df_filtered["PremTot"]
    return X, y
