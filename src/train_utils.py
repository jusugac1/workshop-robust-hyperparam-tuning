# =================
# ==== IMPORTS ====
# =================

import pandas as pd
import optuna
import yaml
from typing import Any, Literal


# ===================
# ==== FUNCTIONS ====
# ===================

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


def build_search_space(
    trial: optuna.trial.Trial,
    hyperparams_search_space: dict[str, Any],
) -> dict[str, Any]:
    """Build the search space for hyperparameter optimization based on the provided sampling parameters.

    Args:
        trial (Trial): The Optuna trial object used for sampling hyperparameters.
        hyperparams_search_space (dict[str, Any]): The search space for hyperparameter tuning.

    Returns:
        dict[str, Any]: A dictionary containing the sampled hyperparameters for the current trial.
    """
    hyperparams = {}

    for hyparam_name, sampling_params in hyperparams_search_space.items():
        if sampling_params["sampling_type"] == "categorical":
            hyperparams[hyparam_name] = eval(  # noqa: S307
                f"trial.suggest_{sampling_params['sampling_type']}('{hyparam_name}', {sampling_params['choices']})"
            )
        else:
            hyperparams[hyparam_name] = eval(  # noqa: S307
                f"trial.suggest_{sampling_params['sampling_type']}('{hyparam_name}', {sampling_params['min']}, {sampling_params['max']})"
            )
    return hyperparams


def load_conf_parameters(conf_file: str) -> dict[str, Any]:
    """Load configuration parameters from a YAML file.

    Args:
        conf_file (str): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: A dictionary containing the configuration parameters.
    """
    with open(conf_file, "r") as file:
        params = yaml.safe_load(file)
    return params
