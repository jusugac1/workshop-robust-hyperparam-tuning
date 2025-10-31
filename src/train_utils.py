# =================
# ==== IMPORTS ====
# =================

import pandas as pd
import optuna
import yaml
from typing import Any, Literal, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from optuna.samplers import TPESampler
from functools import partial


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


def optimize_hyperparams_hgb(
    trial: optuna.trial.Trial,
    search_params: dict[str, Any],
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: list[str],
) -> float:
    """Optimize the hyperparameters of the HistGradientBoosting model.
    
    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        search_params (dict[str, Any]): The search space parameters.
        df_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        df_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target.
        categorical_features (list[str]): List of categorical feature names.

    Returns:
        (float): The validation loss.
    """
    # Build search space
    hyperparams = build_search_space(trial, search_params)

    # Define the model
    model = HistGradientBoostingRegressor(
        categorical_features=categorical_features,
        early_stopping=True,
        random_state=42,
        **hyperparams,
    )
    model.fit(X=df_train, y=y_train, X_val=df_val, y_val=y_val)
    val_predictions = model.predict(df_val)
    return root_mean_squared_error(y_true=y_val, y_pred=val_predictions)


def run_bayesian_optimization(
    df_train: pd.DataFrame,
    y_train: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: list[str],
    search_params: dict[str, dict[str, Any]],
    default_params_list: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run the Bayesian optimization process using Optuna.

    Args:
        df_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        df_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target.
        categorical_features (list[str]): List of categorical feature names.
        search_params (dict[str, dict[str, Any]]): The search space parameters.
        default_params_list (Optional[list[dict[str, Any]]]): List of default hyperparameters
            set to enqueue before bayesian optimization.
    
    Returns:
        (dict[str, Any]): The best hyperparameters found during optimization.
    """
    # Create the Optuna study with TPE sampler
    study = optuna.create_study(
        study_name="basic_hgb_opt",
        direction="minimize",
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,
            multivariate=True
        )
    )

    # Create the objective
    objective = partial(
        optimize_hyperparams_hgb,
        search_params=search_params,
        df_train=df_train,
        y_train=y_train,
        df_val=df_val,
        y_val=y_val,
        categorical_features=categorical_features,
    )

    # Start the Bayesian Optimization with combination of default parameters if given
    if default_params_list is not None:
        for default_params in default_params_list:
            study.enqueue_trial(default_params)
    
    # Run Bayesian optimization
    study.optimize(objective, n_trials=50, show_progress_bar=True, n_jobs=1)

    # Get best hyperparams
    return study.best_params
