"""
Evaluation module for HMM market regimes project.
"""

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, Any


def direction_labels(returns: np.ndarray) -> np.ndarray:
    """
    Convert returns to direction labels (1 for non-negative, 0 for negative).

    Parameters:
    -----------
    returns : np.ndarray
        Array of returns.

    Returns:
    --------
    np.ndarray
        Array of direction labels (1 for >=0, 0 for <0).
    """
    return np.where(np.asarray(returns) >= 0, 1, 0)


def evaluate_predictions(name: str,
                         test_df: pd.DataFrame,
                         pred_return: np.ndarray,
                         runtime_sec: float = float('nan'),
                         log_likelihood_train: float = float('nan'),
                         log_likelihood_test: float = float('nan')) -> Dict[str, Any]:
    """
    Evaluate model predictions using multiple metrics.

    Parameters:
    -----------
    name : str
        Name of the model.
    test_df : pd.DataFrame
        Test data (must contain 'next_log_return', 'next_close', 'current_close' columns).
    pred_return : np.ndarray
        Predicted next day's log return.
    runtime_sec : float, default np.nan
        Training/runtime in seconds.
    log_likelihood_train : float, default np.nan
        Log likelihood on training data.
    log_likelihood_test : float, default np.nan
        Log likelihood on test data.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing evaluation metrics.
    """
    pred_return = np.asarray(pred_return)
    actual_return = test_df["next_log_return"].values
    actual_close = test_df["next_close"].values
    current_close = test_df["current_close"].values
    
    # Convert predicted log return to predicted price
    pred_close = current_close * np.exp(pred_return)
    
    # Directional Prediction Accuracy (DPA)
    dpa = accuracy_score(direction_labels(actual_return), direction_labels(pred_return))
    
    # Return-based metrics
    mae_ret = mean_absolute_error(actual_return, pred_return)
    rmse_ret = np.sqrt(mean_squared_error(actual_return, pred_return))
    
    # Price-based MAPE
    mape_price = np.mean(np.abs((actual_close - pred_close) / actual_close)) * 100
    
    return {
        "model": name,
        "DPA_direction_accuracy": dpa,
        "MAE_return": mae_ret,
        "RMSE_return": rmse_ret,
        "MAPE_price_%": mape_price,
        "log_likelihood_train": log_likelihood_train,
        "log_likelihood_test": log_likelihood_test,
        "runtime_sec": runtime_sec
    }


def average_duration(states: np.ndarray, state: int) -> float:
    """
    Calculate average duration of a given state in a sequence of states.

    Parameters:
    -----------
    states : np.ndarray
        Sequence of state labels.
    state : int
        State label for which to calculate average duration.

    Returns:
    --------
    float
        Average duration of the specified state.
    """
    lengths, cur = [], 0
    for s in states:
        if s == state:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return float(np.mean(lengths)) if lengths else 0.0


def summarize_states(df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    """
    Create a summary table of hidden states based on training data.

    Parameters:
    -----------
    df : pd.DataFrame
        Original feature data (must contain 'log_return', 'rolling_volatility_20' columns).
    states : np.ndarray
        Hidden states for each observation in df.

    Returns:
    --------
    pd.DataFrame
        Summary table with one row per state containing:
        - state: state label
        - frequency_%: percentage of observations in this state
        - mean_daily_return_%: mean log return in this state (as percentage)
        - volatility_daily_%: standard deviation of log return in this state (as percentage)
        - mean_rolling_volatility_%: mean rolling volatility in this state (as percentage)
        - avg_duration_days: average consecutive days in this state
    """
    # Create a copy to avoid SettingWithCopyWarning
    tmp = df.copy()
    tmp["state"] = states
    
    rows = []
    for s in sorted(tmp["state"].unique()):
        part = tmp[tmp["state"] == s]
        rows.append({
            "state": int(s),
            "frequency_%": 100 * len(part) / len(tmp),
            "mean_daily_return_%": 100 * part["log_return"].mean(),
            "volatility_daily_%": 100 * part["log_return"].std(),
            "mean_rolling_volatility_%": 100 * part["rolling_volatility_20"].mean(),
            "avg_duration_days": average_duration(states, s)
        })
    
    return pd.DataFrame(rows).sort_values("state")