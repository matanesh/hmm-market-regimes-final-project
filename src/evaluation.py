"""
Evaluation module for the HMM market-regimes project.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def direction_labels(returns: np.ndarray) -> np.ndarray:
    """Convert returns to direction labels (1 for non-negative, 0 for negative)."""
    return np.where(np.asarray(returns) >= 0, 1, 0)


def evaluate_predictions(
    name: str,
    test_df: pd.DataFrame,
    pred_return: np.ndarray,
    runtime_sec: float = float("nan"),
    log_likelihood_train: float = float("nan"),
    log_likelihood_test: float = float("nan"),
) -> Dict[str, Any]:
    """Evaluate next-day return predictions using direction and error metrics."""
    pred_return = np.asarray(pred_return)
    actual_return = test_df["next_log_return"].values
    actual_close = test_df["next_close"].values
    current_close = test_df["current_close"].values

    if len(pred_return) != len(test_df):
        raise ValueError("prediction length must match evaluation data")
    if not np.isfinite(pred_return).all():
        raise ValueError("predictions contain non-finite values")

    pred_close = current_close * np.exp(pred_return)
    dpa = accuracy_score(direction_labels(actual_return), direction_labels(pred_return))
    mae_ret = mean_absolute_error(actual_return, pred_return)
    rmse_ret = np.sqrt(mean_squared_error(actual_return, pred_return))
    mape_price = np.mean(np.abs((actual_close - pred_close) / actual_close)) * 100

    return {
        "model": name,
        "DPA_direction_accuracy": dpa,
        "MAE_return": mae_ret,
        "RMSE_return": rmse_ret,
        "MAPE_price_%": mape_price,
        "log_likelihood_train": log_likelihood_train,
        "log_likelihood_test": log_likelihood_test,
        "runtime_sec": runtime_sec,
    }


def average_duration(states: np.ndarray, state: int) -> float:
    """Calculate average consecutive duration of a state in days/observations."""
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


def _drawdown_from_current_close(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return drawdown from the running peak when a current-close series exists."""
    if "current_close" not in df.columns:
        return None
    prices = pd.to_numeric(df["current_close"], errors="coerce")
    running_peak = prices.cummax()
    return prices / running_peak - 1.0


def summarize_states(
    df: pd.DataFrame,
    states: np.ndarray,
    posterior_probabilities: Optional[np.ndarray] = None,
    posterior_entropy_values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Create a quantitative profile of every inferred hidden state.

    In addition to return and volatility, the table reports intraday range,
    volume behavior, downside frequency, drawdown and optional posterior
    confidence/entropy.  These diagnostics are intended to support post-hoc
    semantic interpretation of states; numeric state IDs are never treated as
    economically meaningful labels by themselves.
    """
    states = np.asarray(states)
    if len(df) != len(states):
        raise ValueError("states length must match DataFrame length")
    if len(df) == 0:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["state"] = states
    drawdown = _drawdown_from_current_close(tmp)
    if drawdown is not None:
        tmp["drawdown"] = drawdown

    if posterior_probabilities is not None:
        probs = np.asarray(posterior_probabilities, dtype=float)
        if probs.ndim != 2 or len(probs) != len(tmp):
            raise ValueError("posterior_probabilities shape does not match DataFrame")
        tmp["posterior_confidence"] = probs.max(axis=1)

    if posterior_entropy_values is not None:
        entropy = np.asarray(posterior_entropy_values, dtype=float)
        if entropy.ndim != 1 or len(entropy) != len(tmp):
            raise ValueError("posterior_entropy_values shape does not match DataFrame")
        tmp["posterior_entropy"] = entropy

    rows = []
    for s in sorted(tmp["state"].unique()):
        mask = tmp["state"] == s
        part = tmp.loc[mask]
        row = {
            "state": int(s),
            "frequency_%": 100 * len(part) / len(tmp),
            "mean_daily_return_%": 100 * part["log_return"].mean(),
            "mean_abs_return_%": 100 * part["log_return"].abs().mean(),
            "volatility_daily_%": 100 * part["log_return"].std(),
            "mean_rolling_volatility_%": 100 * part["rolling_volatility_20"].mean(),
            "downside_frequency_%": 100 * (part["log_return"] < 0).mean(),
            "avg_duration_days": average_duration(states, s),
        }

        if "daily_range" in part.columns:
            row["mean_daily_range_%"] = 100 * part["daily_range"].mean()
        if "volume_change" in part.columns:
            # volume_change is a log ratio, so this is reported explicitly as log-percent.
            row["mean_log_volume_change_pct"] = 100 * part["volume_change"].mean()
            row["mean_abs_log_volume_change_pct"] = 100 * part["volume_change"].abs().mean()
        if "drawdown" in part.columns:
            row["mean_drawdown_%"] = 100 * part["drawdown"].mean()
            row["worst_drawdown_%"] = 100 * part["drawdown"].min()
        if "posterior_confidence" in part.columns:
            row["mean_posterior_confidence"] = part["posterior_confidence"].mean()
        if "posterior_entropy" in part.columns:
            row["mean_posterior_entropy"] = part["posterior_entropy"].mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def evaluate_predictions_by_state(
    df: pd.DataFrame,
    pred_return: np.ndarray,
    states: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate a forecast separately inside each inferred regime.

    This is a descriptive diagnostic.  It reveals whether aggregate errors are
    concentrated in particular regimes, but it must not be used to redesign the
    model after looking at held-out test outcomes.
    """
    pred = np.asarray(pred_return, dtype=float)
    states = np.asarray(states)
    if len(df) != len(pred) or len(df) != len(states):
        raise ValueError("df, predictions and states must have equal length")

    rows = []
    for state in sorted(np.unique(states)):
        mask = states == state
        part = df.iloc[np.flatnonzero(mask)]
        part_pred = pred[mask]
        metrics = evaluate_predictions(model_name, part, part_pred)
        rows.append(
            {
                "model": model_name,
                "state": int(state),
                "n_observations": int(mask.sum()),
                "DPA_direction_accuracy": metrics["DPA_direction_accuracy"],
                "MAE_return": metrics["MAE_return"],
                "RMSE_return": metrics["RMSE_return"],
                "MAPE_price_%": metrics["MAPE_price_%"],
            }
        )
    return pd.DataFrame(rows)
