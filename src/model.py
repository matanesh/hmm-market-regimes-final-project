"""
Model module for HMM market regimes project.

The project uses two complementary forms of HMM inference:
- hard state decoding, useful for readable regime timelines;
- posterior state probabilities, useful for uncertainty-aware analysis and
  soft next-step forecasts.
"""

import math
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def gaussian_hmm_parameter_count(
    n_states: int, n_features: int, covariance_type: str
) -> int:
    """Return the number of free parameters in a hmmlearn GaussianHMM."""
    if n_states < 1 or n_features < 1:
        raise ValueError("n_states and n_features must be positive")

    covariance_parameters = {
        "spherical": n_states,
        "diag": n_states * n_features,
        "full": n_states * n_features * (n_features + 1) // 2,
        "tied": n_features * (n_features + 1) // 2,
    }
    if covariance_type not in covariance_parameters:
        raise ValueError(f"unsupported covariance_type: {covariance_type}")

    start_probabilities = n_states - 1
    transition_probabilities = n_states * (n_states - 1)
    means = n_states * n_features
    return (
        start_probabilities
        + transition_probabilities
        + means
        + covariance_parameters[covariance_type]
    )


def gaussian_hmm_diagnostics(model: GaussianHMM, X_train: np.ndarray) -> dict:
    """Calculate reusable convergence, likelihood and complexity diagnostics."""
    observations = np.asarray(X_train)
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError("X_train must be a non-empty two-dimensional array")

    log_likelihood = float(model.score(observations))
    n_observations = len(observations)
    n_parameters = gaussian_hmm_parameter_count(
        model.n_components, model.n_features, model.covariance_type
    )
    monitor = model.monitor_
    iterations = getattr(monitor, "iter", len(getattr(monitor, "history", ())))

    return {
        "converged": bool(monitor.converged),
        "iterations": int(iterations),
        "train_log_likelihood": log_likelihood,
        "train_log_likelihood_per_observation": log_likelihood / n_observations,
        "n_parameters": n_parameters,
        "aic": 2 * n_parameters - 2 * log_likelihood,
        "bic": n_parameters * math.log(n_observations) - 2 * log_likelihood,
    }


def train_gaussian_hmm(
    X_train_scaled: np.ndarray,
    n_states: int,
    covariance_type: str = "full",
    n_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[GaussianHMM, float]:
    """Train a Gaussian HMM using Baum-Welch / EM."""
    start_time = time.time()
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
    )
    model.fit(X_train_scaled)
    train_time = time.time() - start_time
    return model, train_time


def _history_context(X_history_scaled: np.ndarray, context_window: int) -> np.ndarray:
    """Return the bounded history supplied before sequential out-of-sample inference."""
    history = np.asarray(X_history_scaled)
    if history.ndim != 2 or len(history) == 0:
        raise ValueError("X_history_scaled must be a non-empty two-dimensional array")
    if context_window < 1:
        raise ValueError("context_window must be positive")
    return history[-context_window:] if len(history) > context_window else history


def decode_past_only_states(
    model: GaussianHMM,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    context_window: int = 100,
) -> np.ndarray:
    """Decode each out-of-sample state without using observations after that date.

    At test step ``t`` the sequence supplied to the model contains only a bounded
    training-history context and test observations through ``t``.  This avoids the
    common mistake of decoding the entire held-out sequence at once and then using
    future observations to influence an earlier state assignment.
    """
    test = np.asarray(X_test_scaled)
    if test.ndim != 2:
        raise ValueError("X_test_scaled must be two-dimensional")
    if len(test) == 0:
        return np.array([], dtype=int)

    context = _history_context(X_train_scaled, context_window)
    states = []
    for i in range(len(test)):
        seq = np.vstack([context, test[: i + 1]])
        states.append(model.predict(seq)[-1])
    return np.asarray(states, dtype=int)


def decode_past_only_posteriors(
    model: GaussianHMM,
    X_history_scaled: np.ndarray,
    X_future_scaled: np.ndarray,
    context_window: int = 100,
) -> np.ndarray:
    """Return past-only posterior state probabilities for sequential observations.

    ``hmmlearn.predict_proba`` performs forward-backward inference on the sequence
    it receives.  Because the sequence is truncated at the current observation,
    the posterior for its final row is conditioned on observations available only
    up to that time.  The result can therefore be used as an uncertainty-aware
    out-of-sample regime estimate without future leakage.
    """
    future = np.asarray(X_future_scaled)
    if future.ndim != 2:
        raise ValueError("X_future_scaled must be two-dimensional")
    if len(future) == 0:
        return np.empty((0, model.n_components), dtype=float)

    context = _history_context(X_history_scaled, context_window)
    rows = []
    for i in range(len(future)):
        seq = np.vstack([context, future[: i + 1]])
        rows.append(model.predict_proba(seq)[-1])

    posteriors = np.asarray(rows, dtype=float)
    if posteriors.shape != (len(future), model.n_components):
        raise RuntimeError("unexpected posterior-probability shape")
    return posteriors


def posterior_entropy(posteriors: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute Shannon entropy of state posteriors for each observation.

    When ``normalize=True`` entropy is divided by ``log(K)`` and therefore lies
    in [0, 1].  Zero means a nearly certain state assignment; one means that the
    posterior mass is close to uniform across states.
    """
    probs = np.asarray(posteriors, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError("posteriors must have shape (n_observations, n_states)")
    if len(probs) == 0:
        return np.array([], dtype=float)
    if np.any(probs < -1e-12):
        raise ValueError("posterior probabilities cannot be negative")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("each posterior row must sum to one")

    clipped = np.clip(probs, 1e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    if normalize and probs.shape[1] > 1:
        entropy = entropy / np.log(probs.shape[1])
    return entropy


def posterior_confidence(posteriors: np.ndarray) -> np.ndarray:
    """Return the maximum posterior state probability for each observation."""
    probs = np.asarray(posteriors, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 1:
        raise ValueError("posteriors must have shape (n_observations, n_states)")
    return probs.max(axis=1) if len(probs) else np.array([], dtype=float)


def hmm_next_return_predictions(
    model: GaussianHMM,
    train_df: pd.DataFrame,
    train_states: np.ndarray,
    test_states_past_only: np.ndarray,
) -> np.ndarray:
    """Predict next return from a hard current-state assignment.

    The state-specific raw-return mean is estimated on training data.  The model's
    transition row maps the current state into next-state probabilities, and the
    forecast is the corresponding expected next-state return.
    """
    n_states = model.n_components
    tmp = train_df.copy()
    tmp["state"] = train_states
    global_mean = train_df["log_return"].mean()

    state_return_mean = np.zeros(n_states)
    for s in range(n_states):
        vals = tmp.loc[tmp["state"] == s, "log_return"]
        state_return_mean[s] = vals.mean() if len(vals) else global_mean

    return np.asarray(
        [np.dot(model.transmat_[s], state_return_mean) for s in test_states_past_only]
    )


def hmm_next_return_predictions_soft(
    model: GaussianHMM,
    train_df: pd.DataFrame,
    train_posteriors: np.ndarray,
    current_posteriors_past_only: np.ndarray,
) -> np.ndarray:
    """Predict next return using the full posterior distribution over current states.

    This is a probabilistically smoother counterpart to hard-state forecasting:

    1. estimate each state's raw-return mean with posterior responsibility weights;
    2. propagate the current posterior one step through the transition matrix;
    3. take the expectation of the state-specific return means.

    No held-out target values are used.
    """
    train_probs = np.asarray(train_posteriors, dtype=float)
    current_probs = np.asarray(current_posteriors_past_only, dtype=float)
    n_states = model.n_components

    if train_probs.shape != (len(train_df), n_states):
        raise ValueError("train_posteriors shape does not match training data/model")
    if current_probs.ndim != 2 or current_probs.shape[1] != n_states:
        raise ValueError("current_posteriors_past_only has incompatible shape")

    train_returns = train_df["log_return"].to_numpy(dtype=float)
    global_mean = float(np.mean(train_returns))
    responsibility = train_probs.sum(axis=0)
    weighted_sum = train_probs.T @ train_returns
    state_return_mean = np.divide(
        weighted_sum,
        responsibility,
        out=np.full(n_states, global_mean, dtype=float),
        where=responsibility > 1e-12,
    )

    next_state_probs = current_probs @ model.transmat_
    return next_state_probs @ state_return_mean
