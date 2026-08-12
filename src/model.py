"""
Model module for HMM market regimes project.
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


def train_gaussian_hmm(X_train_scaled: np.ndarray, 
                       n_states: int, 
                       covariance_type: str = "full",
                       n_iter: int = 300,
                       tol: float = 1e-4,
                       random_state: Optional[int] = None,
                       verbose: bool = False) -> Tuple[GaussianHMM, float]:
    """
    Train a Gaussian HMM using the Baum-Welch algorithm.

    Parameters:
    -----------
    X_train_scaled : np.ndarray
        Scaled training features.
    n_states : int
        Number of hidden states.
    covariance_type : str, default "full"
        Type of covariance parameters.
    n_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence threshold.
    random_state : int, optional
        Seed for random number generator.
    verbose : bool, default False
        Whether to print convergence messages.

    Returns:
    --------
    Tuple[GaussianHMM, float]
        Trained model and training runtime in seconds.
    """
    start_time = time.time()
    model = GaussianHMM(n_components=n_states,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        tol=tol,
                        random_state=random_state,
                        verbose=verbose)
    model.fit(X_train_scaled)
    train_time = time.time() - start_time
    return model, train_time


def decode_past_only_states(model: GaussianHMM,
                            X_train_scaled: np.ndarray,
                            X_test_scaled: np.ndarray,
                            context_window: int = 100) -> np.ndarray:
    """
    Decode hidden states for test data using only past information (no peeking into future).
    For each test point, we use the context from the end of training and all previous test points.

    Parameters:
    -----------
    model : GaussianHMM
        Trained HMM model.
    X_train_scaled : np.ndarray
        Scaled training features.
    X_test_scaled : np.ndarray
        Scaled test features.
    context_window : int, default 100
        Number of training observations to use as initial context.

    Returns:
    --------
    np.ndarray
        Array of hidden states for the test data.
    """
    states = []
    # Use the last `context_window` of training data as initial context
    if len(X_train_scaled) > context_window:
        context = X_train_scaled[-context_window:]
    else:
        context = X_train_scaled  # if training data is shorter than context_window

    for i in range(len(X_test_scaled)):
        # Sequence is context + all test observations up to current index
        seq = np.vstack([context, X_test_scaled[:i+1]])
        # Predict states for the entire sequence and take the last one (current test point)
        states.append(model.predict(seq)[-1])
    return np.array(states)


def hmm_next_return_predictions(model: GaussianHMM,
                                train_df: pd.DataFrame,
                                train_states: np.ndarray,
                                test_states_past_only: np.ndarray) -> np.ndarray:
    """
    Predict next day's log return using the HMM model.
    The prediction is the expected log return given the current state, which is computed as:
        E[return | state] = sum_{next_state} P(next_state | state) * E[return | next_state]
    We approximate E[return | state] by the average log return of training observations in that state.

    Parameters:
    -----------
    model : GaussianHMM
        Trained HMM model.
    train_df : pd.DataFrame
        Training data (with features).
    train_states : np.ndarray
        Hidden states for training data.
    test_states_past_only : np.ndarray
        Hidden states for test data (decoded past-only).

    Returns:
    --------
    np.ndarray
        Predicted next day's log return for each test point.
    """
    n_states = model.n_components
    # Create a temporary DataFrame with states for training data
    tmp = train_df.copy()
    tmp["state"] = train_states
    
    # Global mean return (fallback if a state has no observations)
    global_mean = train_df["log_return"].mean()
    
    # Compute mean return for each state in training data
    state_return_mean = np.zeros(n_states)
    for s in range(n_states):
        vals = tmp.loc[tmp["state"] == s, "log_return"]
        state_return_mean[s] = vals.mean() if len(vals) else global_mean
    
    # For each test state, compute the expected return as the dot product of
    # the transition row (from current state to next state) and the state return means
    pred_returns = np.array([np.dot(model.transmat_[s], state_return_mean) 
                             for s in test_states_past_only])
    return pred_returns