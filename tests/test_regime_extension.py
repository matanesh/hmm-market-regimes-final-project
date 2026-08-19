import numpy as np
import pandas as pd
import pytest

from src.data import expanding_window_splits
from src.evaluation import summarize_states
from src.model import (
    hmm_next_return_predictions_soft,
    posterior_confidence,
    posterior_entropy,
)


def _feature_frame(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    x = np.linspace(-0.02, 0.02, n)
    return pd.DataFrame(
        {
            "log_return": x,
            "rolling_volatility_20": np.linspace(0.005, 0.03, n),
            "daily_range": np.linspace(0.01, 0.04, n),
            "volume_change": np.sin(np.linspace(0, 5, n)) * 0.1,
            "current_close": 100 * np.exp(np.cumsum(x)),
            "next_log_return": np.roll(x, -1),
            "next_close": 101.0,
            "target_date": dates[1:],
        },
        index=dates[:-1],
    )


def test_posterior_entropy_and_confidence_have_expected_extremes():
    probs = np.array(
        [
            [1.0, 0.0, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
            [0.8, 0.1, 0.1],
        ]
    )
    entropy = posterior_entropy(probs, normalize=True)
    confidence = posterior_confidence(probs)

    assert entropy[0] == pytest.approx(0.0, abs=1e-10)
    assert entropy[1] == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < entropy[2] < 1.0
    assert np.allclose(confidence, [1.0, 1 / 3, 0.8])


def test_posterior_entropy_rejects_non_normalized_rows():
    with pytest.raises(ValueError, match="sum to one"):
        posterior_entropy(np.array([[0.7, 0.7]]))


def test_soft_hmm_forecast_propagates_posterior_through_transition_matrix():
    class FakeModel:
        n_components = 2
        transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])

    train_df = pd.DataFrame({"log_return": [0.01, 0.02, -0.01, -0.02]})
    train_post = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    current = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    pred = hmm_next_return_predictions_soft(FakeModel(), train_df, train_post, current)

    # State means are +1.5% and -1.5%.  One-step transition expectations:
    # from state 0: .9*(.015)+.1*(-.015)=.012
    # from state 1: .2*(.015)+.8*(-.015)=-.009
    assert pred[0] == pytest.approx(0.012)
    assert pred[1] == pytest.approx(-0.009)
    assert pred[2] == pytest.approx(0.0015)


def test_expanding_window_splits_are_chronological_and_target_safe():
    frame = _feature_frame(100)
    folds = expanding_window_splits(
        frame,
        n_splits=3,
        initial_train_fraction=0.55,
        validation_fraction=0.15,
        test_fraction=0.10,
    )

    assert len(folds) == 3
    previous_test_start = None
    for fold in folds:
        assert fold.train.index.max() < fold.validation.index.min()
        assert fold.validation.index.max() < fold.test.index.min()
        assert fold.train["target_date"].max() < fold.validation.index.min()
        assert fold.validation["target_date"].max() < fold.test.index.min()
        if previous_test_start is not None:
            assert fold.test.index.min() > previous_test_start
        previous_test_start = fold.test.index.min()

    assert len(folds[1].train) > len(folds[0].train)
    assert len(folds[2].train) > len(folds[1].train)


def test_state_summary_contains_risk_range_volume_drawdown_and_uncertainty():
    frame = _feature_frame(10)
    states = np.array([0] * 5 + [1] * 5)
    probs = np.array([[0.9, 0.1]] * 5 + [[0.2, 0.8]] * 5)
    entropy = posterior_entropy(probs)

    summary = summarize_states(
        frame,
        states,
        posterior_probabilities=probs,
        posterior_entropy_values=entropy,
    )

    required = {
        "mean_daily_return_%",
        "volatility_daily_%",
        "mean_daily_range_%",
        "mean_log_volume_change_pct",
        "downside_frequency_%",
        "mean_drawdown_%",
        "avg_duration_days",
        "mean_posterior_confidence",
        "mean_posterior_entropy",
    }
    assert required.issubset(summary.columns)
    assert len(summary) == 2
