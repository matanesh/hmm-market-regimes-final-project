import math

import pandas as pd
import pytest

from src.data import build_features, target_safe_train_validation_test_split
from src.model import gaussian_hmm_diagnostics, gaussian_hmm_parameter_count


def _next_day_target_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "feature": range(11),
            "target": range(1, 12),
            "target_date": dates[1:],
        },
        index=dates[:-1],
    )


def test_build_features_records_the_next_observation_as_target_date():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    raw = pd.DataFrame(
        {
            "Adj Close": [100.0 + i for i in range(25)],
            "Close": [100.0 + i for i in range(25)],
            "High": [101.0 + i for i in range(25)],
            "Low": [99.0 + i for i in range(25)],
            "Volume": [1000.0 + i for i in range(25)],
        },
        index=dates,
    )

    featured, _ = build_features(raw)

    assert "target_date" in featured.columns
    assert (featured["target_date"] == featured.index.to_series().shift(-1)).iloc[:-1].all()
    assert featured.iloc[-1]["target_date"] == dates[-1]


def test_target_safe_split_drops_pre_segment_boundary_targets():
    splits = target_safe_train_validation_test_split(
        _next_day_target_frame(), validation_size=0.20, test_size=0.30
    )

    assert list(splits.train.index) == list(pd.date_range("2024-01-01", periods=4))
    assert list(splits.validation.index) == [pd.Timestamp("2024-01-06")]
    assert list(splits.test.index) == list(pd.date_range("2024-01-08", periods=4))
    assert splits.train["target_date"].max() < splits.validation.index.min()
    assert splits.validation["target_date"].max() < splits.test.index.min()


def test_target_safe_split_rejects_non_monotonic_or_overlapping_time_index():
    frame = _next_day_target_frame()
    non_monotonic = frame.iloc[[0, 2, 1, *range(3, len(frame))]]
    with pytest.raises(ValueError, match="strictly increasing"):
        target_safe_train_validation_test_split(non_monotonic, 0.20, 0.30)

    duplicate = frame.copy()
    duplicate.index = list(frame.index[:-1]) + [frame.index[-2]]
    with pytest.raises(ValueError, match="unique"):
        target_safe_train_validation_test_split(duplicate, 0.20, 0.30)


@pytest.mark.parametrize(
    ("covariance_type", "expected"),
    [("spherical", 23), ("diag", 32), ("full", 50), ("tied", 30)],
)
def test_gaussian_hmm_parameter_count_matches_covariance_formula(
    covariance_type, expected
):
    assert gaussian_hmm_parameter_count(3, 4, covariance_type) == expected


def test_gaussian_hmm_diagnostics_normalizes_likelihood_and_information_criteria():
    class Monitor:
        converged = True
        iter = 17

    class FittedModel:
        n_components = 2
        n_features = 3
        covariance_type = "diag"
        monitor_ = Monitor()

        @staticmethod
        def score(observations):
            assert len(observations) == 10
            return -123.0

    diagnostics = gaussian_hmm_diagnostics(FittedModel(), [[0.0, 0.0, 0.0]] * 10)

    assert diagnostics["converged"] is True
    assert diagnostics["iterations"] == 17
    assert diagnostics["train_log_likelihood"] == -123.0
    assert diagnostics["train_log_likelihood_per_observation"] == -12.3
    assert diagnostics["n_parameters"] == 15
    assert diagnostics["aic"] == 276.0
    assert diagnostics["bic"] == pytest.approx(246.0 + 15 * math.log(10))
