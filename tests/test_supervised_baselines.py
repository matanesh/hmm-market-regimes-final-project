import numpy as np
import pandas as pd

from src.data import TargetSafeSplits
from run_supervised_baselines import (
    SupervisedConfig,
    compare_with_hmm,
    fit_and_evaluate_asset,
)


def _partition(start: str, n: int, feature_offset: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    direction = np.array([0, 1] * (n // 2))
    returns = np.where(direction == 1, 0.01, -0.01)
    return pd.DataFrame(
        {
            "log_return": feature_offset + direction.astype(float),
            "rolling_volatility_20": feature_offset + direction * 0.1 + 0.01,
            "daily_range": feature_offset + direction * 0.01 + 0.02,
            "volume_change": feature_offset + direction * 0.05,
            "next_log_return": returns,
            "target_date": dates + pd.Timedelta(days=1),
        },
        index=dates,
    )


def test_supervised_config_uses_prespecified_assets_features_and_fixed_models():
    config = SupervisedConfig()

    assert config.assets == ("SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "BTC-USD", "JPM", "NVDA")
    assert config.feature_cols == (
        "log_return",
        "rolling_volatility_20",
        "daily_range",
        "volume_change",
    )
    assert config.logistic_c == 1.0
    assert config.logistic_max_iter == 2000
    assert config.rf_n_estimators == 500
    assert config.rf_max_depth == 6
    assert config.rf_min_samples_leaf == 10
    assert config.hgb_max_iter == 200
    assert config.hgb_learning_rate == 0.05
    assert config.hgb_max_leaf_nodes == 15


def test_fit_and_evaluate_asset_scales_logistic_from_train_only_and_returns_finite_metrics():
    train = _partition("2020-01-01", 40, feature_offset=0.0)
    validation = _partition("2020-03-01", 20, feature_offset=10.0)
    test = _partition("2020-04-01", 20, feature_offset=20.0)
    splits = TargetSafeSplits(train=train, validation=validation, test=test)

    results, qa = fit_and_evaluate_asset("TEST", splits, SupervisedConfig())

    assert set(results["model"]) == {
        "Logistic Regression",
        "Random Forest Classifier",
        "HistGradientBoostingClassifier",
    }
    assert (results["n_test_samples"] == len(test)).all()
    assert np.isfinite(results[["DPA_direction_accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score", "runtime_sec"]].to_numpy()).all()
    assert qa["scaler_fit_partition"] == "train"
    assert qa["scaler_fit_rows"] == len(train)
    assert qa["scaler_feature_means"] == train[list(SupervisedConfig().feature_cols)].mean().to_dict()
    assert qa["test_used_for_model_selection"] is False


def test_compare_with_hmm_keeps_requested_hmm_rows_without_selecting_a_winner():
    supervised = pd.DataFrame(
        {
            "asset": ["SPY", "SPY", "SPY"],
            "model": ["Logistic Regression", "Random Forest Classifier", "HistGradientBoostingClassifier"],
            "DPA_direction_accuracy": [0.50, 0.51, 0.52],
        }
    )
    hmm = pd.DataFrame(
        {
            "asset": ["SPY", "SPY", "SPY"],
            "model": ["Naive - train mean", "Gaussian HMM - hard state", "Gaussian HMM - soft posterior"],
            "DPA_direction_accuracy": [0.53, 0.54, 0.55],
        }
    )

    comparison = compare_with_hmm(supervised, hmm)

    assert list(comparison["model"]) == [
        "Naive - train mean",
        "Logistic Regression",
        "Random Forest Classifier",
        "HistGradientBoostingClassifier",
        "Gaussian HMM - hard state",
        "Gaussian HMM - soft posterior",
    ]
    assert "winner" not in comparison.columns
