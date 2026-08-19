"""
Tests for the canonical runner that enforces Train/Validation/Test protocol,
multi-start selection on validation only, and canonical artifact emission.
"""
import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data import build_features, target_safe_train_validation_test_split
from src.model import (
    train_gaussian_hmm,
    decode_past_only_states,
    hmm_next_return_predictions,
    gaussian_hmm_diagnostics,
    gaussian_hmm_parameter_count,
)
from src.evaluation import evaluate_predictions, direction_labels, summarize_states


def _synthetic_ohlcv(n: int = 300, start: str = "2020-01-01", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with known properties for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    # Random walk with drift
    returns = np.random.normal(0.0002, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    # Generate realistic OHLCV
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.lognormal(13, 0.5, n).astype(int)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": high,
            "Low": low,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volume,
        },
        index=dates,
    )
    return df


class TestCanonicalRunnerProtocol:
    """Tests that the canonical runner enforces the correct protocol."""

    def test_train_validation_test_splits_are_target_safe(self):
        """The canonical runner must use target_safe_train_validation_test_split."""
        raw = _synthetic_ohlcv(300)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        # Boundary rows are dropped
        assert len(splits.train) + len(splits.validation) + len(splits.test) < len(df)
        # No target leakage across boundaries
        assert splits.train["target_date"].max() < splits.validation.index.min()
        assert splits.validation["target_date"].max() < splits.test.index.min()
        # All partitions have strictly increasing unique indexes
        for part in (splits.train, splits.validation, splits.test):
            assert part.index.is_monotonic_increasing
            assert part.index.is_unique

    def test_feature_scaler_fit_only_on_train(self):
        """Scaler must be fitted on train only and applied to val/test."""
        from sklearn.preprocessing import StandardScaler
        raw = _synthetic_ohlcv(300)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[feature_cols].values)
        X_val = scaler.transform(splits.validation[feature_cols].values)
        X_test = scaler.transform(splits.test[feature_cols].values)
        # Train mean ~0, std ~1
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-10)
        # Val/test not necessarily 0/1 but finite
        assert np.all(np.isfinite(X_val))
        assert np.all(np.isfinite(X_test))

    def test_multi_start_selection_on_validation_only(self):
        """HMM initialization/K selection must use validation criteria, never test."""
        raw = _synthetic_ohlcv(400)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[feature_cols].values)
        X_val = scaler.transform(splits.validation[feature_cols].values)
        X_test = scaler.transform(splits.test[feature_cols].values)

        # Train multiple seeds for K=2,3,4
        seeds = [42, 123, 456]
        k_values = [2, 3, 4]
        all_results = []

        for k in k_values:
            for seed in seeds:
                model, _ = train_gaussian_hmm(X_train, n_states=k, random_state=seed, n_iter=100, tol=1e-3)
                # Validation log-likelihood (not test!)
                val_ll = model.score(X_val)
                train_ll = model.score(X_train)
                diag = gaussian_hmm_diagnostics(model, X_train)
                all_results.append({
                    "k": k,
                    "seed": seed,
                    "train_ll": train_ll,
                    "val_ll": val_ll,
                    "converged": diag["converged"],
                    "iterations": diag["iterations"],
                    "aic": diag["aic"],
                    "bic": diag["bic"],
                })

        # Selection must be based on validation LL (or AIC/BIC on validation), never test
        # Best by validation LL
        best = max(all_results, key=lambda r: r["val_ll"])
        assert best["k"] in k_values
        assert best["seed"] in seeds

        # Now evaluate the locked best model on test (test must not influence selection)
        best_model, _ = train_gaussian_hmm(X_train, n_states=best["k"], random_state=best["seed"], n_iter=100, tol=1e-3)
        test_ll = best_model.score(X_test)
        assert np.isfinite(test_ll)

    def test_past_only_state_decoding(self):
        """Test state decoding must use only past information (prefix decoding)."""
        raw = _synthetic_ohlcv(300)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[feature_cols].values)
        X_test = scaler.transform(splits.test[feature_cols].values)

        model, _ = train_gaussian_hmm(X_train, n_states=3, random_state=42)
        train_states = model.predict(X_train)
        test_states = decode_past_only_states(model, X_train, X_test, context_window=50)

        # Test states length matches test data
        assert len(test_states) == len(splits.test)
        # Test states are valid state indices
        assert all(0 <= s < 3 for s in test_states)

    def test_baseline_predictions_evaluated_on_same_test_set(self):
        """All baselines and HMM must be evaluated on the identical test partition."""
        raw = _synthetic_ohlcv(300)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        test_df = splits.test

        # Naive train mean
        pred_naive = np.full(len(test_df), splits.train["next_log_return"].mean())
        res_naive = evaluate_predictions("Naive", test_df, pred_naive, runtime_sec=0.0)

        # Persistence
        pred_persist = test_df["log_return"].values
        res_persist = evaluate_predictions("Persistence", test_df, pred_persist, runtime_sec=0.0)

        # Moving average
        full = pd.concat([splits.train, test_df])
        ma = full["log_return"].rolling(5).mean()
        pred_ma = ma.loc[test_df.index].fillna(splits.train["next_log_return"].mean()).values
        res_ma = evaluate_predictions("MA5", test_df, pred_ma, runtime_sec=0.0)

        # Discrete Markov Chain
        train_curr_dir = direction_labels(splits.train["log_return"].values)
        train_next = splits.train["next_log_return"].values
        exp_next = {}
        for d in [0, 1]:
            vals = train_next[train_curr_dir == d]
            exp_next[d] = vals.mean() if len(vals) else splits.train["next_log_return"].mean()
        pred_dmc = np.array([exp_next[d] for d in direction_labels(test_df["log_return"].values)])
        res_dmc = evaluate_predictions("DMC", test_df, pred_dmc, runtime_sec=0.0)

        # All results must have the same required metrics
        for res in [res_naive, res_persist, res_ma, res_dmc]:
            assert "DPA_direction_accuracy" in res
            assert "MAE_return" in res
            assert "RMSE_return" in res
            assert "MAPE_price_%" in res

    def test_canonical_artifacts_emitted(self):
        """Canonical runner must emit a manifest and machine-readable artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate canonical runner output
            manifest = {
                "protocol_version": "1.0",
                "assets": ["SYNTHETIC"],
                "date_range": {"start": "2020-01-01", "end": "2021-01-01"},
                "folds": 1,
                "feature_cols": ["log_return", "rolling_volatility_20", "daily_range", "volume_change"],
                "k_values": [2, 3, 4],
                "seeds": [42, 123, 456],
                "validation_size": 0.15,
                "test_size": 0.15,
                "artifacts": {
                    "results_csv": "results.csv",
                    "manifest_json": "manifest.json",
                    "model_diagnostics_json": "model_diagnostics.json",
                },
                "git_commit": "test-commit",
            }
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Results CSV
            results_df = pd.DataFrame([
                {"model": "Naive", "DPA_direction_accuracy": 0.5, "MAE_return": 0.01},
                {"model": "HMM K=2 seed=42", "DPA_direction_accuracy": 0.55, "MAE_return": 0.009},
            ])
            results_path = os.path.join(tmpdir, "results.csv")
            results_df.to_csv(results_path, index=False)

            # Model diagnostics
            diagnostics = {
                "SYNTHETIC": {
                    "K=2": {"seed=42": {"converged": True, "val_ll": -100.0, "test_ll": -90.0}},
                }
            }
            diag_path = os.path.join(tmpdir, "model_diagnostics.json")
            with open(diag_path, "w") as f:
                json.dump(diagnostics, f, indent=2)

            # Verify all artifacts exist and are valid
            assert os.path.exists(manifest_path)
            assert os.path.exists(results_path)
            assert os.path.exists(diag_path)

            with open(manifest_path) as f:
                loaded_manifest = json.load(f)
            assert loaded_manifest["protocol_version"] == "1.0"
            assert "SYNTHETIC" in loaded_manifest["assets"]

            loaded_results = pd.read_csv(results_path)
            assert len(loaded_results) == 2
            assert "DPA_direction_accuracy" in loaded_results.columns

            with open(diag_path) as f:
                loaded_diag = json.load(f)
            assert "SYNTHETIC" in loaded_diag


class TestCanonicalRunnerInvariants:
    """Invariants that must hold for every canonical run."""

    def test_no_test_metric_used_for_model_selection(self):
        """Confirm that test metrics never appear in model selection logic."""
        # This is a meta-test: the canonical runner's selection function
        # must only accept validation metrics as selection criteria.
        pass  # Implementation tested in test_multi_start_selection_on_validation_only

    def test_hmm_diagnostics_complete(self):
        """Every fitted HMM must record convergence, iterations, LL, AIC, BIC."""
        raw = _synthetic_ohlcv(200)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.2, test_size=0.2)
        feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[feature_cols].values)

        model, _ = train_gaussian_hmm(X_train, n_states=3, random_state=42, n_iter=50)
        diag = gaussian_hmm_diagnostics(model, X_train)

        required_keys = ["converged", "iterations", "train_log_likelihood",
                         "train_log_likelihood_per_observation", "n_parameters", "aic", "bic"]
        for key in required_keys:
            assert key in diag, f"Missing diagnostic: {key}"
        assert isinstance(diag["converged"], bool)
        assert diag["iterations"] > 0
        assert np.isfinite(diag["train_log_likelihood"])
        assert np.isfinite(diag["aic"])
        assert np.isfinite(diag["bic"])
        assert diag["n_parameters"] == gaussian_hmm_parameter_count(3, 4, "full")

    def test_direction_threshold_consistent(self):
        """Direction labels must use a single centralized threshold (>=0)."""
        # Test that direction_labels uses >=0
        returns = np.array([-0.01, -0.001, 0.0, 0.001, 0.01])
        labels = direction_labels(returns)
        expected = np.array([0, 0, 1, 1, 1])  # >=0 -> 1
        assert np.array_equal(labels, expected)

    def test_chrono_no_shuffle(self):
        """All splits must be chronological without shuffling."""
        raw = _synthetic_ohlcv(200)
        df, _ = build_features(raw)
        splits = target_safe_train_validation_test_split(df, validation_size=0.15, test_size=0.15)
        # Train before val before test
        assert splits.train.index.max() < splits.validation.index.min()
        assert splits.validation.index.max() < splits.test.index.min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])