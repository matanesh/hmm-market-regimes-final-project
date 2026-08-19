#!/usr/bin/env python3
"""Fixed supervised classification baselines for next-day return direction.

This runner is intentionally separate from the HMM runners.  It reuses the
project's loading, feature-construction, and target-safe chronological split
functions, evaluates three pre-specified classifiers once on each held-out test
segment, and does not tune or select models using validation or test outcomes.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data import (
    TargetSafeSplits,
    build_features,
    load_asset,
    target_safe_train_validation_test_split,
    validate_data,
)
from src.evaluation import direction_labels


@dataclass(frozen=True)
class SupervisedConfig:
    """Pre-specified, fixed configurations; no hyperparameter search is run."""

    assets: Tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "BTC-USD", "JPM", "NVDA"
    )
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    feature_cols: Tuple[str, ...] = (
        "log_return",
        "rolling_volatility_20",
        "daily_range",
        "volume_change",
    )
    validation_size: float = 0.15
    test_size: float = 0.15
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    rf_n_estimators: int = 500
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 10
    rf_random_state: int = 42
    rf_n_jobs: int = -1
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 15
    hgb_random_state: int = 42
    cache_dir: str = "data_cache"
    output_root: str = "experiments_supervised"
    hmm_run_dir: str = "experiments_extended/extended_20260811_121957"
    protocol_version: str = "supervised-baselines-1.0"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, tuple):
                payload[key] = list(value)
        return payload


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _package_versions() -> Dict[str, str]:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def _models(config: SupervisedConfig) -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            C=config.logistic_c,
            max_iter=config.logistic_max_iter,
            random_state=config.rf_random_state,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.rf_random_state,
            n_jobs=config.rf_n_jobs,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=config.hgb_max_iter,
            learning_rate=config.hgb_learning_rate,
            max_leaf_nodes=config.hgb_max_leaf_nodes,
            random_state=config.hgb_random_state,
        ),
    }


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) != 2:
        raise ValueError("held-out test target must contain both Up and Down classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("classifier probabilities must be finite values in [0, 1]")
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "DPA_direction_accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def fit_and_evaluate_asset(
    asset: str, splits: TargetSafeSplits, config: SupervisedConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit fixed classifiers on Train only and evaluate the untouched Test once.

    Validation is retained as its own target-safe chronological partition to keep
    the project protocol identical to the HMM extension.  Since all classifier
    settings are pre-specified, neither validation nor test is used for tuning,
    selection, calibration, threshold adjustment, or refitting.
    """
    features = list(config.feature_cols)
    X_train_raw = splits.train[features].to_numpy(dtype=float)
    X_test_raw = splits.test[features].to_numpy(dtype=float)
    y_train = direction_labels(splits.train["next_log_return"].to_numpy(dtype=float))
    y_test = direction_labels(splits.test["next_log_return"].to_numpy(dtype=float))
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"{asset}: training target must contain both Up and Down classes")

    # Logistic Regression requires scaling.  Fit only on the Train partition.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    records = []
    for name, model in _models(config).items():
        start = time.perf_counter()
        X_train = X_train_scaled if name == "Logistic Regression" else X_train_raw
        X_test = X_test_scaled if name == "Logistic Regression" else X_test_raw
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = _classification_metrics(y_test, probabilities)
        records.append(
            {
                "asset": asset,
                "model": name,
                **metrics,
                "n_test_samples": int(len(splits.test)),
                "runtime_sec": float(time.perf_counter() - start),
                "train_start": str(splits.train.index.min().date()),
                "train_end": str(splits.train.index.max().date()),
                "validation_start": str(splits.validation.index.min().date()),
                "validation_end": str(splits.validation.index.max().date()),
                "test_start": str(splits.test.index.min().date()),
                "test_end": str(splits.test.index.max().date()),
            }
        )

    qa = {
        "asset": asset,
        "scaler_fit_partition": "train",
        "scaler_fit_rows": int(len(splits.train)),
        "scaler_feature_means": {
            feature: float(mean) for feature, mean in zip(features, scaler.mean_)
        },
        "validation_used_for_model_selection": False,
        "test_used_for_model_selection": False,
        "target_safe_split": True,
        "shuffle_used": False,
    }
    return pd.DataFrame(records), qa


def compare_with_hmm(supervised: pd.DataFrame, hmm: pd.DataFrame) -> pd.DataFrame:
    """Create a descriptive fixed-order DPA comparison without selecting a winner."""
    requested_hmm = [
        "Naive - train mean",
        "Gaussian HMM - hard state",
        "Gaussian HMM - soft posterior",
    ]
    requested_supervised = [
        "Logistic Regression",
        "Random Forest Classifier",
        "HistGradientBoostingClassifier",
    ]
    ordered_models = [requested_hmm[0], *requested_supervised, *requested_hmm[1:]]
    combined = pd.concat(
        [
            hmm.loc[hmm["model"].isin(requested_hmm), ["asset", "model", "DPA_direction_accuracy"]],
            supervised.loc[
                supervised["model"].isin(requested_supervised),
                ["asset", "model", "DPA_direction_accuracy"],
            ],
        ],
        ignore_index=True,
    )
    combined["model"] = pd.Categorical(combined["model"], categories=ordered_models, ordered=True)
    return combined.sort_values(["asset", "model"]).reset_index(drop=True)


def _plot_dpa_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    model_order = list(comparison["model"].cat.categories)
    pivot = comparison.pivot(index="asset", columns="model", values="DPA_direction_accuracy")
    pivot = pivot.reindex(columns=model_order)
    ax = pivot.plot(
        kind="bar", figsize=(14, 6), width=0.82,
        color=["#6b7280", "#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#db2777"],
    )
    ax.set_ylabel("DPA / accuracy")
    ax.set_xlabel("Asset")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out next-day direction accuracy: fixed supervised and HMM baselines")
    ax.legend(title="Model", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run(config: SupervisedConfig, force_download: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_root) / f"supervised_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: Dict[str, Any] = {
        "protocol_version": config.protocol_version,
        "run_id": output_dir.name,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "config": config.to_dict(),
        "package_versions": _package_versions(),
        "status": "running",
        "loaded_assets": [],
        "data_load_failures": [],
        "asset_status": [],
        "qa": [],
        "artifacts": [],
    }

    all_results = []
    for index, asset in enumerate(config.assets, start=1):
        print(f"Loading and evaluating {asset} ({index}/{len(config.assets)})...")
        try:
            raw = validate_data(
                load_asset(
                    asset, config.start_date, config.end_date,
                    cache_dir=config.cache_dir, force_download=force_download,
                )
            )
            featured, _ = build_features(raw)
            splits = target_safe_train_validation_test_split(
                featured, validation_size=config.validation_size, test_size=config.test_size
            )
            results, qa = fit_and_evaluate_asset(asset, splits, config)
            all_results.append(results)
            manifest["loaded_assets"].append(asset)
            manifest["qa"].append(qa)
            manifest["asset_status"].append(
                {"asset": asset, "status": "completed", "n_test_samples": int(len(splits.test))}
            )
            print("  completed")
        except Exception as exc:
            manifest["data_load_failures"].append({"asset": asset, "error": str(exc)})
            manifest["asset_status"].append({"asset": asset, "status": "failed", "error": str(exc)})
            print(f"  FAILED: {exc}")

    if len(all_results) != len(config.assets):
        manifest["status"] = "failed"
        _write_json(output_dir / "manifest.json", manifest)
        raise RuntimeError("one or more assets failed; no partial supervised experiment is published")

    supervised = pd.concat(all_results, ignore_index=True)
    metric_columns = ["DPA_direction_accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score", "runtime_sec"]
    if not np.isfinite(supervised[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("supervised results contain unexplained non-finite metrics")
    supervised_path = output_dir / "supervised_results_all_assets.csv"
    supervised.to_csv(supervised_path, index=False)

    summary = (
        supervised.groupby("model", as_index=False)
        .agg(mean_DPA_direction_accuracy=("DPA_direction_accuracy", "mean"), assets=("asset", "nunique"))
        .sort_values("model")
    )
    summary_path = output_dir / "supervised_summary.csv"
    summary.to_csv(summary_path, index=False)

    hmm_path = Path(config.hmm_run_dir) / "test_metrics_all_assets.csv"
    hmm = pd.read_csv(hmm_path)
    comparison = compare_with_hmm(supervised, hmm)
    if set(comparison["asset"].astype(str)) != set(config.assets):
        raise ValueError("comparison is missing one or more requested assets")
    comparison_path = output_dir / "comparison_with_hmm.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_path = output_dir / "dpa_comparison_all_assets.png"
    _plot_dpa_comparison(comparison, plot_path)
    manifest["status"] = "completed"
    manifest["artifacts"] = [str(path) for path in (supervised_path, summary_path, comparison_path, plot_path)]
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Completed supervised baselines: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--hmm-run-dir", default=SupervisedConfig.hmm_run_dir)
    args = parser.parse_args()
    config = SupervisedConfig(hmm_run_dir=args.hmm_run_dir)
    run(config, force_download=args.force_download)


if __name__ == "__main__":
    main()
