#!/usr/bin/env python3
"""Extended robustness and regime-information analysis for the HMM project.

This runner is deliberately separate from ``run_canonical.py``.  The original
canonical artifacts remain frozen; this script asks additional questions that
were identified during external review:

1. Are inferred states stable across random initializations?
2. Are conclusions robust across expanding chronological windows?
3. Do posterior probabilities quantify uncertainty around regime boundaries?
4. Does an SPY regime describe the behavior of economically different assets?
5. Is HMM more useful as a latent-state estimator than as a one-day predictor?

The held-out test segment is not touched during K/seed selection.  After model
selection on validation log-likelihood, the selected specification is refit on
all pre-test observations (train + validation) and evaluated once on test.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.data import (
    TargetSafeSplits,
    build_features,
    expanding_window_splits,
    load_asset,
    target_safe_train_validation_test_split,
    validate_data,
)
from src.evaluation import (
    direction_labels,
    evaluate_predictions,
    evaluate_predictions_by_state,
    summarize_states,
)
from src.model import (
    decode_past_only_posteriors,
    decode_past_only_states,
    gaussian_hmm_diagnostics,
    hmm_next_return_predictions,
    hmm_next_return_predictions_soft,
    posterior_confidence,
    posterior_entropy,
    train_gaussian_hmm,
)


@dataclass(frozen=True)
class ExtendedConfig:
    """Pre-specified configuration for the robustness extension."""

    assets: Tuple[str, ...] = (
        "SPY",      # broad U.S. equities / reference market
        "QQQ",      # growth and technology-heavy equities
        "IWM",      # U.S. small caps
        "TLT",      # long-duration U.S. Treasuries
        "GLD",      # gold
        "HYG",      # high-yield corporate credit
        "BTC-USD",  # high-volatility alternative asset
        "JPM",      # large financial equity
        "NVDA",     # high-beta growth equity
    )
    reference_asset: str = "SPY"
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
    k_values: Tuple[int, ...] = (2, 3, 4)
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 2026)
    covariance_type: str = "full"
    n_iter: int = 300
    tol: float = 1e-4
    context_window: int = 100
    ma_window: int = 5

    # Robustness analysis.  Walk-forward is intentionally limited to SPY so the
    # extension remains computationally modest and easy to defend orally.
    walk_forward_assets: Tuple[str, ...] = ("SPY",)
    walk_forward_folds: int = 3
    walk_initial_train_fraction: float = 0.55
    walk_validation_fraction: float = 0.15
    walk_test_fraction: float = 0.10

    cache_dir: str = "data_cache"
    download_sleep_sec: float = 1.0
    output_root: str = "experiments_extended"
    protocol_version: str = "extended-regime-analysis-1.0"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


@dataclass
class CandidateFit:
    """In-memory HMM candidate plus serializable selection diagnostics."""

    k: int
    seed: int
    model: Any
    train_states: np.ndarray
    record: Dict[str, Any]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()[:12] or "unknown"
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "sklearn", "hmmlearn", "yfinance"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "unavailable"
    return versions


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _load_all_assets(
    config: ExtendedConfig,
    force_download: bool,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[Dict[str, str]]]:
    """Download/cache the whole asset panel before any modeling begins."""
    frames: Dict[str, pd.DataFrame] = {}
    price_cols: Dict[str, str] = {}
    failures: List[Dict[str, str]] = []

    for index, asset in enumerate(config.assets):
        print(f"Loading {asset} ({index + 1}/{len(config.assets)})...")
        try:
            raw = load_asset(
                asset,
                config.start_date,
                config.end_date,
                cache_dir=config.cache_dir,
                force_download=force_download,
            )
            raw = validate_data(raw)
            featured, price_col = build_features(raw)
            missing = [c for c in config.feature_cols if c not in featured.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing}")
            frames[asset] = featured
            price_cols[asset] = price_col
            print(
                f"  {len(featured)} usable rows: "
                f"{featured.index.min().date()} to {featured.index.max().date()}"
            )
        except Exception as exc:
            failures.append({"asset": asset, "error": str(exc)})
            print(f"  FAILED: {exc}")

        # A short pause matters only on first download; cached reruns return quickly.
        if not force_download and index < len(config.assets) - 1:
            time.sleep(config.download_sleep_sec)

    return frames, price_cols, failures


def _fit_candidates(
    config: ExtendedConfig,
    splits: TargetSafeSplits,
) -> Tuple[List[CandidateFit], StandardScaler, np.ndarray, np.ndarray]:
    """Fit all K x seed candidates on train and score them on validation only."""
    feature_cols = list(config.feature_cols)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits.train[feature_cols].to_numpy())
    X_val = scaler.transform(splits.validation[feature_cols].to_numpy())

    candidates: List[CandidateFit] = []
    for k in config.k_values:
        for seed in config.seeds:
            start = time.time()
            try:
                model, fit_time = train_gaussian_hmm(
                    X_train,
                    n_states=k,
                    covariance_type=config.covariance_type,
                    n_iter=config.n_iter,
                    tol=config.tol,
                    random_state=seed,
                    verbose=False,
                )
                diag = gaussian_hmm_diagnostics(model, X_train)
                train_states = model.predict(X_train)
                occupancy = np.bincount(train_states, minlength=k) / len(train_states)
                val_ll = float(model.score(X_val))
                record = {
                    "k": k,
                    "seed": seed,
                    "converged": bool(diag["converged"]),
                    "iterations": int(diag["iterations"]),
                    "train_log_likelihood": float(diag["train_log_likelihood"]),
                    "train_log_likelihood_per_observation": float(
                        diag["train_log_likelihood_per_observation"]
                    ),
                    "validation_log_likelihood": val_ll,
                    "validation_log_likelihood_per_observation": val_ll / len(X_val),
                    "aic_train": float(diag["aic"]),
                    "bic_train": float(diag["bic"]),
                    "n_parameters": int(diag["n_parameters"]),
                    "fit_time_sec": float(fit_time),
                    "wall_time_sec": float(time.time() - start),
                    "min_state_occupancy_pct": float(100 * occupancy.min()),
                    "max_state_occupancy_pct": float(100 * occupancy.max()),
                }
                candidates.append(
                    CandidateFit(
                        k=k,
                        seed=seed,
                        model=model,
                        train_states=train_states,
                        record=record,
                    )
                )
                mark = "OK" if diag["converged"] else "NOT-CONVERGED"
                print(
                    f"    K={k} seed={seed}: {mark}, "
                    f"val LL/obs={record['validation_log_likelihood_per_observation']:.4f}"
                )
            except Exception as exc:
                print(f"    K={k} seed={seed}: FAILED ({exc})")
                candidates.append(
                    CandidateFit(
                        k=k,
                        seed=seed,
                        model=None,
                        train_states=np.array([], dtype=int),
                        record={
                            "k": k,
                            "seed": seed,
                            "converged": False,
                            "error": str(exc),
                            "wall_time_sec": float(time.time() - start),
                        },
                    )
                )

    return candidates, scaler, X_train, X_val


def _select_candidate(candidates: Sequence[CandidateFit]) -> CandidateFit:
    """Choose the converged candidate with highest validation log-likelihood."""
    eligible = [
        c
        for c in candidates
        if c.model is not None
        and c.record.get("converged")
        and np.isfinite(c.record.get("validation_log_likelihood", np.nan))
    ]
    if not eligible:
        raise RuntimeError("no converged HMM candidate is available for selection")
    return max(eligible, key=lambda c: c.record["validation_log_likelihood"])


def _seed_stability_table(asset: str, candidates: Sequence[CandidateFit]) -> pd.DataFrame:
    """Quantify label-invariant state-sequence agreement across random seeds.

    Adjusted Rand Index (ARI) is invariant to permutations of state IDs, which is
    exactly what is required for HMM seed stability.  ARI near 1 means two fits
    partition the training dates similarly even if their numeric labels differ.
    """
    rows: List[Dict[str, Any]] = []
    for k in sorted({c.k for c in candidates}):
        group = [
            c
            for c in candidates
            if c.k == k and c.model is not None and c.record.get("converged")
        ]
        pairwise_ari: List[float] = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairwise_ari.append(
                    float(adjusted_rand_score(group[i].train_states, group[j].train_states))
                )

        val_ll = [c.record["validation_log_likelihood_per_observation"] for c in group]
        bic = [c.record["bic_train"] for c in group]
        min_occ = [c.record["min_state_occupancy_pct"] for c in group]
        rows.append(
            {
                "asset": asset,
                "k": k,
                "n_converged_seeds": len(group),
                "n_seed_pairs": len(pairwise_ari),
                "mean_pairwise_ARI": float(np.mean(pairwise_ari)) if pairwise_ari else np.nan,
                "min_pairwise_ARI": float(np.min(pairwise_ari)) if pairwise_ari else np.nan,
                "max_pairwise_ARI": float(np.max(pairwise_ari)) if pairwise_ari else np.nan,
                "std_validation_LL_per_obs": float(np.std(val_ll, ddof=1)) if len(val_ll) > 1 else 0.0,
                "std_BIC_train": float(np.std(bic, ddof=1)) if len(bic) > 1 else 0.0,
                "min_state_occupancy_across_seeds_pct": float(np.min(min_occ)) if min_occ else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _baseline_predictions(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ma_window: int,
) -> Dict[str, np.ndarray]:
    """Generate simple baselines using only information available before each target."""
    history_mean = float(history_df["next_log_return"].mean())
    predictions: Dict[str, np.ndarray] = {
        "Naive - train mean": np.full(len(test_df), history_mean),
        "Naive - persistence": test_df["log_return"].to_numpy(dtype=float),
    }

    full = pd.concat([history_df, test_df]).sort_index().copy()
    full["ma_pred"] = full["log_return"].rolling(ma_window).mean()
    predictions[f"Moving Average {ma_window}"] = (
        full.loc[test_df.index, "ma_pred"].fillna(history_mean).to_numpy(dtype=float)
    )

    current_direction = direction_labels(history_df["log_return"].to_numpy())
    next_returns = history_df["next_log_return"].to_numpy(dtype=float)
    conditional_mean: Dict[int, float] = {}
    for direction in (0, 1):
        values = next_returns[current_direction == direction]
        conditional_mean[direction] = (
            float(np.mean(values)) if len(values) else history_mean
        )
    test_direction = direction_labels(test_df["log_return"].to_numpy())
    predictions["Discrete Markov Chain"] = np.asarray(
        [conditional_mean[int(d)] for d in test_direction], dtype=float
    )
    return predictions


def _posterior_uncertainty_summary(
    test_df: pd.DataFrame,
    posterior_states: np.ndarray,
    entropy: np.ndarray,
    confidence: np.ndarray,
) -> Dict[str, Any]:
    """Describe whether uncertainty concentrates around switches and turbulent days."""
    if len(test_df) != len(posterior_states) or len(test_df) != len(entropy):
        raise ValueError("posterior diagnostics must match test length")

    switch = np.zeros(len(test_df), dtype=bool)
    if len(switch) > 1:
        switch[1:] = posterior_states[1:] != posterior_states[:-1]

    # Descriptive neighborhood around a decoded switch.  This uses future switch
    # locations only for retrospective interpretation, never as a prediction input.
    near_switch = switch.copy()
    if len(switch) > 1:
        near_switch[:-1] |= switch[1:]
        near_switch[1:] |= switch[:-1]

    abs_return = test_df["log_return"].abs().to_numpy(dtype=float)
    rolling_vol = test_df["rolling_volatility_20"].to_numpy(dtype=float)
    q25, q75 = np.quantile(entropy, [0.25, 0.75]) if len(entropy) else (np.nan, np.nan)
    low = entropy <= q25
    high = entropy >= q75

    def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n_test_observations": int(len(test_df)),
        "n_posterior_argmax_switches": int(switch.sum()),
        "switch_rate_pct": float(100 * switch.mean()) if len(switch) else np.nan,
        "mean_entropy": float(np.mean(entropy)) if len(entropy) else np.nan,
        "median_entropy": float(np.median(entropy)) if len(entropy) else np.nan,
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else np.nan,
        "mean_entropy_on_switch_days": safe_mean(entropy, switch),
        "mean_entropy_on_non_switch_days": safe_mean(entropy, ~switch),
        "mean_entropy_near_switch_plus_minus_1": safe_mean(entropy, near_switch),
        "entropy_abs_return_correlation": safe_corr(entropy, abs_return),
        "entropy_rolling_volatility_correlation": safe_corr(entropy, rolling_vol),
        "low_entropy_quartile_mean_abs_return_pct": 100 * safe_mean(abs_return, low),
        "high_entropy_quartile_mean_abs_return_pct": 100 * safe_mean(abs_return, high),
        "low_entropy_quartile_mean_rolling_vol_pct": 100 * safe_mean(rolling_vol, low),
        "high_entropy_quartile_mean_rolling_vol_pct": 100 * safe_mean(rolling_vol, high),
    }


def _fit_final_and_evaluate(
    config: ExtendedConfig,
    asset: str,
    splits: TargetSafeSplits,
    selected: CandidateFit,
) -> Dict[str, Any]:
    """Refit selected K/seed on train+validation and evaluate held-out test once."""
    feature_cols = list(config.feature_cols)
    history = pd.concat([splits.train, splits.validation]).sort_index().copy()

    final_scaler = StandardScaler()
    X_history = final_scaler.fit_transform(history[feature_cols].to_numpy())
    X_test = final_scaler.transform(splits.test[feature_cols].to_numpy())

    model, fit_time = train_gaussian_hmm(
        X_history,
        n_states=selected.k,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        random_state=selected.seed,
        verbose=False,
    )
    diag = gaussian_hmm_diagnostics(model, X_history)

    history_states = model.predict(X_history)
    history_posteriors = model.predict_proba(X_history)
    test_viterbi_states = decode_past_only_states(
        model, X_history, X_test, config.context_window
    )
    test_posteriors = decode_past_only_posteriors(
        model, X_history, X_test, config.context_window
    )
    test_posterior_states = np.argmax(test_posteriors, axis=1)
    entropy = posterior_entropy(test_posteriors, normalize=True)
    confidence = posterior_confidence(test_posteriors)

    hard_pred = hmm_next_return_predictions(
        model, history, history_states, test_viterbi_states
    )
    soft_pred = hmm_next_return_predictions_soft(
        model, history, history_posteriors, test_posteriors
    )

    prediction_map = _baseline_predictions(history, splits.test, config.ma_window)
    prediction_map["Gaussian HMM - hard state"] = hard_pred
    prediction_map["Gaussian HMM - soft posterior"] = soft_pred

    metric_rows: List[Dict[str, Any]] = []
    conditioned_tables: List[pd.DataFrame] = []
    for model_name, prediction in prediction_map.items():
        evaluation = evaluate_predictions(model_name, splits.test, prediction)
        metric_rows.append(
            {
                "asset": asset,
                "model": model_name,
                "DPA_direction_accuracy": evaluation["DPA_direction_accuracy"],
                "MAE_return": evaluation["MAE_return"],
                "RMSE_return": evaluation["RMSE_return"],
                "MAPE_price_%": evaluation["MAPE_price_%"],
            }
        )
        conditioned = evaluate_predictions_by_state(
            splits.test,
            prediction,
            test_posterior_states,
            model_name,
        )
        conditioned.insert(0, "asset", asset)
        conditioned_tables.append(conditioned)

    history_entropy = posterior_entropy(history_posteriors, normalize=True)
    state_summary_history = summarize_states(
        history,
        history_states,
        posterior_probabilities=history_posteriors,
        posterior_entropy_values=history_entropy,
    )
    state_summary_test = summarize_states(
        splits.test,
        test_posterior_states,
        posterior_probabilities=test_posteriors,
        posterior_entropy_values=entropy,
    )

    posterior_daily = splits.test[
        [
            "log_return",
            "rolling_volatility_20",
            "daily_range",
            "volume_change",
            "current_close",
            "next_log_return",
        ]
    ].copy()
    posterior_daily["state_viterbi_past_only"] = test_viterbi_states
    posterior_daily["state_posterior_argmax"] = test_posterior_states
    posterior_daily["posterior_confidence"] = confidence
    posterior_daily["posterior_entropy"] = entropy
    for state in range(model.n_components):
        posterior_daily[f"p_state_{state}"] = test_posteriors[:, state]

    return {
        "model": model,
        "selected_k": selected.k,
        "selected_seed": selected.seed,
        "selected_by": "validation_log_likelihood",
        "final_fit_converged": bool(diag["converged"]),
        "final_fit_iterations": int(diag["iterations"]),
        "final_fit_time_sec": float(fit_time),
        "final_train_log_likelihood": float(diag["train_log_likelihood"]),
        "final_test_log_likelihood": float(model.score(X_test)),
        "final_aic_train": float(diag["aic"]),
        "final_bic_train": float(diag["bic"]),
        "transition_matrix": model.transmat_.copy(),
        "metrics": pd.DataFrame(metric_rows),
        "metrics_by_state": pd.concat(conditioned_tables, ignore_index=True),
        "state_summary_history": state_summary_history,
        "state_summary_test": state_summary_test,
        "posterior_daily": posterior_daily,
        "uncertainty_summary": _posterior_uncertainty_summary(
            splits.test, test_posterior_states, entropy, confidence
        ),
    }


def _save_asset_outputs(
    asset_dir: Path,
    asset: str,
    candidates: Sequence[CandidateFit],
    stability: pd.DataFrame,
    selected: CandidateFit,
    final: Dict[str, Any],
) -> List[str]:
    """Write machine-readable diagnostics plus a small set of focused figures."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: List[str] = []

    candidate_df = pd.DataFrame([c.record for c in candidates])
    files = {
        "candidate_diagnostics.csv": candidate_df,
        "seed_stability.csv": stability,
        "test_metrics.csv": final["metrics"],
        "test_metrics_by_state.csv": final["metrics_by_state"],
        "state_summary_history.csv": final["state_summary_history"],
        "state_summary_test.csv": final["state_summary_test"],
        "posterior_daily.csv": final["posterior_daily"],
    }
    for filename, frame in files.items():
        path = asset_dir / filename
        frame.to_csv(path, index=True if filename == "posterior_daily.csv" else False)
        artifact_paths.append(str(path))

    transmat = pd.DataFrame(final["transition_matrix"])
    transmat.index = [f"from_state_{i}" for i in range(len(transmat))]
    transmat.columns = [f"to_state_{i}" for i in range(len(transmat))]
    trans_path = asset_dir / "transition_matrix.csv"
    transmat.to_csv(trans_path)
    artifact_paths.append(str(trans_path))

    summary_json = {
        "asset": asset,
        "selected_k": final["selected_k"],
        "selected_seed": final["selected_seed"],
        "selected_by": final["selected_by"],
        "selection_validation_log_likelihood": selected.record.get(
            "validation_log_likelihood"
        ),
        "selection_validation_log_likelihood_per_observation": selected.record.get(
            "validation_log_likelihood_per_observation"
        ),
        "final_fit_converged": final["final_fit_converged"],
        "final_fit_iterations": final["final_fit_iterations"],
        "final_fit_time_sec": final["final_fit_time_sec"],
        "final_train_log_likelihood": final["final_train_log_likelihood"],
        "final_test_log_likelihood": final["final_test_log_likelihood"],
        "final_aic_train": final["final_aic_train"],
        "final_bic_train": final["final_bic_train"],
        "posterior_uncertainty": final["uncertainty_summary"],
    }
    summary_path = asset_dir / "selected_model_summary.json"
    _write_json(summary_path, summary_json)
    artifact_paths.append(str(summary_path))

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        daily = final["posterior_daily"]

        fig, ax = plt.subplots(figsize=(13, 5))
        scatter = ax.scatter(
            daily.index,
            daily["current_close"],
            c=daily["state_posterior_argmax"],
            s=12,
        )
        ax.set_title(f"{asset}: held-out price by posterior-argmax regime")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.3)
        fig.colorbar(scatter, ax=ax, label="Hidden state")
        fig.tight_layout()
        path = asset_dir / "test_price_by_regime.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        artifact_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(daily.index, daily["posterior_entropy"])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{asset}: normalized posterior regime uncertainty")
        ax.set_xlabel("Date")
        ax.set_ylabel("Posterior entropy [0, 1]")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = asset_dir / "posterior_entropy_timeline.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        artifact_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            final["transition_matrix"],
            annot=True,
            fmt=".2f",
            ax=ax,
            xticklabels=[f"State {i}" for i in range(final["selected_k"])],
            yticklabels=[f"State {i}" for i in range(final["selected_k"])],
        )
        ax.set_title(f"{asset}: transition matrix")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        fig.tight_layout()
        path = asset_dir / "transition_matrix.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        artifact_paths.append(str(path))
    except Exception as exc:
        print(f"  Plot warning for {asset}: {exc}")

    return artifact_paths


def _cross_asset_by_reference_state(
    reference_asset: str,
    reference_daily: pd.DataFrame,
    asset_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Measure other assets conditional on the reference asset's test-time regime."""
    reference = reference_daily[
        ["state_posterior_argmax", "posterior_entropy", "log_return"]
    ].rename(columns={"log_return": "reference_log_return"})

    rows: List[Dict[str, Any]] = []
    for asset, frame in asset_frames.items():
        joined = reference.join(
            frame[["log_return", "rolling_volatility_20"]],
            how="inner",
        ).dropna()
        if joined.empty:
            continue

        for state, part in joined.groupby("state_posterior_argmax"):
            ref_ret = part["reference_log_return"].to_numpy(dtype=float)
            asset_ret = part["log_return"].to_numpy(dtype=float)
            if len(part) >= 3 and np.std(ref_ret) > 0 and np.std(asset_ret) > 0:
                correlation = float(np.corrcoef(ref_ret, asset_ret)[0, 1])
            else:
                correlation = np.nan
            ref_variance = float(np.var(ref_ret, ddof=1)) if len(part) > 1 else np.nan
            beta = (
                float(np.cov(asset_ret, ref_ret, ddof=1)[0, 1] / ref_variance)
                if len(part) > 1 and np.isfinite(ref_variance) and ref_variance > 0
                else np.nan
            )
            rows.append(
                {
                    "reference_asset": reference_asset,
                    "reference_state": int(state),
                    "asset": asset,
                    "n_overlapping_days": int(len(part)),
                    "mean_return_%": 100 * float(part["log_return"].mean()),
                    "volatility_%": 100 * float(part["log_return"].std()),
                    "mean_abs_return_%": 100 * float(part["log_return"].abs().mean()),
                    "downside_frequency_%": 100 * float((part["log_return"] < 0).mean()),
                    "mean_rolling_volatility_%": 100
                    * float(part["rolling_volatility_20"].mean()),
                    "correlation_with_reference": correlation,
                    "beta_to_reference": beta,
                    "same_direction_with_reference_%": 100
                    * float(
                        (
                            direction_labels(asset_ret)
                            == direction_labels(ref_ret)
                        ).mean()
                    ),
                    "mean_return_difference_vs_reference_%": 100
                    * float((part["log_return"] - part["reference_log_return"]).mean()),
                    "mean_reference_posterior_entropy": float(
                        part["posterior_entropy"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _walk_forward_analysis(
    config: ExtendedConfig,
    asset: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat model selection/evaluation over a few expanding chronological folds."""
    folds = expanding_window_splits(
        df,
        n_splits=config.walk_forward_folds,
        initial_train_fraction=config.walk_initial_train_fraction,
        validation_fraction=config.walk_validation_fraction,
        test_fraction=config.walk_test_fraction,
    )

    rows: List[Dict[str, Any]] = []
    for fold_idx, splits in enumerate(folds):
        print(f"  Walk-forward {asset}, fold {fold_idx + 1}/{len(folds)}")
        candidates, _, _, _ = _fit_candidates(config, splits)
        selected = _select_candidate(candidates)
        final = _fit_final_and_evaluate(config, asset, splits, selected)
        metrics = final["metrics"].set_index("model")

        row: Dict[str, Any] = {
            "asset": asset,
            "fold": fold_idx,
            "train_start": splits.train.index.min().date().isoformat(),
            "train_end": splits.train.index.max().date().isoformat(),
            "validation_start": splits.validation.index.min().date().isoformat(),
            "validation_end": splits.validation.index.max().date().isoformat(),
            "test_start": splits.test.index.min().date().isoformat(),
            "test_end": splits.test.index.max().date().isoformat(),
            "selected_k": selected.k,
            "selected_seed": selected.seed,
            "selected_validation_LL_per_obs": selected.record[
                "validation_log_likelihood_per_observation"
            ],
            "mean_posterior_entropy": final["uncertainty_summary"]["mean_entropy"],
            "switch_rate_pct": final["uncertainty_summary"]["switch_rate_pct"],
        }
        for model_name in metrics.index:
            key = (
                model_name.lower()
                .replace(" - ", "_")
                .replace(" ", "_")
                .replace("=", "")
            )
            row[f"{key}_DPA"] = float(metrics.loc[model_name, "DPA_direction_accuracy"])
            row[f"{key}_MAE"] = float(metrics.loc[model_name, "MAE_return"])
        rows.append(row)

    return pd.DataFrame(rows)


def _make_overview_figures(output_dir: Path) -> List[str]:
    """Create figures that directly answer robustness questions."""
    artifacts: List[str] = []
    try:
        import matplotlib.pyplot as plt

        stability_path = output_dir / "seed_stability_all_assets.csv"
        if stability_path.exists():
            stability = pd.read_csv(stability_path)
            plot = stability.dropna(subset=["mean_pairwise_ARI"])
            if not plot.empty:
                pivot = plot.pivot(index="asset", columns="k", values="mean_pairwise_ARI")
                fig, ax = plt.subplots(figsize=(10, 5))
                pivot.plot(kind="bar", ax=ax)
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("Mean pairwise Adjusted Rand Index")
                ax.set_title("Hidden-state stability across random seeds")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                path = output_dir / "seed_stability_ari.png"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                artifacts.append(str(path))

        walk_path = output_dir / "walk_forward_results.csv"
        if walk_path.exists():
            walk = pd.read_csv(walk_path)
            hard_col = "gaussian_hmm_hard_state_DPA"
            soft_col = "gaussian_hmm_soft_posterior_DPA"
            baseline_col = "naive_train_mean_DPA"
            if all(c in walk.columns for c in (hard_col, soft_col, baseline_col)):
                fig, ax = plt.subplots(figsize=(9, 5))
                x = np.arange(len(walk))
                ax.plot(x, walk[hard_col], marker="o", label="HMM hard state")
                ax.plot(x, walk[soft_col], marker="o", label="HMM soft posterior")
                ax.plot(x, walk[baseline_col], marker="o", label="Naive train mean")
                ax.axhline(0.5, linestyle="--", linewidth=1, label="50% reference")
                ax.set_xticks(x, [f"Fold {i + 1}" for i in range(len(walk))])
                ax.set_ylim(0.35, 0.70)
                ax.set_ylabel("Directional Prediction Accuracy")
                ax.set_title("SPY expanding-window predictive robustness")
                ax.grid(alpha=0.3)
                ax.legend()
                fig.tight_layout()
                path = output_dir / "walk_forward_dpa.png"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                artifacts.append(str(path))
    except Exception as exc:
        print(f"Overview plot warning: {exc}")
    return artifacts


def run_extended(config: ExtendedConfig, force_download: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"extended_{timestamp}"
    output_dir = Path(config.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "protocol_version": config.protocol_version,
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "config": config.to_dict(),
        "package_versions": _package_versions(),
        "status": "running",
        "asset_status": [],
        "artifacts": [],
    }

    print("=" * 72)
    print("EXTENDED HMM REGIME / ROBUSTNESS ANALYSIS")
    print("=" * 72)
    print(f"Output directory: {output_dir}")

    frames, price_cols, load_failures = _load_all_assets(config, force_download)
    manifest["data_load_failures"] = load_failures
    manifest["loaded_assets"] = list(frames)
    manifest["price_columns"] = price_cols

    all_metrics: List[pd.DataFrame] = []
    all_stability: List[pd.DataFrame] = []
    final_by_asset: Dict[str, Dict[str, Any]] = {}

    for asset in config.assets:
        if asset not in frames:
            manifest["asset_status"].append(
                {"asset": asset, "status": "skipped", "reason": "data load failed"}
            )
            continue

        print("\n" + "=" * 72)
        print(f"MAIN SPLIT: {asset}")
        print("=" * 72)
        try:
            splits = target_safe_train_validation_test_split(
                frames[asset],
                validation_size=config.validation_size,
                test_size=config.test_size,
            )
            candidates, _, _, _ = _fit_candidates(config, splits)
            selected = _select_candidate(candidates)
            stability = _seed_stability_table(asset, candidates)
            final = _fit_final_and_evaluate(config, asset, splits, selected)

            asset_dir = output_dir / asset
            asset_artifacts = _save_asset_outputs(
                asset_dir, asset, candidates, stability, selected, final
            )
            manifest["artifacts"].extend(asset_artifacts)
            all_metrics.append(final["metrics"])
            all_stability.append(stability)
            final_by_asset[asset] = final

            manifest["asset_status"].append(
                {
                    "asset": asset,
                    "status": "completed",
                    "selected_k": selected.k,
                    "selected_seed": selected.seed,
                    "test_start": splits.test.index.min().date().isoformat(),
                    "test_end": splits.test.index.max().date().isoformat(),
                }
            )
        except Exception as exc:
            print(f"ERROR processing {asset}: {exc}")
            manifest["asset_status"].append(
                {"asset": asset, "status": "failed", "error": str(exc)}
            )

    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        path = output_dir / "test_metrics_all_assets.csv"
        metrics.to_csv(path, index=False)
        manifest["artifacts"].append(str(path))

    if all_stability:
        stability = pd.concat(all_stability, ignore_index=True)
        path = output_dir / "seed_stability_all_assets.csv"
        stability.to_csv(path, index=False)
        manifest["artifacts"].append(str(path))

    # Reference-regime analysis: other assets are not modeled jointly.  We simply
    # ask how their observed behavior changes conditional on SPY's held-out state.
    if config.reference_asset in final_by_asset:
        cross = _cross_asset_by_reference_state(
            config.reference_asset,
            final_by_asset[config.reference_asset]["posterior_daily"],
            frames,
        )
        cross_path = output_dir / "cross_asset_by_reference_state.csv"
        cross.to_csv(cross_path, index=False)
        manifest["artifacts"].append(str(cross_path))

    walk_tables: List[pd.DataFrame] = []
    for asset in config.walk_forward_assets:
        if asset not in frames:
            continue
        try:
            walk = _walk_forward_analysis(config, asset, frames[asset])
            walk_tables.append(walk)
        except Exception as exc:
            print(f"Walk-forward failed for {asset}: {exc}")
            manifest.setdefault("walk_forward_failures", []).append(
                {"asset": asset, "error": str(exc)}
            )

    if walk_tables:
        walk_all = pd.concat(walk_tables, ignore_index=True)
        path = output_dir / "walk_forward_results.csv"
        walk_all.to_csv(path, index=False)
        manifest["artifacts"].append(str(path))

    manifest["artifacts"].extend(_make_overview_figures(output_dir))
    manifest["status"] = "completed"
    manifest_path = output_dir / "manifest.json"
    # Register the manifest before writing it so its own artifact list is complete.
    manifest["artifacts"].append(str(manifest_path))
    _write_json(manifest_path, manifest)

    print("\n" + "=" * 72)
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"Results: {output_dir}")
    print("=" * 72)
    return output_dir


def _parse_assets(value: str) -> Tuple[str, ...]:
    assets = tuple(part.strip() for part in value.split(",") if part.strip())
    if not assets:
        raise argparse.ArgumentTypeError("at least one asset is required")
    return assets


def main() -> None:
    defaults = ExtendedConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=_parse_assets,
        default=defaults.assets,
        help="Comma-separated asset list. Default is the economically diverse panel.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore local CSV cache and download the requested date range again.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only seeds 42/123 and skip walk-forward; useful as a smoke test.",
    )
    args = parser.parse_args()

    config = replace(defaults, assets=args.assets)
    if args.quick:
        config = replace(config, seeds=(42, 123), walk_forward_assets=())

    run_extended(config, force_download=args.force_download)


if __name__ == "__main__":
    main()
