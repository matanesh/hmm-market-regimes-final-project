#!/usr/bin/env python3
"""
Canonical runner for Gaussian HMM market regimes analysis.

This runner enforces a strict Train/Validation/Test protocol with:
- Target-safe chronological splits (no boundary leakage)
- Multi-start HMM fitting with selection on validation criteria only
- Locked model evaluation on held-out test set
- Complete diagnostics recording (convergence, iterations, LL, AIC, BIC)
- Canonical artifact emission with manifest
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Local imports
from src.data import (
    load_asset,
    validate_data,
    build_features,
    target_safe_train_validation_test_split,
    TargetSafeSplits,
)
from src.model import (
    train_gaussian_hmm,
    decode_past_only_states,
    hmm_next_return_predictions,
    gaussian_hmm_diagnostics,
)
from src.evaluation import (
    evaluate_predictions,
    direction_labels,
    summarize_states,
)

warnings.filterwarnings("ignore")


# ─── Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanonicalConfig:
    """Immutable configuration for a canonical run."""
    # Universe and dates (pre-specified, not selected by test performance)
    assets: Tuple[str, ...] = ("SPY", "GLD", "TLT", "BTC-USD", "DIS")
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"  # Fixed end date for reproducibility

    # Chronological protocol
    validation_size: float = 0.15
    test_size: float = 0.15

    # Features (fixed)
    feature_cols: Tuple[str, ...] = (
        "log_return",
        "rolling_volatility_20",
        "daily_range",
        "volume_change",
    )

    # HMM hyperparameters (pre-specified)
    k_values: Tuple[int, ...] = (2, 3, 4)
    covariance_type: str = "full"
    n_iter: int = 300
    tol: float = 1e-4
    seeds: Tuple[int, ...] = (42, 123, 456)  # Multiple initializations
    context_window: int = 100

    # Baselines
    ma_window: int = 5

    # Output
    output_root: str = "experiments_canonical"
    protocol_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalConfig":
        # Convert lists back to tuples
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)


# ─── Data structures for canonical results ────────────────────────────────

@dataclass
class ModelResult:
    """Result for a single model (baseline or HMM) on one asset/fold."""
    asset: str
    fold: int
    model_name: str
    model_family: str  # "baseline" or "hmm"
    # Selection metadata (empty for baselines)
    k: Optional[int] = None
    seed: Optional[int] = None
    selected_by: Optional[str] = None  # "val_ll", "val_aic", "val_bic"
    # Metrics
    dpa: float = 0.0
    mae_return: float = 0.0
    rmse_return: float = 0.0
    mape_price_pct: float = 0.0
    log_likelihood_train: float = float("nan")
    log_likelihood_val: float = float("nan")
    log_likelihood_test: float = float("nan")
    # HMM diagnostics
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    n_parameters: Optional[int] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    runtime_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Replace NaN with None for JSON
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
        return d


@dataclass
class FoldResult:
    """All results for one asset/fold combination."""
    asset: str
    fold: int
    train_dates: Tuple[str, str]
    val_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    baseline_results: List[ModelResult]
    hmm_results_all_starts: List[ModelResult]  # All K×seed combinations
    selected_hmm: Optional[ModelResult]  # The one chosen by validation
    test_results: List[ModelResult]  # Baselines + selected HMM on test


# ─── Canonical Runner ────────────────────────────────────────────────────

class CanonicalRunner:
    """Executes the canonical protocol and emits versioned artifacts."""

    def __init__(self, config: CanonicalConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"canonical_{self.timestamp}"
        self.output_dir = Path(config.output_root) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Manifest data
        self.manifest: Dict[str, Any] = {
            "protocol_version": config.protocol_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": config.to_dict(),
            "assets": list(config.assets),
            "folds_completed": [],
            "artifacts": {},
            "git_commit": self._get_git_commit(),
        }

    def _get_git_commit(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    def run(self) -> List[FoldResult]:
        """Execute the full canonical protocol for all assets."""
        all_fold_results = []

        for asset in self.config.assets:
            print(f"\n{'='*60}")
            print(f"Processing {asset}")
            print(f"{'='*60}")

            try:
                fold_result = self._process_asset(asset)
                all_fold_results.append(fold_result)
                self.manifest["folds_completed"].append({
                    "asset": asset,
                    "fold": 0,
                    "status": "completed",
                })
            except Exception as e:
                print(f"ERROR processing {asset}: {e}")
                import traceback
                traceback.print_exc()
                self.manifest["folds_completed"].append({
                    "asset": asset,
                    "fold": 0,
                    "status": "failed",
                    "error": str(e),
                })

        # Write manifest
        self._write_manifest()
        self._write_results_csv(all_fold_results)
        self._write_model_diagnostics_json(all_fold_results)

        print(f"\n{'='*60}")
        print(f"Canonical run complete. Output: {self.output_dir}")
        print(f"{'='*60}")

        return all_fold_results

    def _process_asset(self, asset: str) -> FoldResult:
        """Process a single asset through the full protocol."""
        # 1. Load data
        print(f"  Loading {asset} from {self.config.start_date} to {self.config.end_date}...")
        raw_df = load_asset(asset, self.config.start_date, self.config.end_date)
        raw_df = validate_data(raw_df)
        print(f"    Raw shape: {raw_df.shape}")

        # 2. Build features
        df, price_col = build_features(raw_df)
        print(f"    Feature shape: {df.shape}")

        # 3. Target-safe Train/Validation/Test split
        splits = target_safe_train_validation_test_split(
            df,
            validation_size=self.config.validation_size,
            test_size=self.config.test_size,
        )
        print(f"    Train: {len(splits.train)} ({splits.train.index.min().date()} to {splits.train.index.max().date()})")
        print(f"    Val:   {len(splits.validation)} ({splits.validation.index.min().date()} to {splits.validation.index.max().date()})")
        print(f"    Test:  {len(splits.test)} ({splits.test.index.min().date()} to {splits.test.index.max().date()})")

        # 4. Scale features (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[list(self.config.feature_cols)].values)
        X_val = scaler.transform(splits.validation[list(self.config.feature_cols)].values)
        X_test = scaler.transform(splits.test[list(self.config.feature_cols)].values)

        # 5. Run baselines on test set
        print("  Running baselines...")
        baseline_results = self._run_baselines(splits.train, splits.validation, splits.test)

        # 6. Multi-start HMM fitting for each K
        print("  Multi-start HMM fitting...")
        hmm_all_starts = self._run_hmm_multi_start(
            splits.train, splits.validation, splits.test,
            X_train, X_val, X_test
        )

        # 7. Select best HMM by validation log-likelihood
        converged_starts = [r for r in hmm_all_starts if r.converged]
        if not converged_starts:
            raise RuntimeError(f"No converged HMM fits for {asset}")

        # Select by validation LL (could also use AIC/BIC on validation)
        selected = max(converged_starts, key=lambda r: r.log_likelihood_val)
        selected.selected_by = "val_ll"
        print(f"    Selected: K={selected.k}, seed={selected.seed}, val_LL={selected.log_likelihood_val:.2f}")

        # 8. Evaluate selected HMM + baselines on TEST (locked model)
        print("  Evaluating on test set...")
        test_results = self._evaluate_on_test(
            splits.train, splits.test,
            X_train, X_test,
            baseline_results, selected
        )

        # Build fold result
        fold_result = FoldResult(
            asset=asset,
            fold=0,
            train_dates=(
                str(splits.train.index.min().date()),
                str(splits.train.index.max().date()),
            ),
            val_dates=(
                str(splits.validation.index.min().date()),
                str(splits.validation.index.max().date()),
            ),
            test_dates=(
                str(splits.test.index.min().date()),
                str(splits.test.index.max().date()),
            ),
            baseline_results=baseline_results,
            hmm_results_all_starts=hmm_all_starts,
            selected_hmm=selected,
            test_results=test_results,
        )

        # Save per-asset artifacts
        self._save_asset_artifacts(asset, fold_result, splits, price_col)

        return fold_result

    def _run_baselines(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> List[ModelResult]:
        """Run all baseline models. Returns results on validation for reference."""
        results = []
        train_mean = train_df["next_log_return"].mean()

        # Naive - train mean return
        start = time.time()
        pred = np.full(len(test_df), train_mean)
        res = evaluate_predictions("Naive - train mean", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Naive - persistence
        start = time.time()
        pred = test_df["log_return"].values
        res = evaluate_predictions("Naive - persistence", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Moving Average
        start = time.time()
        full_df = pd.concat([train_df, val_df, test_df]).copy()
        full_df["ma_pred"] = full_df["log_return"].rolling(self.config.ma_window).mean()
        pred = full_df.loc[test_df.index, "ma_pred"].fillna(train_mean).values
        res = evaluate_predictions(f"Moving Average {self.config.ma_window}", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Discrete Markov Chain
        start = time.time()
        train_curr_dir = direction_labels(train_df["log_return"].values)
        train_next = train_df["next_log_return"].values
        exp_next = {}
        for d in [0, 1]:
            vals = train_next[train_curr_dir == d]
            exp_next[d] = vals.mean() if len(vals) else train_mean
        pred = np.array([exp_next[d] for d in direction_labels(test_df["log_return"].values)])
        res = evaluate_predictions("Discrete Markov Chain", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        return results

    def _run_hmm_multi_start(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> List[ModelResult]:
        """Run HMM for all K × seed combinations. Evaluate on validation."""
        results = []

        for k in self.config.k_values:
            for seed in self.config.seeds:
                print(f"    K={k}, seed={seed}...", end=" ", flush=True)
                start = time.time()

                try:
                    model, fit_time = train_gaussian_hmm(
                        X_train,
                        n_states=k,
                        covariance_type=self.config.covariance_type,
                        n_iter=self.config.n_iter,
                        tol=self.config.tol,
                        random_state=seed,
                        verbose=False,
                    )
                    runtime = time.time() - start

                    # Diagnostics
                    diag = gaussian_hmm_diagnostics(model, X_train)

                    # Validation log-likelihood (for selection)
                    val_ll = float(model.score(X_val))
                    train_ll = float(model.score(X_train))
                    test_ll = float(model.score(X_test))

                    # Past-only state decoding on validation
                    train_states = model.predict(X_train)
                    val_states = decode_past_only_states(
                        model, X_train, X_val, self.config.context_window
                    )

                    # Validation predictions
                    val_pred = hmm_next_return_predictions(
                        model, train_df, train_states, val_states
                    )
                    val_res = evaluate_predictions(
                        f"HMM K={k} seed={seed}", val_df, val_pred, runtime,
                        train_ll, val_ll
                    )

                    result = ModelResult(
                        asset="", fold=0,
                        model_name=f"Gaussian HMM K={k} seed={seed}",
                        model_family="hmm",
                        k=k, seed=seed,
                        dpa=val_res["DPA_direction_accuracy"],
                        mae_return=val_res["MAE_return"],
                        rmse_return=val_res["RMSE_return"],
                        mape_price_pct=val_res["MAPE_price_%"],
                        log_likelihood_train=train_ll,
                        log_likelihood_val=val_ll,
                        log_likelihood_test=test_ll,
                        converged=diag["converged"],
                        iterations=diag["iterations"],
                        n_parameters=diag["n_parameters"],
                        aic=diag["aic"],
                        bic=diag["bic"],
                        runtime_sec=runtime,
                    )
                    results.append(result)
                    status = "✓" if diag["converged"] else "✗"
                    print(f"{status} val_LL={val_ll:.2f} converged={diag['converged']}")

                except Exception as e:
                    print(f"FAILED: {e}")
                    # Record failed attempt
                    results.append(ModelResult(
                        asset="", fold=0,
                        model_name=f"Gaussian HMM K={k} seed={seed}",
                        model_family="hmm",
                        k=k, seed=seed,
                        runtime_sec=time.time() - start,
                        converged=False,
                    ))

        return results

    def _evaluate_on_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        X_train: np.ndarray,
        X_test: np.ndarray,
        baseline_results: List[ModelResult],
        selected_hmm: ModelResult,
    ) -> List[ModelResult]:
        """Evaluate baselines + selected HMM on the held-out test set."""
        results = []

        # Re-run baselines on test (they're fast, ensures exact same test set)
        train_mean = train_df["next_log_return"].mean()

        # Naive train mean
        start = time.time()
        pred = np.full(len(test_df), train_mean)
        res = evaluate_predictions("Naive - train mean", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Persistence
        start = time.time()
        pred = test_df["log_return"].values
        res = evaluate_predictions("Naive - persistence", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Moving Average
        start = time.time()
        full_df = pd.concat([train_df, test_df]).copy()
        full_df["ma_pred"] = full_df["log_return"].rolling(self.config.ma_window).mean()
        pred = full_df.loc[test_df.index, "ma_pred"].fillna(train_mean).values
        res = evaluate_predictions(f"Moving Average {self.config.ma_window}", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Discrete Markov Chain
        start = time.time()
        train_curr_dir = direction_labels(train_df["log_return"].values)
        train_next = train_df["next_log_return"].values
        exp_next = {}
        for d in [0, 1]:
            vals = train_next[train_curr_dir == d]
            exp_next[d] = vals.mean() if len(vals) else train_mean
        pred = np.array([exp_next[d] for d in direction_labels(test_df["log_return"].values)])
        res = evaluate_predictions("Discrete Markov Chain", test_df, pred, time.time() - start)
        results.append(ModelResult(
            asset="", fold=0, model_name=res["model"], model_family="baseline",
            dpa=res["DPA_direction_accuracy"], mae_return=res["MAE_return"],
            rmse_return=res["RMSE_return"], mape_price_pct=res["MAPE_price_%"],
            runtime_sec=res["runtime_sec"],
        ))

        # Selected HMM on test (re-train with same seed on train, evaluate on test)
        print(f"    Evaluating selected HMM (K={selected_hmm.k}, seed={selected_hmm.seed}) on test...")
        model, fit_time = train_gaussian_hmm(
            X_train,
            n_states=selected_hmm.k,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=selected_hmm.seed,
            verbose=False,
        )
        train_states = model.predict(X_train)
        test_states = decode_past_only_states(model, X_train, X_test, self.config.context_window)
        test_pred = hmm_next_return_predictions(model, train_df, train_states, test_states)
        test_ll = float(model.score(X_test))
        train_ll = float(model.score(X_train))
        test_res = evaluate_predictions(
            f"Gaussian HMM K={selected_hmm.k} (selected)", test_df, test_pred,
            fit_time, train_ll, test_ll
        )

        results.append(ModelResult(
            asset="", fold=0,
            model_name=test_res["model"],
            model_family="hmm",
            k=selected_hmm.k,
            seed=selected_hmm.seed,
            selected_by=selected_hmm.selected_by,
            dpa=test_res["DPA_direction_accuracy"],
            mae_return=test_res["MAE_return"],
            rmse_return=test_res["RMSE_return"],
            mape_price_pct=test_res["MAPE_price_%"],
            log_likelihood_train=train_ll,
            log_likelihood_test=test_ll,
            converged=True,
            iterations=selected_hmm.iterations,
            n_parameters=selected_hmm.n_parameters,
            aic=selected_hmm.aic,
            bic=selected_hmm.bic,
            runtime_sec=test_res["runtime_sec"],
        ))

        return results

    def _save_asset_artifacts(
        self,
        asset: str,
        fold_result: FoldResult,
        splits: TargetSafeSplits,
        price_col: str,
    ) -> None:
        """Save per-asset plots and tables."""
        asset_dir = self.output_dir / asset
        asset_dir.mkdir(exist_ok=True)

        # State summary for selected HMM
        if fold_result.selected_hmm:
            # Re-train selected model on train to get states
            feature_cols = list(self.config.feature_cols)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(splits.train[feature_cols].values)

            model, _ = train_gaussian_hmm(
                X_train,
                n_states=fold_result.selected_hmm.k,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=fold_result.selected_hmm.seed,
                verbose=False,
            )
            train_states = model.predict(X_train)
            state_summary = summarize_states(splits.train, train_states)
            state_summary.to_csv(asset_dir / f"state_table_K{fold_result.selected_hmm.k}.csv", index=False)

            # Transition matrix
            transmat_df = pd.DataFrame(
                model.transmat_,
                index=[f"from_state_{i}" for i in range(fold_result.selected_hmm.k)],
                columns=[f"to_state_{i}" for i in range(fold_result.selected_hmm.k)],
            )
            transmat_df.to_csv(asset_dir / f"transition_matrix_K{fold_result.selected_hmm.k}.csv")

            # Transition analysis
            from src.analysis import analyze_transitions, plot_transition_heatmap, plot_state_characteristics
            state_names = [f"State {i}" for i in range(fold_result.selected_hmm.k)]
            analysis = analyze_transitions(model.transmat_, state_names)

            # Save analysis as JSON (convert numpy)
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    if obj.dtype.kind == 'c':
                        return [{"real": x.real, "imag": x.imag} for x in obj]
                    return obj.tolist()
                elif isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                return obj

            analysis_serializable = {
                "n_states": analysis["n_states"],
                "state_names": analysis["state_names"],
                "average_diagonal_probability": float(analysis["average_diagonal_probability"]),
                "average_row_entropy": float(analysis["average_row_entropy"]),
                "stationary_distribution": convert(analysis["stationary_distribution"]),
                "mean_recurrence_time": convert(analysis["mean_recurrence_time"]),
                "spectral_gap": float(analysis["spectral_gap"]),
                "eigenvalues": convert(analysis["eigenvalues"]),
            }
            with open(asset_dir / f"transition_analysis_K{fold_result.selected_hmm.k}.json", "w") as f:
                json.dump(analysis_serializable, f, indent=2)

            # Plots
            try:
                import matplotlib.pyplot as plt

                # Combined train+test states for plotting
                X_test = scaler.transform(splits.test[feature_cols].values)
                test_states = decode_past_only_states(model, X_train, X_test, self.config.context_window)

                plot_df = pd.concat([splits.train, splits.test]).copy()
                plot_df["state"] = np.concatenate([train_states, test_states])

                # Price colored by states
                plt.figure(figsize=(13, 5))
                sc = plt.scatter(plot_df.index, plot_df[price_col], c=plot_df["state"], s=8)
                plt.title(f"{asset} price colored by HMM states (K={fold_result.selected_hmm.k})")
                plt.xlabel("Date"); plt.ylabel("Price")
                plt.grid(True, alpha=0.3); plt.colorbar(sc, label="Hidden state")
                plt.tight_layout()
                plt.savefig(asset_dir / f"{asset}_price_states_K{fold_result.selected_hmm.k}.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Returns colored by states
                plt.figure(figsize=(13, 4))
                sc = plt.scatter(plot_df.index, plot_df["log_return"], c=plot_df["state"], s=8)
                plt.title(f"{asset} log returns colored by HMM states (K={fold_result.selected_hmm.k})")
                plt.xlabel("Date"); plt.ylabel("Log return")
                plt.grid(True, alpha=0.3); plt.colorbar(sc, label="Hidden state")
                plt.tight_layout()
                plt.savefig(asset_dir / f"{asset}_returns_states_K{fold_result.selected_hmm.k}.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Transition heatmap
                fig = plot_transition_heatmap(
                    model.transmat_, state_names,
                    title=f"Transition Matrix - {asset} HMM K={fold_result.selected_hmm.k}",
                    save_path=str(asset_dir / f"{asset}_transition_matrix_K{fold_result.selected_hmm.k}.png")
                )
                plt.close(fig)

                # State characteristics
                fig = plot_state_characteristics(
                    state_summary,
                    save_path=str(asset_dir / f"{asset}_state_characteristics_K{fold_result.selected_hmm.k}.png")
                )
                plt.close(fig)

            except Exception as e:
                print(f"    Warning: Plot generation failed for {asset}: {e}")

    def _write_manifest(self) -> None:
        """Write the canonical manifest."""
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        self.manifest["artifacts"]["manifest"] = "manifest.json"

    def _write_results_csv(self, all_fold_results: List[FoldResult]) -> None:
        """Write consolidated results CSV."""
        rows = []
        for fr in all_fold_results:
            for res in fr.test_results:
                row = res.to_dict()
                row["asset"] = fr.asset
                row["fold"] = fr.fold
                rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        self.manifest["artifacts"]["results_csv"] = "results.csv"

        # Also write per-asset test results
        for fr in all_fold_results:
            asset_rows = [r.to_dict() for r in fr.test_results]
            asset_df = pd.DataFrame(asset_rows)
            asset_df.to_csv(self.output_dir / fr.asset / "test_results.csv", index=False)

    def _write_model_diagnostics_json(self, all_fold_results: List[FoldResult]) -> None:
        """Write complete HMM diagnostics for all starts."""
        diag = {}
        for fr in all_fold_results:
            asset_diag = {}
            for res in fr.hmm_results_all_starts:
                key = f"K={res.k}_seed={res.seed}"
                asset_diag[key] = {
                    "k": res.k,
                    "seed": res.seed,
                    "converged": res.converged,
                    "iterations": res.iterations,
                    "train_log_likelihood": res.log_likelihood_train,
                    "val_log_likelihood": res.log_likelihood_val,
                    "test_log_likelihood": res.log_likelihood_test,
                    "n_parameters": res.n_parameters,
                    "aic": res.aic,
                    "bic": res.bic,
                    "dpa_val": res.dpa,
                    "runtime_sec": res.runtime_sec,
                }
            if fr.selected_hmm:
                asset_diag["selected"] = {
                    "k": fr.selected_hmm.k,
                    "seed": fr.selected_hmm.seed,
                    "selected_by": fr.selected_hmm.selected_by,
                }
            diag[fr.asset] = asset_diag

        diag_path = self.output_dir / "model_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)
        self.manifest["artifacts"]["model_diagnostics"] = "model_diagnostics.json"


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    """Run the canonical protocol with default configuration."""
    config = CanonicalConfig()
    runner = CanonicalRunner(config)
    results = runner.run()

    # Print summary
    print("\n" + "="*60)
    print("CANONICAL RUN SUMMARY")
    print("="*60)
    for fr in results:
        print(f"\n{fr.asset}:")
        for tr in fr.test_results:
            sel = " ← SELECTED" if tr.model_family == "hmm" and tr.k == fr.selected_hmm.k else ""
            print(f"  {tr.model_name:40s} DPA={tr.dpa:.4f} MAE={tr.mae_return:.6f}{sel}")


if __name__ == "__main__":
    main()