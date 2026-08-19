#!/usr/bin/env python3
"""External-context diagnostics for an executed extended HMM run.

This script intentionally operates *after* the HMM experiment.  It does not use
VIX or duration diagnostics to fit/select the model.  Instead, it asks two
external-validity/model-adequacy questions:

1. Do SPY regimes learned only from SPY's own OHLCV-derived features correspond
   to materially different VIX environments?
2. Does the ordinary first-order HMM's geometric state-duration assumption
   resemble the empirical dwell times observed in decoded regimes?

Because both analyses are post-hoc diagnostics, they can strengthen or weaken
interpretation without contaminating the held-out prediction protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _latest_extended_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("extended_*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No extended_* directory found under {root}")
    return runs[-1]


def _download_or_load_vix(
    start_date: str,
    end_date: str,
    cache_dir: Path,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load VIX Close from a range-specific cache or Yahoo Finance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"VIX_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
    )
    if cache_path.exists() and not force_download:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return frame.sort_index()

    raw = yf.download(
        "^VIX",
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no ^VIX data")
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw = raw.xs("^VIX", axis=1, level=-1)
        except Exception:
            raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw.columns:
        raise ValueError("^VIX download does not contain a Close column")

    frame = raw[["Close"]].rename(columns={"Close": "vix_close"}).dropna().copy()
    frame["vix_log_change"] = np.log(frame["vix_close"] / frame["vix_close"].shift(1))
    frame.to_csv(cache_path)
    return frame.sort_index()


def _vix_by_spy_regime(
    run_dir: Path,
    vix: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    posterior_path = run_dir / "SPY" / "posterior_daily.csv"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing SPY posterior output: {posterior_path}")

    spy = pd.read_csv(posterior_path, index_col=0, parse_dates=True)
    required = {"state_posterior_argmax", "posterior_entropy", "log_return"}
    missing = required.difference(spy.columns)
    if missing:
        raise ValueError(f"SPY posterior output missing columns: {sorted(missing)}")

    joined = spy.join(vix, how="inner").dropna(subset=["vix_close"])
    if joined.empty:
        raise ValueError("SPY test dates and VIX dates have no overlap")

    rows: List[Dict[str, float]] = []
    for state, part in joined.groupby("state_posterior_argmax"):
        rows.append(
            {
                "spy_state": int(state),
                "n_days": int(len(part)),
                "mean_vix": float(part["vix_close"].mean()),
                "median_vix": float(part["vix_close"].median()),
                "p90_vix": float(part["vix_close"].quantile(0.90)),
                "mean_vix_log_change_%": 100 * float(part["vix_log_change"].mean()),
                "share_vix_above_20_%": 100 * float((part["vix_close"] >= 20).mean()),
                "share_vix_above_30_%": 100 * float((part["vix_close"] >= 30).mean()),
                "mean_spy_abs_return_%": 100 * float(part["log_return"].abs().mean()),
                "mean_spy_posterior_entropy": float(part["posterior_entropy"].mean()),
            }
        )

    entropy = joined["posterior_entropy"].to_numpy(dtype=float)
    vix_level = joined["vix_close"].to_numpy(dtype=float)
    abs_vix_change = joined["vix_log_change"].abs().to_numpy(dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    q25, q75 = np.quantile(entropy, [0.25, 0.75])
    low = entropy <= q25
    high = entropy >= q75
    overall = {
        "n_overlapping_days": int(len(joined)),
        "entropy_vix_level_correlation": corr(entropy, vix_level),
        "entropy_abs_vix_change_correlation": corr(entropy, abs_vix_change),
        "low_entropy_quartile_mean_vix": float(vix_level[low].mean()),
        "high_entropy_quartile_mean_vix": float(vix_level[high].mean()),
        "low_entropy_quartile_mean_abs_vix_change_%": 100
        * float(abs_vix_change[low].mean()),
        "high_entropy_quartile_mean_abs_vix_change_%": 100
        * float(abs_vix_change[high].mean()),
    }
    return pd.DataFrame(rows).sort_values("spy_state"), overall


def _duration_diagnostics(run_dir: Path) -> pd.DataFrame:
    """Compare empirical dwell times to HMM-implied geometric means.

    For a first-order HMM, if a state's self-transition probability is a_ii,
    its state duration is geometric with expected length 1/(1-a_ii).  Large
    systematic discrepancies between that expectation and decoded empirical
    dwell times point to a duration-model limitation and motivate Hidden
    semi-Markov models as a possible extension.
    """
    rows: List[Dict[str, float]] = []
    for asset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        transition_path = asset_dir / "transition_matrix.csv"
        state_path = asset_dir / "state_summary_history.csv"
        if not transition_path.exists() or not state_path.exists():
            continue

        transition = pd.read_csv(transition_path, index_col=0).to_numpy(dtype=float)
        states = pd.read_csv(state_path)
        if "state" not in states.columns or "avg_duration_days" not in states.columns:
            continue

        for _, state_row in states.iterrows():
            state = int(state_row["state"])
            pii = float(transition[state, state])
            implied = float(1.0 / max(1.0 - pii, 1e-12))
            empirical = float(state_row["avg_duration_days"])
            rows.append(
                {
                    "asset": asset_dir.name,
                    "state": state,
                    "self_transition_probability": pii,
                    "hmm_implied_mean_duration_days": implied,
                    "decoded_empirical_mean_duration_days": empirical,
                    "empirical_to_implied_duration_ratio": empirical / implied
                    if implied > 0
                    else np.nan,
                    "absolute_duration_gap_days": abs(empirical - implied),
                }
            )

    return pd.DataFrame(rows)


def _save_plots(run_dir: Path, vix_by_state: pd.DataFrame, duration: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt

        if not vix_by_state.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(vix_by_state["spy_state"].astype(str), vix_by_state["mean_vix"])
            ax.set_xlabel("SPY hidden state")
            ax.set_ylabel("Mean VIX level")
            ax.set_title("External validation: VIX level conditional on SPY HMM regime")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "vix_by_spy_regime.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

        if not duration.empty:
            plot = duration.copy()
            labels = plot["asset"].astype(str) + " S" + plot["state"].astype(str)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.scatter(
                plot["hmm_implied_mean_duration_days"],
                plot["decoded_empirical_mean_duration_days"],
            )
            maximum = float(
                np.nanmax(
                    [
                        plot["hmm_implied_mean_duration_days"].max(),
                        plot["decoded_empirical_mean_duration_days"].max(),
                    ]
                )
            )
            ax.plot([0, maximum], [0, maximum], linestyle="--", linewidth=1)
            for x, y, label in zip(
                plot["hmm_implied_mean_duration_days"],
                plot["decoded_empirical_mean_duration_days"],
                labels,
            ):
                ax.annotate(label, (x, y), fontsize=7, alpha=0.8)
            ax.set_xlabel("HMM-implied mean duration 1/(1-a_ii)")
            ax.set_ylabel("Decoded empirical mean duration")
            ax.set_title("State-duration adequacy of the first-order HMM")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "duration_model_diagnostic.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"Plot warning: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific experiments_extended/extended_* directory. Default: latest.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Refresh cached ^VIX observations.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_extended_run(Path("experiments_extended"))
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extended manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    config = manifest["config"]
    vix = _download_or_load_vix(
        config["start_date"],
        config["end_date"],
        Path(config.get("cache_dir", "data_cache")),
        force_download=args.force_download,
    )
    vix_by_state, vix_overall = _vix_by_spy_regime(run_dir, vix)
    duration = _duration_diagnostics(run_dir)

    vix_by_state.to_csv(run_dir / "vix_by_spy_regime.csv", index=False)
    with (run_dir / "vix_external_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(vix_overall, handle, indent=2)
    duration.to_csv(run_dir / "state_duration_diagnostics.csv", index=False)
    _save_plots(run_dir, vix_by_state, duration)

    print(f"External-context diagnostics written to {run_dir}")
    print("VIX by SPY regime:")
    print(vix_by_state.to_string(index=False))
    print("\nVIX/entropy summary:")
    print(json.dumps(vix_overall, indent=2))
    if not duration.empty:
        print("\nLargest duration-model gaps:")
        print(
            duration.sort_values("absolute_duration_gap_days", ascending=False)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
