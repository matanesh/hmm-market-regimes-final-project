# Gaussian HMM for Market Regime Analysis

**Matan Eshel**
Student ID: 203502802
Advanced Methods in Machine Learning
Afeka Academic College of Engineering

---

## 1. Project Overview

This project uses Gaussian Hidden Markov Models (HMMs) to identify latent market regimes and examines next-day return-direction prediction. Gaussian HMMs with \(K=2,3,4\) hidden states are compared with transparent baselines and supervised machine-learning models under a strict chronological evaluation protocol that prevents data leakage.

**Main Finding:** The HMM did not demonstrate a consistent advantage for next-day directional prediction (DPA 52–55%). However, it provided a useful and interpretable representation of latent market regimes, including differences in return behavior, volatility, drawdown, state persistence, and cross-asset correlations.

**Reinforcement Learning (RL)** is presented only as future work and is **not** part of the reported experiments.

---

## 2. Final Dataset / Assets

The final 9-asset universe:
- **SPY** — broad U.S. equities (reference market)
- **QQQ** — growth/technology-heavy equities
- **IWM** — U.S. small caps
- **TLT** — long-duration U.S. Treasuries
- **GLD** — gold
- **HYG** — high-yield corporate credit
- **BTC-USD** — high-volatility alternative asset
- **JPM** — large financial equity
- **NVDA** — high-beta growth equity

Date range: **2014-01-01 to 2024-12-31** (pre-specified, not selected by test performance).

---

## 3. Features

Four contemporaneous features known by the current day's close:
| Feature | Description |
|---------|-------------|
| `log_return` | log(Adj Close / previous Adj Close) |
| `rolling_volatility_20` | 20-day rolling standard deviation of log returns |
| `daily_range` | (High - Low) / Close |
| `volume_change` | log(Volume / previous Volume) |

**Targets (never used as HMM features):**
- `next_log_return` — next day's log return
- `next_close` — next day's closing price

---

## 4. Models and Baselines

### HMM
- Gaussian HMM with `full` covariance (hmmlearn 0.3.3)
- \(K \in \{2,3,4\}\) with seeds \(\{42, 123, 456, 789, 2026\}\)
- Model selection **only on validation log-likelihood** (never test)
- Past-only state decoding (no future peeking)
- Refitted on Train+Validation before final test evaluation

### Baselines (evaluated on identical test partition)
1. **Naive - train mean** — constant prediction = train mean next-day return
2. **Naive - persistence** — predict today's return for tomorrow
3. **Moving Average 5** — rolling 5-day mean of returns
4. **Discrete Markov Chain** — 2-state first-order chain on daily direction

### Supervised ML Baselines (pre-specified, no tuning on test)
- **Logistic Regression** — C=1.0, max_iter=2000, scaled on train only
- **Random Forest** — 500 trees, max_depth=6, min_samples_leaf=10
- **HistGradientBoosting** — 200 iterations, learning_rate=0.05, max_leaf_nodes=15

### Metrics
- **DPA** — Directional Prediction Accuracy (primary)
- **Balanced Accuracy, ROC-AUC, Log Loss, Brier Score** — classification
- **MAE, RMSE, MAPE (price %)** — return/price magnitude

---

## 5. Repository Structure

```
├── HMM_Market_Regimes_Project.ipynb   # Main notebook (uses saved artifacts by default)
├── requirements.txt                   # Exact pinned dependencies
├── run_extended_analysis.py           # Full HMM robustness experiment (9 assets, 5 seeds)
├── run_supervised_baselines.py        # Supervised baselines experiment
├── analyze_extended_context.py        # Post-hoc VIX / duration diagnostics
├── src/
│   ├── __init__.py
│   ├── data.py                        # Loading, features, target-safe splits
│   ├── model.py                       # HMM training, decoding, prediction
│   ├── evaluation.py                  # Metrics, state summaries, posterior diagnostics
│   └── analysis.py                    # Transition analysis, plotting utilities
├── experiments_extended/
│   └── extended_20260811_121957/      # FINAL HMM artifacts (figures, CSVs, JSONs)
│       ├── SPY/                       #   Per-asset artifacts
│       ├── QQQ/
│       ├── IWM/
│       ├── TLT/
│       ├── GLD/
│       ├── HYG/
│       ├── BTC-USD/
│       ├── JPM/
│       ├── NVDA/
│       ├── seed_stability_all_assets.csv
│       ├── test_metrics_all_assets.csv
│       ├── cross_asset_by_reference_state.csv
│       ├── vix_by_spy_regime.csv
│       ├── manifest.json
│       └── *.png                      # All figures referenced in report
├── experiments_supervised/
│   └── supervised_20260811_133344/    # FINAL supervised baselines artifacts
│       ├── supervised_results_all_assets.csv
│       ├── supervised_summary.csv
│       ├── comparison_with_hmm.csv
│       ├── dpa_comparison_all_assets.png
│       └── manifest.json
├── reports/
│   ├── report.pdf                     # Final compiled PDF
│   ├── report.tex                     # LaTeX source (portable: Noto Serif Hebrew / David CLM)
│   ├── references.bib
│   └── sections/                      # Modular .tex sections
│       ├── 00_abstract.tex
│       ├── 01_intro.tex
│       ├── 02_data_protocol.tex
│       ├── 03_method.tex
│       ├── 04_experiments.tex
│       ├── 05_regime_results.tex
│       ├── 06_prediction_results.tex
│       ├── 07_discussion.tex
│       ├── 08_rl_future_work.tex
│       ├── 09_limitations_conclusion.tex
│       └── 10_reproducibility.tex
└── tests/
    ├── test_protocol_foundation.py
    ├── test_canonical_runner.py
    ├── test_supervised_baselines.py
    └── test_regime_extension.py
```

---

## 6. QUICK REVIEW PATH

Easiest way for a lecturer to inspect the work:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

jupyter notebook HMM_Market_Regimes_Project.ipynb
```

The notebook **uses saved verified experiment artifacts by default** (no network, no retraining required). It loads `experiments_extended/extended_20260811_121957/` and `experiments_supervised/supervised_20260811_133344/`, validates their integrity, and displays all tables, figures, and findings.

---

## Run in Google Colab

### Recommended: Run in Google Colab

For the easiest platform-independent review, use Google Colab.
This avoids local Python, Windows, WSL, and dependency configuration issues.

Open the complete project notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matanesh/hmm-market-regimes-final-project/blob/main/HMM_Market_Regimes_Project_Colab.ipynb)

This Colab notebook contains the **complete scientific notebook** plus the
environment setup required for Colab. It is **not just a launcher** — the
full analysis, code, tables, figures, and conclusions are all in this
single notebook.

### How to use

1. Open the Colab link above.
2. Select **Runtime → Run all**.
3. Wait for setup (repository clone, dependency install, tests).
4. Continue directly through the complete scientific notebook.
5. See the full analysis, code, tables, figures, and conclusions in this same notebook.

No local environment, Windows/WSL configuration, or manual file uploads are required.

---

## Recommended Review Order

1. **Fastest** — view the already-executed
   `HMM_Market_Regimes_Project.ipynb` directly on GitHub (saved outputs visible).

2. **Recommended reproducibility check** — use **Run in Google Colab** (badge above).
   The Colab notebook contains the complete scientific notebook plus setup.

3. **Advanced / full reproduction** — clone the repository locally and run:
   ```bash
   python run_extended_analysis.py
   ```

Google Colab is recommended for reviewing and verifying the notebook
because it provides a clean and consistent environment.

> **Note:** Colab is not necessarily the best environment for the complete
> 9-asset, 5-seed experiment, because runtime limits and Yahoo Finance
> network availability may vary. The Colab notebook includes optional
> cells for full re-training (disabled by default).

---

## 7. VERIFY NOTEBOOK

Execute the notebook top-to-bottom programmatically:

```bash
jupyter nbconvert \
  --to notebook \
  --execute HMM_Market_Regimes_Project.ipynb \
  --output /tmp/HMM_verified.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

Requirements:
- Exit code 0
- No tracebacks
- All figures, CSVs, JSONs load correctly
- No absolute paths (e.g., `/root/...`)
- No network required to view final results
- Notebook displays the 9-asset version

---

## 8. RUN TESTS

```bash
python -m pytest tests/ -q
```

Tests cover:
- Target-safe chronological splitting (no boundary leakage)
- Feature scaler fitted on train only
- Multi-start HMM selection on validation only
- Past-only state decoding
- Supervised baselines scaling and evaluation
- Posterior entropy/confidence diagnostics
- Expanding-window robustness folds

---

## 9. FULL EXPERIMENT RE-RUN

The final artifacts are already included. New runs create **new timestamped directories** and **do not modify** the artifacts the report is based on.

### HMM Extended Analysis (9 assets, 5 seeds, walk-forward on SPY)
```bash
python run_extended_analysis.py
```

Output: `experiments_extended/extended_<timestamp>/`

### Supervised Baselines
```bash
python run_supervised_baselines.py
```

Output: `experiments_supervised/supervised_<timestamp>/`

### Post-hoc Context Diagnostics (VIX, state duration)
```bash
python analyze_extended_context.py --run-dir experiments_extended/<new_run_id>
```

---

## 10. BUILD REPORT (LaTeX → PDF)

```bash
cd reports
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

Produces `reports/report.pdf`.
The LaTeX source uses a portable font setup:
- **Default:** FreeSerif (standard in TeX Live via `fonts-freefont-ttf`)
- **Monospace:** DejaVu Sans Mono (standard in TeX Live)
- **Optional local override:** David CLM (if installed locally)

The `report.tex` defines `\hebrewmainfont` as `FreeSerif` by default. If David CLM is available on your system, uncomment the relevant lines in `report.tex` to use it instead.

---

## 11. Final Verified Artifacts

The report and notebook are based on these exact directories:

| Directory | Description |
|-----------|-------------|
| `experiments_extended/extended_20260811_121957/` | HMM results: state tables, transition matrices, posterior entropy, VIX validation, cross-asset correlations, seed stability ARI, walk-forward DPA |
| `experiments_supervised/supervised_20260811_133344/` | Supervised baselines: per-asset DPA/ROC-AUC/Balanced Acc/Log Loss/Brier, comparison with HMM, summary figure |

All figures referenced in `reports/sections/*.tex` exist in these directories.

---

## 12. Scientific Note

**This project makes no profitability claim and does not constitute investment advice.**
The experimental design prevents data leakage via chronological target-safe splits. Train/Validation/Test partitions are strictly ordered. Model selection uses validation criteria only. RL is future work only and was not evaluated.

---

## 13. Reproducibility Notes

- **Python 3.11+** (tested on 3.11.15)
- **Exact dependency versions** in `requirements.txt`
- **Git commit** recorded in experiment manifests
- **Random seeds** fixed for all stochastic components
- **No external API keys** required (Yahoo Finance via `yfinance` only)
- **All final artifacts included** — rerunning experiments is optional for review
