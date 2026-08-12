# Gaussian HMM for Market Regime Analysis

**Student:** Matan Eshel / מתן אשל  
**Student ID:** 203502802  
**Course:** שיטות מתקדמות בלמידת מכונה

## Project Overview

This project uses Gaussian Hidden Markov Models (HMMs) to identify latent market regimes and to examine next-day return-direction prediction. Gaussian HMMs with \(K=2,3,4\) hidden states are compared with transparent baselines and supervised machine-learning models under a chronological evaluation protocol.

## Main Finding

The HMM did not demonstrate a consistent advantage for next-day prediction. However, it provided a useful and interpretable representation of latent market regimes, including differences in return behavior, volatility, and state persistence.

Reinforcement Learning (RL) is presented only as future work and is **not** part of the experiments reported in this project.

## Main Files

- `HMM_Market_Regimes_Project.ipynb` — Main executed notebook containing the code, experiments, visualizations, and results.
- `reports/report.pdf` — Final project report.
- `reports/report.tex` — LaTeX source of the report.
- `requirements.txt` — Python dependencies required to run the project.
- `run_canonical.py` — Reproduces the main chronological HMM experiment and creates versioned experiment outputs.
- `src/data.py`, `src/model.py`, `src/evaluation.py`, `src/analysis.py` — Data preparation, HMM training, evaluation metrics, and regime-analysis utilities.
- `model_comparison.py` — Comparison workflow for the HMM, baseline methods, and supervised ML models.
- `experiments_canonical/` and related experiment-output directories — Saved results and artifacts on which the notebook and report are based.

## Reproducibility

1. Install the dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Open `HMM_Market_Regimes_Project.ipynb` in Jupyter Notebook or JupyterLab.
3. Run `python run_canonical.py` to reproduce the central HMM experiment; use `model_comparison.py` for the model-comparison workflow.
4. Final experiment artifacts are already included in the repository, so rerunning the experiments is not required to review the reported results.

## Important Notes

- The project makes **no profitability claim** and does not constitute investment advice.
- The experimental design is intended to prevent data leakage.
- Train, validation, and test partitions are chronological.
- RL is future work only and was not evaluated in the reported experiments.
