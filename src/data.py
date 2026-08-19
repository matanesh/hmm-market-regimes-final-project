"""
Data loading, feature construction and chronological splitting utilities.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class TargetSafeSplits:
    """Chronological partitions whose pre-segment boundary targets are removed."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten multi-index columns returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def _cache_path(
    cache_dir: str,
    ticker: str,
    start_date: str,
    end_date: Optional[str],
) -> Path:
    """Return a range-specific CSV cache path for one downloaded asset."""
    safe_ticker = ticker.replace("/", "_").replace("^", "index_")
    safe_start = start_date.replace("-", "")
    safe_end = (end_date or "latest").replace("-", "")
    return Path(cache_dir) / f"{safe_ticker}_{safe_start}_{safe_end}.csv"


def load_asset(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load historical OHLCV data, optionally using a reproducible local cache.

    The cache is intentionally range-specific.  Once a download succeeds, later
    notebook/report reruns can reuse the exact CSV instead of repeatedly depending
    on Yahoo Finance availability or rate limits.
    """
    cache_file = None
    if cache_dir:
        cache_file = _cache_path(cache_dir, ticker, start_date, end_date)
        if cache_file.exists() and not force_download:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached.index.name = "Date"
            if cached.empty:
                raise ValueError(f"Cached data for {ticker} is empty: {cache_file}")
            return cached.sort_index()

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    df = flatten_yfinance_columns(df, ticker).sort_index()
    if df.empty:
        cache_hint = f" No cache was available at {cache_file}." if cache_file else ""
        raise ValueError(
            f"No data was downloaded for ticker {ticker}. "
            f"Check ticker/date/internet connection or Yahoo rate limits.{cache_hint}"
        )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite values and remove incomplete OHLCV rows."""
    return df.replace([np.inf, -np.inf], np.nan).dropna().copy()


def build_features(df: pd.DataFrame, price_col: str = "Adj Close") -> Tuple[pd.DataFrame, str]:
    """Build leakage-safe contemporaneous features and next-day targets.

    HMM features use information known by the current day's close:
    log return, 20-day rolling volatility, intraday range and volume change.
    ``next_log_return`` and ``next_close`` are targets only and are never included
    in the HMM feature matrix.
    """
    if price_col not in df.columns:
        price_col = "Close" if "Close" in df.columns else df.columns[-1]

    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["rolling_volatility_20"] = df["log_return"].rolling(20).std()
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df["target_date"] = pd.Series(df.index, index=df.index).shift(-1)
    df["next_log_return"] = df["log_return"].shift(-1)
    df["next_close"] = df[price_col].shift(-1)
    df["current_close"] = df[price_col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df, price_col


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations chronologically without shuffling."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _validate_target_safe_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_date_col: str,
) -> TargetSafeSplits:
    """Validate chronology and next-target boundaries for three partitions."""
    partitions = (train, validation, test)
    if any(part.empty for part in partitions):
        raise ValueError("split sizes leave an empty target-safe partition")
    if any(
        not part.index.is_unique or not part.index.is_monotonic_increasing
        for part in partitions
    ):
        raise ValueError("partitions must have strictly increasing, unique indexes")
    if not train.index.max() < validation.index.min() < test.index.min():
        raise ValueError("partitions must be chronological and non-overlapping")

    target_dates = [part[target_date_col] for part in partitions]
    if any(dates.isna().any() for dates in target_dates):
        raise ValueError("target dates must be present for every retained row")
    for dates, part in zip(target_dates, partitions):
        if (dates <= part.index.to_series()).any():
            raise ValueError("each target date must be later than its feature date")
    if train[target_date_col].max() >= validation.index.min():
        raise ValueError("training targets must precede validation observations")
    if validation[target_date_col].max() >= test.index.min():
        raise ValueError("validation targets must precede test observations")

    return TargetSafeSplits(train=train, validation=validation, test=test)


def target_safe_train_validation_test_split(
    df: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    target_date_col: str = "target_date",
) -> TargetSafeSplits:
    """Create one chronological Train/Validation/Test split without target leakage."""
    if not df.index.is_unique:
        raise ValueError("time index must be unique")
    if not df.index.is_monotonic_increasing:
        raise ValueError("time index must be strictly increasing")
    if target_date_col not in df.columns:
        raise ValueError(f"missing target date column: {target_date_col}")
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    train_end = int(len(df) * (1 - validation_size - test_size))
    validation_end = int(len(df) * (1 - test_size))
    if train_end < 2 or validation_end - train_end < 2 or len(df) - validation_end < 1:
        raise ValueError("split sizes leave an empty target-safe partition")

    # Omit the row immediately before each new segment: its next-day target lies
    # at the boundary and would otherwise cross from one partition into the next.
    train = df.iloc[: train_end - 1].copy()
    validation = df.iloc[train_end : validation_end - 1].copy()
    test = df.iloc[validation_end:].copy()
    return _validate_target_safe_partitions(train, validation, test, target_date_col)


def expanding_window_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    initial_train_fraction: float = 0.55,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.10,
    target_date_col: str = "target_date",
) -> List[TargetSafeSplits]:
    """Create chronological expanding-window folds for robustness analysis.

    Example with the defaults and three folds (approximately):

    - Fold 1: train 0-55%, validation 55-70%, test 70-80%
    - Fold 2: train 0-65%, validation 65-80%, test 80-90%
    - Fold 3: train 0-75%, validation 75-90%, test 90-100%

    Earlier held-out observations may legitimately become training/validation data
    in later folds, as they would in a real expanding-history workflow.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if target_date_col not in df.columns:
        raise ValueError(f"missing target date column: {target_date_col}")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("time index must be unique and increasing")
    fractions = (initial_train_fraction, validation_fraction, test_fraction)
    if any(x <= 0 or x >= 1 for x in fractions):
        raise ValueError("split fractions must lie between zero and one")
    if sum(fractions) > 1:
        raise ValueError("initial train + validation + test fractions cannot exceed one")

    n = len(df)
    initial_train_len = int(n * initial_train_fraction)
    val_len = max(2, int(n * validation_fraction))
    test_len = max(1, int(n * test_fraction))
    required = initial_train_len + val_len + test_len
    if required > n:
        raise ValueError("not enough observations for requested expanding-window split")

    if n_splits == 1:
        step = 0
    else:
        available_shift = n - required
        step = available_shift // (n_splits - 1)
        if step < 1:
            raise ValueError("not enough remaining history to create distinct folds")

    folds: List[TargetSafeSplits] = []
    for fold_idx in range(n_splits):
        train_end = initial_train_len + fold_idx * step
        val_end = train_end + val_len
        test_end = val_end + test_len
        if fold_idx == n_splits - 1:
            test_end = min(n, test_end)
        if test_end > n:
            break

        train = df.iloc[: train_end - 1].copy()
        validation = df.iloc[train_end : val_end - 1].copy()
        test = df.iloc[val_end:test_end].copy()
        folds.append(
            _validate_target_safe_partitions(train, validation, test, target_date_col)
        )

    if len(folds) != n_splits:
        raise ValueError(f"created {len(folds)} folds, expected {n_splits}")
    return folds
