"""Comparison stage.

Loads the two provider CSVs for a ticker, joins them on date, and flags
rows where the cumulative (normalized) total-return factor disagrees
between providers.

What each provider's "adjusted close" actually contains
-------------------------------------------------------
Empirically verified against the raw CSVs in ``data/raw/``:

* ``yfinance`` ``Close``     -- **split-adjusted**, NOT dividend-adjusted
  (despite ``auto_adjust=False``; recent yfinance versions silently
  apply split adjustments to the "unadjusted" Close column).
* ``yfinance`` ``Adj Close`` -- split AND dividend adjusted, using
  Yahoo's *subtractive* convention (each historical price reduced by
  the cash dividend amount on every prior ex-date).
* ``stooq``    ``Close``     -- split AND dividend adjusted, using a
  *multiplicative* (CRSP-style) convention: each prior price multiplied
  by ``(close_on_ex - div) / close_on_ex``.

So the two providers' fully-adjusted total-return series are computed
by two different industry-standard methods. For a non-dividend stock
(AMZN, GOOG, META) they agree to within rounding. For a high-yield
stock (PFE, KHC, MO) they differ by an amount roughly proportional to
the cumulative dividend yield -- a real, measurable, methodology-driven
disagreement that anyone running a backtest off these providers needs
to know about.

Adjustment-factor convention used here
--------------------------------------
For each provider we normalize its *fully adjusted* close to its most
recent value:

    adj_factor[t] = adjusted_close[t] / adjusted_close[latest]

For Yahoo that means ``yahoo_adj_close`` (Yahoo's subtractive total-
return). For Stooq it's the ``stooq_close`` column directly (Stooq's
multiplicative total-return). Both factors are unitless cumulative
total-return curves anchored at 1.0 today; their disagreement isolates
provider-methodology drift.

Two flagging rules are applied independently and OR'd together:

* **Statistical:** |z-score of discrepancy| > 3.
* **Threshold:**   raw absolute discrepancy > 0.01 (i.e. > 1%).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from . import PROCESSED_DIR, RAW_DIR, SPLITS_DIR, TICKERS, get_universe

logger = logging.getLogger(__name__)

Z_THRESHOLD: float = 3.0
RAW_THRESHOLD: float = 0.01


def split_only_factor(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Compute a split-only cumulative adjustment factor.

    Reserved for future split-isolated analyses. Each split with ratio
    ``r`` on date ``d`` divides every observation strictly before ``d``
    by ``r``; the resulting series is then normalized by its most recent
    value. Not used in the default comparison flow because both
    providers' ``Close``/``Adj Close`` columns already incorporate
    splits at source.

    Args:
        close: Daily close series. Pass a raw (unadjusted) close to get
            a meaningful split-only factor; passing an already-adjusted
            close together with the same provider's splits would
            double-count.
        splits: Per-date split-ratio series. May be empty.

    Returns:
        Series of the same shape as ``close`` containing the split-only
        adjustment factor.
    """
    if close.empty:
        return close.astype(float)
    adj = close.astype(float).copy()
    if not splits.empty:
        for split_date, ratio in splits.items():
            ratio = float(ratio)
            if ratio <= 0 or not np.isfinite(ratio):
                continue
            mask = adj.index < pd.Timestamp(split_date)
            adj.loc[mask] = adj.loc[mask] / ratio
    latest = float(adj.iloc[-1])
    if latest == 0 or not np.isfinite(latest):
        return pd.Series(np.nan, index=adj.index)
    return adj / latest


def _normalize_to_latest(s: pd.Series) -> pd.Series:
    """Normalize a series so its most recent value is 1.0.

    Args:
        s: Numeric series (e.g. an adjusted-close column).

    Returns:
        ``s / s.iloc[-1]`` if the most recent value is finite and
        non-zero; otherwise a NaN series of the same shape.
    """
    if s.empty:
        return s.astype(float)
    latest = float(s.iloc[-1])
    if latest == 0 or not np.isfinite(latest):
        return pd.Series(np.nan, index=s.index)
    return s.astype(float) / latest


def load_provider_data(
    ticker: str,
    raw_dir: Path = RAW_DIR,
    splits_dir: Path = SPLITS_DIR,
) -> pd.DataFrame:
    """Load both providers' CSVs and compute total-return adjustment factors.

    Each provider's *fully adjusted* close column is normalized to its
    most recent value to produce a cumulative total-return factor. The
    factor columns stored by ``src.ingest`` are overwritten here so that
    the comparison always uses the canonical formula regardless of what
    the ingest stage happened to store.

    Args:
        ticker: Equity ticker.
        raw_dir: Directory containing the raw provider CSVs.
        splits_dir: Reserved for future use (split-only audits).

    Returns:
        DataFrame indexed by date joined on the intersection of trading
        days, with ``yahoo_adj_factor`` (from Yahoo's ``Adj Close``) and
        ``stooq_adj_factor`` (from Stooq's ``Close``).

    Raises:
        FileNotFoundError: if either provider CSV is missing.
    """
    yahoo_path = raw_dir / f"{ticker}_yahoo.csv"
    stooq_path = raw_dir / f"{ticker}_stooq.csv"
    if not yahoo_path.exists():
        raise FileNotFoundError(yahoo_path)
    if not stooq_path.exists():
        raise FileNotFoundError(stooq_path)

    yahoo = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)
    stooq = pd.read_csv(stooq_path, index_col=0, parse_dates=True)
    yahoo.index.name = "date"
    stooq.index.name = "date"

    yahoo["yahoo_adj_factor"] = _normalize_to_latest(yahoo["yahoo_adj_close"])
    stooq["stooq_adj_factor"] = _normalize_to_latest(stooq["stooq_close"])

    merged = yahoo.join(stooq, how="inner")
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


def compute_discrepancy(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-day absolute adjustment-factor discrepancy + z-score.

    Args:
        merged: Output of :func:`load_provider_data`.

    Returns:
        Copy of ``merged`` with two appended columns:
        ``discrepancy`` (non-negative float) and ``z_score`` (float; all
        zeros when the discrepancy column has zero variance).
    """
    out = merged.copy()
    out["discrepancy"] = (out["yahoo_adj_factor"] - out["stooq_adj_factor"]).abs()

    if out["discrepancy"].nunique(dropna=True) <= 1:
        out["z_score"] = 0.0
    else:
        out["z_score"] = zscore(out["discrepancy"].to_numpy(), nan_policy="omit")
    return out


def flag_discrepancies(
    df: pd.DataFrame,
    z_threshold: float = Z_THRESHOLD,
    raw_threshold: float = RAW_THRESHOLD,
) -> pd.DataFrame:
    """Return only the rows that breach either the z-score or raw threshold.

    Args:
        df: Output of :func:`compute_discrepancy`.
        z_threshold: Absolute z-score above which a row is flagged.
        raw_threshold: Absolute discrepancy above which a row is flagged.

    Returns:
        Subset of ``df`` (same columns) containing only flagged rows.
    """
    z_flag = df["z_score"].abs() > z_threshold
    raw_flag = df["discrepancy"] > raw_threshold
    return df.loc[z_flag | raw_flag].copy()


def compare_ticker(
    ticker: str,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Run the full compare stage for a single ticker.

    Args:
        ticker: Equity ticker.
        raw_dir: Where the raw provider CSVs live.
        processed_dir: Where the merged comparison CSV is written.

    Returns:
        DataFrame of *flagged* rows only. The full comparison is persisted
        to ``processed_dir / {ticker}_comparison.csv`` as a side effect.
    """
    merged = load_provider_data(ticker, raw_dir=raw_dir)
    scored = compute_discrepancy(merged)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{ticker}_comparison.csv"
    scored.to_csv(out_path)
    logger.info("Wrote comparison: %s (%d rows)", out_path, len(scored))

    flagged = flag_discrepancies(scored)
    logger.info("%s flagged rows: %d / %d", ticker, len(flagged), len(scored))
    return flagged


def compare_all(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Run the compare stage for every ticker that has both raw files.

    Args:
        tickers: Override the default ticker basket.

    Returns:
        Mapping of ticker -> flagged-rows DataFrame for tickers that
        completed successfully. Tickers missing a raw CSV are logged and
        skipped.
    """
    tickers = tickers or TICKERS
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = compare_ticker(ticker)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s; missing input: %s", ticker, exc)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the compare module."""
    p = argparse.ArgumentParser(
        description="Merge providers, compute discrepancies, flag outliers."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to compare (default: basket).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: configure logging and run the full compare stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    compare_all(tickers=get_universe(args.universe))


if __name__ == "__main__":
    main()
