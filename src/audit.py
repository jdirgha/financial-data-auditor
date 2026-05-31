"""Audit stage.

For every flagged comparison row, decide whether the discrepancy can be
*explained* by a known corporate action and, when it can, attempt to
adjudicate which provider's adjustment is closer to the official split
ratio.

Outputs ``data/processed/{ticker}_audit.csv`` with the columns:

    date, ticker, yahoo_adj_factor, stooq_adj_factor, discrepancy,
    z_score, flag_type, official_ratio, provider_verdict, severity_score
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import PROCESSED_DIR, SPLITS_DIR, TICKERS, get_universe
from .compare import flag_discrepancies

logger = logging.getLogger(__name__)

CORP_ACTION_WINDOW_DAYS: int = 5
PROVIDER_VERDICT_TOL: float = 1e-3


def load_comparison(ticker: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load the full per-day comparison frame written by the compare stage.

    Args:
        ticker: Equity ticker.
        processed_dir: Directory holding ``{ticker}_comparison.csv``.

    Returns:
        DataFrame indexed by date with both providers' columns plus
        ``discrepancy`` and ``z_score``.

    Raises:
        FileNotFoundError: when the comparison CSV does not exist yet.
    """
    path = processed_dir / f"{ticker}_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def load_official_splits(
    ticker: str, splits_dir: Path = SPLITS_DIR
) -> pd.DataFrame:
    """Load the official split calendar produced by the ingest stage.

    Args:
        ticker: Equity ticker.
        splits_dir: Directory holding ``{ticker}_splits.csv``.

    Returns:
        DataFrame indexed by split date with an ``official_ratio`` column.
        Returns an empty frame when no split file exists - some tickers
        have no splits over the observation window.
    """
    path = splits_dir / f"{ticker}_splits.csv"
    if not path.exists():
        logger.info("No official split file for %s; assuming none.", ticker)
        return pd.DataFrame(columns=["official_ratio"], index=pd.DatetimeIndex([], name="date"))
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def _nearest_split(
    date: pd.Timestamp,
    splits: pd.DataFrame,
    window_days: int,
) -> tuple[pd.Timestamp | None, float | None]:
    """Find the nearest official split within ``+/- window_days`` of ``date``.

    Args:
        date: Flagged trading day.
        splits: Output of :func:`load_official_splits`.
        window_days: Half-width of the matching window in calendar days.

    Returns:
        Pair ``(split_date, ratio)`` or ``(None, None)`` when no split is
        within the window.
    """
    if splits.empty:
        return None, None
    diffs = (splits.index - date).days
    mask = np.abs(diffs) <= window_days
    if not mask.any():
        return None, None
    candidates = splits.loc[mask]
    closest = (candidates.index - date).map(lambda d: abs(d.days)).argmin()
    split_date = candidates.index[closest]
    return split_date, float(candidates.iloc[closest]["official_ratio"])


def _implied_split_ratio(
    comparison: pd.DataFrame,
    split_date: pd.Timestamp,
    factor_col: str,
) -> float | None:
    """Estimate a provider's *implied* split ratio across ``split_date``.

    The implied ratio is the jump in cumulative adjustment factor across
    the split, i.e. ``factor[t-]/factor[t+]`` where ``t-`` and ``t+`` are
    the trading days immediately before and after the split.

    Args:
        comparison: Full per-day comparison frame for the ticker.
        split_date: Official split date.
        factor_col: Column name carrying the cumulative adjustment factor
            (``"yahoo_adj_factor"`` or ``"stooq_adj_factor"``).

    Returns:
        The implied ratio, or ``None`` when either side of the split is
        missing from the index (e.g. split fell outside the data window).
    """
    if factor_col not in comparison.columns:
        return None

    idx = comparison.index
    before = idx[idx < split_date]
    after = idx[idx >= split_date]
    if before.empty or after.empty:
        return None

    before_val = comparison.loc[before[-1], factor_col]
    after_val = comparison.loc[after[0], factor_col]
    if not np.isfinite(before_val) or not np.isfinite(after_val) or after_val == 0:
        return None
    return float(before_val / after_val)


def _provider_verdict(
    comparison: pd.DataFrame,
    split_date: pd.Timestamp,
    official_ratio: float,
    tol: float = PROVIDER_VERDICT_TOL,
) -> str:
    """Decide which provider's implied split ratio is closer to official.

    Args:
        comparison: Full comparison frame for the ticker.
        split_date: Official split date.
        official_ratio: Official split ratio from the corporate-actions feed.
        tol: If the absolute gap between the two providers' errors is below
            this, the verdict is ``"ambiguous"``.

    Returns:
        One of ``"yahoo_correct"``, ``"stooq_correct"``, or ``"ambiguous"``.
    """
    yahoo_ratio = _implied_split_ratio(comparison, split_date, "yahoo_adj_factor")
    stooq_ratio = _implied_split_ratio(comparison, split_date, "stooq_adj_factor")

    if yahoo_ratio is None and stooq_ratio is None:
        return "ambiguous"
    if yahoo_ratio is None:
        return "stooq_correct"
    if stooq_ratio is None:
        return "yahoo_correct"

    yahoo_err = abs(yahoo_ratio - official_ratio)
    stooq_err = abs(stooq_ratio - official_ratio)
    if abs(yahoo_err - stooq_err) < tol:
        return "ambiguous"
    return "yahoo_correct" if yahoo_err < stooq_err else "stooq_correct"


def audit_ticker(
    ticker: str,
    processed_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    window_days: int = CORP_ACTION_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build the full audit table for one ticker.

    Args:
        ticker: Equity ticker.
        processed_dir: Directory holding the ticker's comparison CSV; also
            where the audit CSV is written.
        splits_dir: Directory holding the ticker's official split calendar.
        window_days: Half-width of the corporate-action matching window
            in calendar days.

    Returns:
        DataFrame with one row per flagged trading day, with columns
        ``date, ticker, yahoo_adj_factor, stooq_adj_factor, discrepancy,
        z_score, flag_type, official_ratio, provider_verdict,
        severity_score``. The same frame is also persisted to
        ``processed_dir / {ticker}_audit.csv``.
    """
    comparison = load_comparison(ticker, processed_dir=processed_dir)
    flagged = flag_discrepancies(comparison)
    splits = load_official_splits(ticker, splits_dir=splits_dir)

    records: list[dict] = []
    for date, row in flagged.iterrows():
        split_date, official_ratio = _nearest_split(date, splits, window_days)
        if split_date is not None:
            flag_type = "corporate action window"
            verdict = _provider_verdict(comparison, split_date, official_ratio)
        else:
            flag_type = "unexplained discrepancy"
            verdict = "n/a"

        records.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ticker": ticker,
                "yahoo_adj_factor": float(row["yahoo_adj_factor"]),
                "stooq_adj_factor": float(row["stooq_adj_factor"]),
                "discrepancy": float(row["discrepancy"]),
                "z_score": float(row["z_score"]),
                "flag_type": flag_type,
                "official_ratio": official_ratio if official_ratio is not None else np.nan,
                "provider_verdict": verdict,
                "severity_score": round(abs(float(row["z_score"])), 2),
            }
        )

    audit = pd.DataFrame.from_records(
        records,
        columns=[
            "date",
            "ticker",
            "yahoo_adj_factor",
            "stooq_adj_factor",
            "discrepancy",
            "z_score",
            "flag_type",
            "official_ratio",
            "provider_verdict",
            "severity_score",
        ],
    )

    out_path = processed_dir / f"{ticker}_audit.csv"
    audit.to_csv(out_path, index=False)
    logger.info("Wrote audit: %s (%d flagged rows)", out_path, len(audit))
    return audit


def audit_all(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Run the audit stage for every ticker that has a comparison CSV.

    Args:
        tickers: Override the default ticker basket.

    Returns:
        Mapping of ticker -> per-ticker audit DataFrame.
    """
    tickers = tickers or TICKERS
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = audit_ticker(ticker)
        except FileNotFoundError as exc:
            logger.warning("Skipping audit for %s; missing input: %s", ticker, exc)
    return results


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the audit module."""
    p = argparse.ArgumentParser(
        description="Reconcile flags against the official corporate-action calendar."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to audit (default: basket).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: configure logging and run the full audit stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    audit_all(tickers=get_universe(args.universe))


if __name__ == "__main__":
    main()
