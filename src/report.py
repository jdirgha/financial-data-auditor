"""Report stage.

Consolidates per-ticker audit CSVs into a single ``master_audit.csv`` and
prints a clean per-ticker summary table to the console.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import PROCESSED_DIR, TICKERS, get_universe

logger = logging.getLogger(__name__)


def load_all_audits(
    processed_dir: Path = PROCESSED_DIR,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every ``{ticker}_audit.csv`` in ``processed_dir``.

    Args:
        processed_dir: Directory holding per-ticker audit CSVs.
        tickers: Optionally restrict to a subset; defaults to all configured
            tickers whose audit file exists.

    Returns:
        Concatenated DataFrame across all tickers (empty if none found).
    """
    tickers = tickers or TICKERS
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = processed_dir / f"{ticker}_audit.csv"
        if not path.exists():
            logger.warning("Missing audit file for %s: %s", ticker, path)
            continue
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(master: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker headline statistics.

    Args:
        master: Output of :func:`load_all_audits`.

    Returns:
        Per-ticker summary DataFrame with columns:
        ``ticker, total_flags, corp_action_flags, unexplained_flags,
        avg_severity, worst_discrepancy``.
    """
    if master.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_flags",
                "corp_action_flags",
                "unexplained_flags",
                "avg_severity",
                "worst_discrepancy",
            ]
        )

    grp = master.groupby("ticker", sort=False)
    summary = grp.agg(
        total_flags=("date", "size"),
        avg_severity=("severity_score", "mean"),
        worst_discrepancy=("discrepancy", "max"),
    ).reset_index()

    corp = (
        master.assign(_is_corp=master["flag_type"].eq("corporate action window"))
        .groupby("ticker", sort=False)["_is_corp"]
        .sum()
        .rename("corp_action_flags")
    )
    unex = (
        master.assign(_is_unex=master["flag_type"].eq("unexplained discrepancy"))
        .groupby("ticker", sort=False)["_is_unex"]
        .sum()
        .rename("unexplained_flags")
    )

    summary = summary.merge(corp, on="ticker", how="left").merge(
        unex, on="ticker", how="left"
    )
    summary["avg_severity"] = summary["avg_severity"].round(2)
    summary["worst_discrepancy"] = summary["worst_discrepancy"].round(6)
    summary = summary[
        [
            "ticker",
            "total_flags",
            "corp_action_flags",
            "unexplained_flags",
            "avg_severity",
            "worst_discrepancy",
        ]
    ]
    return summary


def build_master(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Build, persist, and return the master audit table.

    Args:
        processed_dir: Directory where ``master_audit.csv`` is written.

    Returns:
        The master audit DataFrame (possibly empty).
    """
    master = load_all_audits(processed_dir=processed_dir)
    out_path = processed_dir / "master_audit.csv"
    master.to_csv(out_path, index=False)
    logger.info("Wrote master audit: %s (%d rows)", out_path, len(master))
    return master


def _print_summary(summary: pd.DataFrame) -> None:
    """Pretty-print the per-ticker summary table to stdout.

    Args:
        summary: Output of :func:`summarize`.
    """
    if summary.empty:
        print("\nNo audit data found. Run ingest -> compare -> audit first.\n")
        return
    print("\n=== Financial Data Integrity Audit: Per-Ticker Summary ===\n")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary.to_string(index=False))
    print("\n=== End summary ===\n")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the report module."""
    p = argparse.ArgumentParser(
        description="Aggregate per-ticker audits into a master table."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to report on (default: basket).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: build the master table and print the summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    tickers = get_universe(args.universe)
    master = load_all_audits(tickers=tickers)
    master.to_csv(PROCESSED_DIR / "master_audit.csv", index=False)
    logger.info(
        "Wrote master audit: %s (%d rows)",
        PROCESSED_DIR / "master_audit.csv", len(master),
    )
    summary = summarize(master)
    _print_summary(summary)


if __name__ == "__main__":
    main()
