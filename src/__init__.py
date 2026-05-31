"""Financial Data Integrity Auditor.

A four-stage pipeline:
    1. ingest  - pull raw price data from two (or three) independent providers.
    2. compare - merge providers and flag adjustment-factor discrepancies.
    3. audit   - classify flags against the official corporate-action calendar.
    4. report  - aggregate per-ticker audits into a master summary table.

Two extra analytical modules sit alongside the pipeline:
    * backtest_impact   - translates discrepancies into PnL on a momentum signal.
    * stats_diagnostics - characterizes the discrepancy distribution and
      compares threshold-detection methods (z-score, MAD, IQR).
"""

from __future__ import annotations

from pathlib import Path

TICKERS: list[str] = ["TSLA", "AAPL", "AMZN", "GME", "NVDA"]

SP100_TICKERS: list[str] = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BIIB", "BK", "BKNG", "BLK", "BMY", "BRK-B",
    "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO",
    "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
    "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT",
    "XOM",
]

TICKER_SETS: dict[str, list[str]] = {
    "basket": TICKERS,
    "sp100":  SP100_TICKERS,
}


def get_universe(name: str) -> list[str]:
    """Resolve a universe alias to its ticker list.

    Args:
        name: One of ``"basket"`` (the 5-ticker default) or ``"sp100"``.

    Returns:
        The corresponding list of ticker symbols.

    Raises:
        ValueError: when ``name`` is not a known alias.
    """
    name = name.lower()
    if name not in TICKER_SETS:
        raise ValueError(
            f"Unknown universe {name!r}. Valid: {sorted(TICKER_SETS)}."
        )
    return TICKER_SETS[name]


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
