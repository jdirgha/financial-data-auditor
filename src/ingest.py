"""Data ingestion stage.

Pulls daily adjusted-close history for the configured universe from up to
three independent providers (Yahoo Finance, Stooq, and optionally Tiingo)
and writes them, plus the official split calendar, to ``data/raw`` and
``data/splits``.

Adjustment-factor convention
----------------------------
Both providers expose an adjusted close; Stooq does not separately expose
an *unadjusted* close, so we define a normalized cumulative factor that
is computable from a single adjusted series:

    adj_factor[t] = adj_close[t] / adj_close[latest]

On the most recent bar this is 1.0; further back in time it decreases by
the cumulative ratio of all subsequent splits and dividends. Two correct
providers should agree on this curve to within rounding.

Designed to be re-run safely: each call overwrites the per-ticker CSV.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from . import RAW_DIR, SPLITS_DIR, get_universe

logger = logging.getLogger(__name__)

STOOQ_URL = "https://stooq.com/q/d/l/"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DEFAULT_START = "2018-01-01"
STOOQ_CALL_SPACING_SEC = 1.5
TIINGO_CALL_SPACING_SEC = 1.0
HTTP_USER_AGENT = "Mozilla/5.0 (compatible; financial-data-auditor/1.0)"


def fetch_yahoo(ticker: str, start: str = DEFAULT_START) -> pd.DataFrame:
    """Fetch unadjusted + adjusted daily closes from Yahoo Finance.

    Args:
        ticker: Equity ticker (e.g. ``"TSLA"``).
        start: ISO-format start date for the history window.

    Returns:
        DataFrame indexed by date with columns
        ``[yahoo_close, yahoo_adj_close, yahoo_adj_factor]``.
        ``yahoo_adj_factor`` is ``yahoo_adj_close[t] / yahoo_adj_close[latest]``.
    """
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"Yahoo returned an empty frame for {ticker!r}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close", "Adj Close"]].copy()
    df.columns = ["yahoo_close", "yahoo_adj_close"]
    df = df.sort_index()

    latest = float(df["yahoo_adj_close"].iloc[-1])
    df["yahoo_adj_factor"] = df["yahoo_adj_close"] / latest
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def fetch_stooq(
    ticker: str,
    api_key: str,
    start: str = DEFAULT_START,
) -> Optional[pd.DataFrame]:
    """Fetch daily (already-adjusted) closes from Stooq.

    Args:
        ticker: US equity ticker. ``.us`` is appended automatically.
        api_key: Stooq API key.
        start: ISO start date for the history window.

    Returns:
        DataFrame indexed by date with columns
        ``[stooq_close, stooq_adj_close, stooq_adj_factor]``, or ``None``
        when Stooq responds with an error payload.
    """
    stooq_symbol = f"{ticker.lower().replace('-', '.')}.us"
    params = {
        "s": stooq_symbol,
        "i": "d",
        "d1": start.replace("-", ""),
        "d2": pd.Timestamp.today().strftime("%Y%m%d"),
        "apikey": api_key,
    }

    try:
        resp = requests.get(
            STOOQ_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": HTTP_USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Stooq request failed for %s: %s", ticker, exc)
        return None

    body = resp.text.strip()
    if not body:
        logger.warning("Stooq returned an empty body for %s", ticker)
        return None

    first_line = body.splitlines()[0]
    if not first_line.lower().startswith("date,"):
        snippet = body[:200].replace("\n", " | ")
        logger.warning(
            "Stooq returned a non-CSV payload for %s "
            "(error or unmapped symbol). First 200 chars: %s",
            ticker,
            snippet,
        )
        return None

    try:
        df = pd.read_csv(io.StringIO(body))
    except Exception as exc:
        logger.warning("Stooq CSV parse failed for %s: %s", ticker, exc)
        return None

    if not {"Date", "Close"}.issubset(df.columns):
        logger.warning(
            "Stooq CSV missing expected columns for %s; got %s",
            ticker, list(df.columns),
        )
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.index.name = "date"

    out = pd.DataFrame(index=df.index)
    out["stooq_close"] = df["Close"].astype(float)
    out["stooq_adj_close"] = df["Close"].astype(float)
    latest = float(out["stooq_adj_close"].iloc[-1])
    out["stooq_adj_factor"] = out["stooq_adj_close"] / latest
    return out


def fetch_tiingo(
    ticker: str,
    api_key: str,
    start: str = DEFAULT_START,
) -> Optional[pd.DataFrame]:
    """Fetch daily adjusted closes from Tiingo (optional third provider).

    Tiingo's free tier (1000 calls/day, 50/hour) returns both unadjusted
    ``close`` and ``adjClose``. Sign up at <https://tiingo.com> and put
    your key in ``.env`` as ``TIINGO_API_KEY=...`` to enable.

    Args:
        ticker: US equity ticker.
        api_key: Tiingo API key.
        start: ISO start date.

    Returns:
        DataFrame indexed by date with columns
        ``[tiingo_close, tiingo_adj_close, tiingo_adj_factor]``, or ``None``
        on any HTTP / parse / authorization failure.
    """
    url = TIINGO_URL.format(ticker=ticker.lower())
    params = {
        "startDate": start,
        "endDate": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "format": "json",
        "token": api_key,
    }
    try:
        resp = requests.get(
            url, params=params, timeout=30,
            headers={"User-Agent": HTTP_USER_AGENT},
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Tiingo request failed for %s: %s", ticker, exc)
        return None

    if not isinstance(payload, list) or not payload:
        logger.warning("Tiingo returned an empty/invalid payload for %s", ticker)
        return None

    df = pd.DataFrame(payload)
    if not {"date", "close", "adjClose"}.issubset(df.columns):
        logger.warning(
            "Tiingo payload missing expected fields for %s; keys: %s",
            ticker, list(df.columns)[:10],
        )
        return None

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").set_index("date")
    df.index.name = "date"

    out = pd.DataFrame(index=df.index)
    out["tiingo_close"] = df["close"].astype(float)
    out["tiingo_adj_close"] = df["adjClose"].astype(float)
    latest = float(out["tiingo_adj_close"].iloc[-1])
    out["tiingo_adj_factor"] = out["tiingo_adj_close"] / latest
    return out


def fetch_official_splits(ticker: str) -> pd.DataFrame:
    """Fetch the official split calendar from Yahoo's corporate-actions feed.

    Args:
        ticker: Equity ticker.

    Returns:
        DataFrame indexed by split date with a single ``official_ratio`` column.
    """
    splits = yf.Ticker(ticker).splits
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    df = pd.DataFrame({"official_ratio": splits})
    df.index.name = "date"
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path``, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Wrote %s rows -> %s", len(df), path.name)


def ingest_all(
    tickers: list[str],
    stooq_key: str,
    tiingo_key: str | None = None,
    start: str = DEFAULT_START,
) -> None:
    """Run the full ingestion stage for every ticker.

    Args:
        tickers: List of equity tickers to ingest.
        stooq_key: Stooq API key (required).
        tiingo_key: Optional Tiingo API key; if provided, also pulls a
            third independent provider per ticker.
        start: ISO start date.
    """
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Ingesting %s", i + 1, n, ticker)

        try:
            yahoo = fetch_yahoo(ticker, start=start)
            _save(yahoo, RAW_DIR / f"{ticker}_yahoo.csv")
        except Exception as exc:
            logger.warning("Yahoo ingest failed for %s: %s", ticker, exc)

        stooq = fetch_stooq(ticker, api_key=stooq_key, start=start)
        if stooq is not None:
            _save(stooq, RAW_DIR / f"{ticker}_stooq.csv")

        if tiingo_key:
            tiingo = fetch_tiingo(ticker, api_key=tiingo_key, start=start)
            if tiingo is not None:
                _save(tiingo, RAW_DIR / f"{ticker}_tiingo.csv")
            time.sleep(TIINGO_CALL_SPACING_SEC)

        try:
            splits = fetch_official_splits(ticker)
            _save(splits, SPLITS_DIR / f"{ticker}_splits.csv")
        except Exception as exc:
            logger.warning("Split calendar fetch failed for %s: %s", ticker, exc)

        if i < n - 1:
            time.sleep(STOOQ_CALL_SPACING_SEC)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ingest module."""
    p = argparse.ArgumentParser(
        description="Pull raw price data from Yahoo + Stooq (+ optional Tiingo)."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to ingest (default: basket).",
    )
    p.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"ISO start date for the price history (default: {DEFAULT_START}).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point.

    Loads ``.env``, configures logging, parses args, and runs ingestion.
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    stooq_key = os.getenv("STOOQ_API_KEY")
    if not stooq_key:
        raise RuntimeError(
            "STOOQ_API_KEY is not set. Solve the captcha at "
            "https://stooq.com/q/d/?s=aapl.us and add STOOQ_API_KEY=<key> "
            "to your .env file."
        )

    tiingo_key = os.getenv("TIINGO_API_KEY")
    if tiingo_key:
        logger.info("Tiingo key detected - third provider will be ingested.")
    else:
        logger.info(
            "No TIINGO_API_KEY set - skipping third provider. "
            "Set one in .env to enable majority-rules adjudication later."
        )

    tickers = get_universe(args.universe)
    logger.info(
        "Ingesting universe=%s (%d tickers) from %s",
        args.universe, len(tickers), args.start,
    )
    ingest_all(tickers=tickers, stooq_key=stooq_key,
               tiingo_key=tiingo_key, start=args.start)


if __name__ == "__main__":
    main()
