"""Downstream impact analysis.

Quantifies the *real-world cost* of provider disagreement by running an
identical strategy on two different adjusted-close series and measuring
the gap in headline performance metrics.

Strategy
--------
A simple monthly-rebalanced long-only momentum signal:

    signal[t]  = 12-month return MINUS 1-month return, as of t-1 close
    weight[i]  = +1 if signal[t] is in the top tercile across tickers, else 0
    weights are equal-weighted within the long leg, capital fully invested

This is the canonical "12-1 momentum" signal used in the
Jegadeesh-Titman (1993) and Carhart (1997) factor literature. It is
deliberately not optimized -- the point is to isolate the effect of
*input data quality*, not strategy design.

Outputs
-------
* ``data/processed/backtest_impact.csv`` -- per-provider performance
  metrics (annualized return, Sharpe, max drawdown).
* ``docs/backtest_impact.png`` -- cumulative-return chart with both
  providers overlaid, and a second panel showing the PnL gap on a
  hypothetical $1M position.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import DOCS_DIR, PROCESSED_DIR, RAW_DIR, get_universe

logger = logging.getLogger(__name__)

POSITION_NOTIONAL: float = 1_000_000.0
LOOKBACK_LONG_MONTHS: int = 12
LOOKBACK_SKIP_MONTHS: int = 1
TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class BacktestResult:
    """Headline performance metrics for one provider's backtest."""

    provider: str
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    total_return: float
    n_trading_days: int
    cumulative_pnl_usd: float
    equity_curve: pd.Series


def _load_provider_panel(
    tickers: list[str],
    column: str,
    raw_dir: Path = RAW_DIR,
    file_suffix: str = "yahoo",
) -> pd.DataFrame:
    """Stack one provider's adjusted-close column into a (date x ticker) panel.

    Args:
        tickers: Ticker universe.
        column: Name of the adjusted-close column to extract
            (e.g. ``"yahoo_adj_close"``).
        raw_dir: Directory containing raw provider CSVs.
        file_suffix: Filename suffix per ticker (``"yahoo"`` or ``"stooq"``).

    Returns:
        Wide DataFrame indexed by date with one column per ticker.
        Tickers whose raw file is missing are silently skipped.
    """
    frames: list[pd.Series] = []
    for ticker in tickers:
        path = raw_dir / f"{ticker}_{file_suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if column not in df.columns:
            continue
        s = df[column].rename(ticker)
        frames.append(s)
    if not frames:
        raise RuntimeError(
            f"No usable CSVs for column={column!r} under {raw_dir}."
        )
    panel = pd.concat(frames, axis=1).sort_index()
    panel.index.name = "date"
    return panel


def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown of an equity curve.

    Args:
        equity: Cumulative-return series starting near 1.0.

    Returns:
        Negative float (e.g. ``-0.23`` for a 23% drawdown).
    """
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min())


def _momentum_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-minus-1-month momentum signal across the panel.

    Args:
        prices: Wide adjusted-close panel (date x ticker).

    Returns:
        DataFrame of cross-sectional momentum signals aligned to ``prices``.
    """
    long_window = LOOKBACK_LONG_MONTHS * 21
    skip_window = LOOKBACK_SKIP_MONTHS * 21
    long_ret = prices.pct_change(long_window, fill_method=None).shift(skip_window)
    short_ret = prices.pct_change(skip_window, fill_method=None).shift(skip_window)
    return long_ret - short_ret


def run_backtest(prices: pd.DataFrame, provider: str) -> BacktestResult:
    """Run the 12-1 momentum backtest on a single provider's price panel.

    Args:
        prices: Wide adjusted-close panel (date x ticker).
        provider: Display label (``"Yahoo"`` or ``"Stooq"``).

    Returns:
        BacktestResult with annualized metrics and full equity curve.
    """
    prices = prices.sort_index()
    daily_returns = prices.pct_change(fill_method=None)

    signal = _momentum_signal(prices)
    signal = signal.reindex(prices.index).ffill(limit=21)

    rebalance = signal.resample("ME").last().dropna(how="all")
    weights_monthly = pd.DataFrame(0.0, index=rebalance.index, columns=prices.columns)
    for dt, row in rebalance.iterrows():
        ranked = row.dropna()
        if ranked.empty:
            continue
        threshold = ranked.quantile(2 / 3)
        longs = ranked[ranked >= threshold].index
        if len(longs) == 0:
            continue
        weights_monthly.loc[dt, longs] = 1.0 / len(longs)

    weights_daily = weights_monthly.reindex(prices.index, method="ffill").fillna(0.0)
    weights_daily = weights_daily.shift(1).fillna(0.0)

    strategy_returns = (weights_daily * daily_returns).sum(axis=1)
    strategy_returns = strategy_returns.fillna(0.0)

    equity = (1.0 + strategy_returns).cumprod()
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0:
        n_days = 1

    total_return = float(equity.iloc[-1] - 1.0)
    annualized_return = float(equity.iloc[-1] ** (365.25 / n_days) - 1.0)
    annualized_vol = float(strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0
    cumulative_pnl_usd = total_return * POSITION_NOTIONAL

    return BacktestResult(
        provider=provider,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(equity),
        total_return=total_return,
        n_trading_days=int(strategy_returns.shape[0]),
        cumulative_pnl_usd=cumulative_pnl_usd,
        equity_curve=equity,
    )


def compare_providers(
    tickers: list[str],
    raw_dir: Path = RAW_DIR,
) -> tuple[BacktestResult, BacktestResult]:
    """Run the same backtest on both providers' adjusted closes.

    Args:
        tickers: Ticker universe.
        raw_dir: Directory holding raw provider CSVs.

    Returns:
        Tuple ``(yahoo_result, stooq_result)``.
    """
    yahoo_prices = _load_provider_panel(
        tickers, column="yahoo_adj_close", raw_dir=raw_dir, file_suffix="yahoo"
    )
    stooq_prices = _load_provider_panel(
        tickers, column="stooq_adj_close", raw_dir=raw_dir, file_suffix="stooq"
    )

    common_cols = sorted(set(yahoo_prices.columns) & set(stooq_prices.columns))
    common_index = yahoo_prices.index.intersection(stooq_prices.index)
    yahoo_prices = yahoo_prices.loc[common_index, common_cols]
    stooq_prices = stooq_prices.loc[common_index, common_cols]

    logger.info(
        "Backtesting %d tickers x %d days across both providers.",
        len(common_cols), len(common_index),
    )

    yahoo_res = run_backtest(yahoo_prices, provider="Yahoo")
    stooq_res = run_backtest(stooq_prices, provider="Stooq")
    return yahoo_res, stooq_res


def _save_metrics_csv(
    results: list[BacktestResult],
    out_path: Path,
) -> pd.DataFrame:
    """Write the per-provider metrics to CSV.

    Args:
        results: List of BacktestResult objects.
        out_path: CSV path.

    Returns:
        The same metrics as a DataFrame (for downstream printing).
    """
    rows = []
    for r in results:
        rows.append({
            "provider": r.provider,
            "n_trading_days": r.n_trading_days,
            "total_return_pct": round(r.total_return * 100, 3),
            "annualized_return_pct": round(r.annualized_return * 100, 3),
            "annualized_vol_pct": round(r.annualized_vol * 100, 3),
            "sharpe": round(r.sharpe, 3),
            "max_drawdown_pct": round(r.max_drawdown * 100, 3),
            "cumulative_pnl_usd": round(r.cumulative_pnl_usd, 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Wrote backtest metrics -> %s", out_path)
    return df


def _save_chart(
    yahoo_res: BacktestResult,
    stooq_res: BacktestResult,
    out_path: Path,
) -> None:
    """Plot overlaid equity curves and the PnL-gap chart.

    Args:
        yahoo_res: BacktestResult for Yahoo.
        stooq_res: BacktestResult for Stooq.
        out_path: Output PNG path.
    """
    yahoo_eq = yahoo_res.equity_curve
    stooq_eq = stooq_res.equity_curve
    common = yahoo_eq.index.intersection(stooq_eq.index)
    yahoo_eq = yahoo_eq.loc[common]
    stooq_eq = stooq_eq.loc[common]
    pnl_gap_usd = (yahoo_eq - stooq_eq) * POSITION_NOTIONAL

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
    )
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(yahoo_eq.index, yahoo_eq.values, label="Yahoo", color="#1E40AF", lw=1.6)
    ax.plot(stooq_eq.index, stooq_eq.values, label="Stooq", color="#059669", lw=1.6)
    ax.set_ylabel("Equity (starts at 1.0)")
    ax.set_title(
        "12-1 momentum backtest: same strategy, two providers' adjusted closes",
        fontsize=13, fontweight="bold", color="#0B1E3A",
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.fill_between(
        pnl_gap_usd.index, 0, pnl_gap_usd.values,
        where=(pnl_gap_usd.values >= 0), color="#1E40AF", alpha=0.5,
        label="Yahoo ahead",
    )
    ax2.fill_between(
        pnl_gap_usd.index, 0, pnl_gap_usd.values,
        where=(pnl_gap_usd.values < 0), color="#DC2626", alpha=0.5,
        label="Stooq ahead",
    )
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("PnL gap, $1M position")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax2.legend(loc="upper left", frameon=False)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote backtest chart -> %s", out_path)


def _print_metrics(df: pd.DataFrame, yahoo_res: BacktestResult, stooq_res: BacktestResult) -> None:
    """Print the metrics table + a one-line interpretation."""
    print("\n=== Downstream Impact: 12-1 Momentum Backtest ===\n")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(df.to_string(index=False))

    sharpe_gap = yahoo_res.sharpe - stooq_res.sharpe
    pnl_gap = yahoo_res.cumulative_pnl_usd - stooq_res.cumulative_pnl_usd
    print(
        f"\nSharpe gap (Yahoo - Stooq): {sharpe_gap:+.3f}  |  "
        f"Cumulative PnL gap on $1M: ${pnl_gap:+,.0f}\n"
        f"=== End ===\n"
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the backtest-impact module."""
    p = argparse.ArgumentParser(
        description="Backtest 12-1 momentum on both providers' adjusted closes."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to backtest (default: basket).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: backtest both providers, persist metrics + chart."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    tickers = get_universe(args.universe)

    yahoo_res, stooq_res = compare_providers(tickers=tickers)

    metrics_df = _save_metrics_csv(
        [yahoo_res, stooq_res],
        out_path=PROCESSED_DIR / "backtest_impact.csv",
    )
    _save_chart(
        yahoo_res, stooq_res,
        out_path=DOCS_DIR / "backtest_impact.png",
    )
    _print_metrics(metrics_df, yahoo_res, stooq_res)


if __name__ == "__main__":
    main()
