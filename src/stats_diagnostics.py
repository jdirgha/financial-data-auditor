"""Statistical diagnostics for the discrepancy series.

The compare-stage flag rule (|z| > 3) is a deliberately simple default.
This module quantifies the trade-offs hidden in that choice so the
threshold is defensible rather than arbitrary.

Three diagnostics are produced
------------------------------
1. **Distribution characterization.**  Empirical moments, percentiles, and
   tail mass of the per-day discrepancy series, plus a QQ-plot against
   the normal distribution. The discrepancy distribution is, in practice,
   heavily fat-tailed; the QQ plot makes that visible.

2. **Detector comparison.**  Flag counts under three different outlier
   detectors at the same nominal "3-sigma" level:

        z-score    : |x - mean| / std        > 3
        MAD        : |x - median| / (1.4826*MAD) > 3
        IQR        : x > Q3 + 1.5*IQR

   The MAD detector is robust to fat tails; the IQR detector is the
   Tukey rule used in classical exploratory data analysis. Comparing
   counts surfaces how much the z-score rule under- or over-reports.

3. **Threshold sensitivity sweep.**  Flag counts under z-thresholds in
   ``{2.0, 2.5, 3.0, 3.5, 4.0, 5.0}`` per ticker. Lets a reviewer pick a
   defensible operating point.

Outputs
-------
* ``data/processed/diagnostics_summary.csv`` -- one row per ticker with
  the headline distribution stats.
* ``data/processed/threshold_sensitivity.csv`` -- flag counts at each
  threshold level per ticker.
* ``docs/stats_diagnostics.png`` -- two-panel chart: pooled-discrepancy
  histogram with normal overlay (left), QQ plot (right).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import DOCS_DIR, PROCESSED_DIR, get_universe

logger = logging.getLogger(__name__)

DEFAULT_Z_THRESHOLDS: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
MAD_SCALE: float = 1.4826


def load_discrepancy_panel(
    tickers: list[str],
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    """Concatenate the ``discrepancy`` series from each ticker's comparison CSV.

    Args:
        tickers: Ticker universe.
        processed_dir: Directory holding ``{ticker}_comparison.csv`` files.

    Returns:
        Long DataFrame with columns ``[date, ticker, discrepancy]``.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        path = processed_dir / f"{ticker}_comparison.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "discrepancy" not in df.columns:
            continue
        sub = df[["discrepancy"]].copy()
        sub["ticker"] = ticker
        sub = sub.reset_index().rename(columns={df.index.name or "Unnamed: 0": "date"})
        if "date" not in sub.columns:
            sub = sub.rename(columns={sub.columns[0]: "date"})
        frames.append(sub[["date", "ticker", "discrepancy"]])
    if not frames:
        raise RuntimeError("No comparison CSVs found - run src.compare first.")
    return pd.concat(frames, ignore_index=True)


def describe_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker headline distribution statistics.

    Args:
        panel: Output of :func:`load_discrepancy_panel`.

    Returns:
        One row per ticker with mean, median, std, MAD, skew, excess
        kurtosis, and the 95 / 99 / 99.9 percentiles of the discrepancy
        series. A final pooled row labelled ``ALL`` is appended.
    """
    def _stats(x: pd.Series) -> pd.Series:
        x = x.dropna().to_numpy()
        if x.size == 0:
            return pd.Series(dtype=float)
        mad = np.median(np.abs(x - np.median(x)))
        return pd.Series({
            "n": x.size,
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "std": float(x.std(ddof=0)),
            "mad": float(mad),
            "skew": float(stats.skew(x, bias=False)) if x.size > 2 else np.nan,
            "excess_kurt": float(stats.kurtosis(x, fisher=True, bias=False))
                            if x.size > 3 else np.nan,
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "p999": float(np.quantile(x, 0.999)),
            "max": float(x.max()),
        })

    per_ticker = panel.groupby("ticker")["discrepancy"].apply(_stats).unstack()
    pooled = _stats(panel["discrepancy"]).to_frame("ALL").T
    return pd.concat([per_ticker, pooled]).round(8)


def detector_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare flag counts across z-score, MAD, and IQR detectors.

    Args:
        panel: Output of :func:`load_discrepancy_panel`.

    Returns:
        One row per ticker (plus pooled ``ALL``) with the number of flags
        each detector would raise at the canonical 3-sigma equivalent
        level. MAD and IQR are robust to fat tails; z-score is not.
    """
    def _counts(x: pd.Series) -> pd.Series:
        x = x.dropna().to_numpy()
        if x.size < 4:
            return pd.Series({"n": x.size, "z3": 0, "mad3": 0, "iqr_tukey": 0})
        mean = x.mean()
        std = x.std(ddof=0)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        z_flags  = int(np.sum(np.abs((x - mean) / std) > 3)) if std > 0 else 0
        mad_flags = int(np.sum(np.abs((x - med) / (MAD_SCALE * mad)) > 3)) if mad > 0 else 0
        iqr_flags = int(np.sum(x > q3 + 1.5 * iqr))
        return pd.Series({
            "n": x.size,
            "z3": z_flags,
            "mad3": mad_flags,
            "iqr_tukey": iqr_flags,
        })

    per_ticker = panel.groupby("ticker")["discrepancy"].apply(_counts).unstack()
    pooled = _counts(panel["discrepancy"]).to_frame("ALL").T
    out = pd.concat([per_ticker, pooled]).astype({"n": int, "z3": int, "mad3": int, "iqr_tukey": int})
    return out


def threshold_sensitivity(
    panel: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_Z_THRESHOLDS,
) -> pd.DataFrame:
    """Sweep z-score thresholds and report flag counts per level.

    Args:
        panel: Output of :func:`load_discrepancy_panel`.
        thresholds: Z-score levels to evaluate.

    Returns:
        Long DataFrame with columns ``[ticker, threshold, flags]``.
    """
    rows: list[dict[str, float | int | str]] = []
    for ticker, sub in panel.groupby("ticker"):
        x = sub["discrepancy"].dropna().to_numpy()
        if x.size < 4:
            continue
        mean, std = x.mean(), x.std(ddof=0)
        if std == 0:
            for t in thresholds:
                rows.append({"ticker": ticker, "threshold": t, "flags": 0})
            continue
        z = np.abs((x - mean) / std)
        for t in thresholds:
            rows.append({"ticker": ticker, "threshold": t,
                         "flags": int(np.sum(z > t))})
    return pd.DataFrame(rows)


def _save_charts(panel: pd.DataFrame, out_path: Path) -> None:
    """Render the histogram + QQ-plot diagnostic chart.

    Args:
        panel: Output of :func:`load_discrepancy_panel`.
        out_path: Output PNG path.
    """
    x = panel["discrepancy"].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    n_bins = 80
    counts, bins, _ = ax.hist(
        x, bins=n_bins, color="#1E40AF", alpha=0.75, edgecolor="white",
        linewidth=0.4, label="Empirical",
    )
    mu, sigma = x.mean(), x.std(ddof=0)
    bin_width = bins[1] - bins[0]
    xx = np.linspace(bins[0], bins[-1], 400)
    pdf = stats.norm.pdf(xx, loc=mu, scale=sigma) * x.size * bin_width
    ax.plot(xx, pdf, color="#DC2626", lw=1.6, label="Normal fit")
    ax.set_yscale("log")
    ax.set_xlabel("Per-day discrepancy")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(
        "Pooled discrepancy distribution (log y) vs normal fit",
        fontsize=12, fontweight="bold", color="#0B1E3A",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    stats.probplot(x, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#1E40AF")
    ax2.get_lines()[0].set_markeredgecolor("#1E40AF")
    ax2.get_lines()[0].set_markersize(3.0)
    ax2.get_lines()[1].set_color("#DC2626")
    ax2.get_lines()[1].set_linewidth(1.4)
    ax2.set_title(
        "QQ plot: discrepancy vs normal",
        fontsize=12, fontweight="bold", color="#0B1E3A",
    )
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles")
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote diagnostics chart -> %s", out_path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the diagnostics module."""
    p = argparse.ArgumentParser(
        description="Characterize the discrepancy distribution + compare detectors."
    )
    p.add_argument(
        "--universe",
        default="basket",
        choices=["basket", "sp100"],
        help="Ticker universe to diagnose (default: basket).",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: build all diagnostics and persist outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    tickers = get_universe(args.universe)
    panel = load_discrepancy_panel(tickers)
    logger.info("Loaded %d ticker-days of discrepancy data.", len(panel))

    summary = describe_distribution(panel)
    summary.to_csv(PROCESSED_DIR / "diagnostics_summary.csv")
    logger.info("Wrote distribution summary -> %s",
                PROCESSED_DIR / "diagnostics_summary.csv")

    detectors = detector_counts(panel)
    detectors.to_csv(PROCESSED_DIR / "detector_counts.csv")
    logger.info("Wrote detector comparison -> %s",
                PROCESSED_DIR / "detector_counts.csv")

    sens = threshold_sensitivity(panel)
    sens.to_csv(PROCESSED_DIR / "threshold_sensitivity.csv", index=False)
    logger.info("Wrote threshold sensitivity -> %s",
                PROCESSED_DIR / "threshold_sensitivity.csv")

    _save_charts(panel, DOCS_DIR / "stats_diagnostics.png")

    print("\n=== Discrepancy Distribution Summary (pooled + per ticker) ===\n")
    with pd.option_context("display.max_columns", None, "display.width", 140,
                           "display.float_format", "{:.3e}".format):
        print(summary.to_string())
    print("\n=== Detector counts at the 3-sigma-equivalent level ===\n")
    print(detectors.to_string())
    print("\nInterpretation: large gaps between z3 and mad3 indicate that the")
    print("z-score rule is being driven by fat tails. MAD is robust; prefer it")
    print("when excess_kurt is large (> ~5).\n")
    print("=== End diagnostics ===\n")


if __name__ == "__main__":
    main()
