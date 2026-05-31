"""Unit tests for ``src.compare``.

These tests run on synthetic dataframes only - no live API calls -
so they are deterministic and CI-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compare import (  # noqa: E402
    Z_THRESHOLD,
    compute_discrepancy,
    flag_discrepancies,
    split_only_factor,
)


def _make_synthetic_frame(
    n: int = 250,
    inject_spike_at: int | None = 100,
    spike_size: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic provider-merged frame.

    A small Gaussian disagreement between providers is baked in, and one
    big outlier is injected at ``inject_spike_at`` so the flagging logic
    has something to catch.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    yahoo_factor = np.linspace(0.5, 1.0, n)
    noise = rng.normal(0.0, 1e-4, size=n)
    stooq_factor = yahoo_factor + noise

    if inject_spike_at is not None:
        stooq_factor[inject_spike_at] += spike_size

    df = pd.DataFrame(
        {
            "yahoo_close": np.full(n, 100.0),
            "yahoo_adj_close": np.full(n, 100.0) * yahoo_factor,
            "yahoo_adj_factor": yahoo_factor,
            "stooq_close": np.full(n, 100.0) * stooq_factor,
            "stooq_adj_close": np.full(n, 100.0) * stooq_factor,
            "stooq_adj_factor": stooq_factor,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Synthetic merged frame with a single injected discrepancy spike."""
    return _make_synthetic_frame()


@pytest.fixture
def scored_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic frame passed through :func:`compute_discrepancy`."""
    return compute_discrepancy(synthetic_df)


def test_discrepancy_is_non_negative(scored_df: pd.DataFrame) -> None:
    """`discrepancy` is an absolute value; it must never go below zero."""
    assert (scored_df["discrepancy"] >= 0).all(), (
        "Discrepancy column contains negative values"
    )


def test_zscore_has_mean_zero_std_one(scored_df: pd.DataFrame) -> None:
    """scipy.stats.zscore should yield approximately N(0, 1)."""
    z = scored_df["z_score"].to_numpy()
    assert abs(z.mean()) < 1e-6, f"z-score mean drifted: {z.mean()}"
    assert abs(z.std(ddof=0) - 1.0) < 1e-6, (
        f"z-score std not 1: {z.std(ddof=0)}"
    )


def test_flagged_rows_all_breach_threshold(scored_df: pd.DataFrame) -> None:
    """Every flagged row must breach the z-score OR the raw threshold."""
    flagged = flag_discrepancies(scored_df)
    breach = (flagged["z_score"].abs() > Z_THRESHOLD) | (
        flagged["discrepancy"] > 0.01
    )
    assert breach.all(), "A flagged row failed both threshold checks"


def test_flagged_rows_capture_injected_spike(scored_df: pd.DataFrame) -> None:
    """Sanity check: the injected spike should land in the flagged set."""
    flagged = flag_discrepancies(scored_df)
    assert not flagged.empty, "No rows were flagged despite injected spike"
    assert flagged["discrepancy"].max() > 0.01


def test_merge_has_no_duplicate_dates(scored_df: pd.DataFrame) -> None:
    """The comparison frame must have a unique date index."""
    assert scored_df.index.is_unique, "Comparison index contains duplicate dates"


def test_split_only_factor_handles_empty_splits() -> None:
    """With no splits, the factor reduces to close[t] / close[latest]."""
    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    close = pd.Series([100.0, 102.0, 104.0, 103.0, 110.0], index=dates)
    factor = split_only_factor(close, pd.Series(dtype=float))
    expected = close / close.iloc[-1]
    np.testing.assert_allclose(factor.to_numpy(), expected.to_numpy())
    assert factor.iloc[-1] == pytest.approx(1.0)


def test_split_only_factor_applies_split_strictly_before() -> None:
    """A 3-for-1 split divides all dates strictly BEFORE the split by 3."""
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    close = pd.Series([300.0, 300.0, 300.0, 100.0, 100.0, 100.0], index=dates)
    splits = pd.Series({dates[3]: 3.0}, name="official_ratio")
    factor = split_only_factor(close, splits)
    pre_split = factor.loc[:dates[2]].to_numpy()
    post_split = factor.loc[dates[3]:].to_numpy()
    np.testing.assert_allclose(pre_split, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(post_split, [1.0, 1.0, 1.0])


def test_constant_discrepancy_yields_zero_zscore() -> None:
    """Zero-variance input should not blow up the z-score calculation."""
    n = 50
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "yahoo_close": np.full(n, 100.0),
            "yahoo_adj_close": np.full(n, 90.0),
            "yahoo_adj_factor": np.full(n, 0.9),
            "stooq_close": np.full(n, 90.0),
            "stooq_adj_close": np.full(n, 90.0),
            "stooq_adj_factor": np.full(n, 0.9),
        },
        index=dates,
    )
    df.index.name = "date"
    scored = compute_discrepancy(df)
    assert (scored["z_score"] == 0.0).all()
    assert flag_discrepancies(scored).empty
