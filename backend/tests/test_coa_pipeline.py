"""
Regression tests for the full Chart-of-Accounts (COA) pipeline.

Mirrors the spirit of ``test_pnl_pipeline.py`` but exercises the wider
``coa_extraction`` + ``coa_forecasting`` modules end-to-end on a synthetic
Fairfield-style workbook.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.coa_extraction import (
    extract_coa_rows,
    extract_coa_multi,
    coa_wide,
)
from services.coa_forecasting import (
    forecast_coa,
    build_coa_rows,
)


def _build_synthetic_workbook(year: int) -> pd.DataFrame:
    """
    Build a header-less wide frame that mimics the Fairfield row-label layout
    used in the rest of the test suite.
    """
    months = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    header = ["Account"] + [d.strftime("%b, %Y") for d in months]

    rows = [
        [""] * 13,                                   # spacer
        [""] * 13,                                   # spacer
        [""] * 13,                                   # spacer
        [""] * 13,                                   # spacer
        [""] * 13,                                   # spacer
        [""] * 13,                                   # spacer
        header,                                      # row 6: months
        ["Room Department Revenue"] + [""] * 12,     # section heading
        ["Total Room Department Revenue"] + [
            float(200_000 + 30_000 * np.sin(m / 12 * 2 * np.pi) + 1_000 * m)
            for m in range(12)
        ],
        ["Total FB and Minor Operating Dept Revenue"] + [
            float(15_000 + 2_000 * np.sin(m / 12 * 2 * np.pi))
            for m in range(12)
        ],
        ["Total Operating Revenue"] + [""] * 12,     # placeholder, recomputed below
        ["Total Departmental Expenses"] + [
            float(60_000 + 5_000 * np.sin(m / 12 * 2 * np.pi))
            for m in range(12)
        ],
        ["Gross Operating Profit"] + [""] * 12,
        ["Net Income loss"] + [
            float(40_000 + 8_000 * np.sin(m / 12 * 2 * np.pi))
            for m in range(12)
        ],
        ["Marketing"] + [float(7_000)] * 12,           # near-constant leaf
        ["Bad Debt Expense"] + [0.0] * 12,             # sparse / zero
    ]

    df = pd.DataFrame(rows)

    room_idx = 8
    fb_idx = 9
    tot_idx = 10
    gop_idx = 12
    for c in range(1, 13):
        room = float(df.iat[room_idx, c])
        fb = float(df.iat[fb_idx, c])
        df.iat[tot_idx, c] = room + fb
        df.iat[gop_idx, c] = (room + fb) - float(df.iat[11, c])
    return df.astype(object)


def test_extract_coa_rows_finds_sections_and_leaves():
    raw = _build_synthetic_workbook(2024)
    long_df, dbg = extract_coa_rows(raw)
    assert not long_df.empty
    assert dbg["leaf_count"] >= 6
    leaf_accounts = set(long_df.loc[~long_df["is_section"], "account"])
    assert "Total Room Department Revenue" in leaf_accounts
    assert "Net Income loss" in leaf_accounts
    assert "Marketing" in leaf_accounts


def test_extract_coa_multi_dedups_across_years():
    f1 = _build_synthetic_workbook(2023)
    f2 = _build_synthetic_workbook(2024)
    combined, dbg = extract_coa_multi([f1, f2])
    assert dbg["total_accounts"] > 0
    months = sorted(combined["Date"].dt.strftime("%Y-%m").unique())
    assert months[0].startswith("2023-")
    assert any(m.startswith("2024-") for m in months)
    assert combined.duplicated(subset=["account", "Date"]).sum() == 0


def test_coa_wide_pivots_excluding_sections():
    raw = _build_synthetic_workbook(2024)
    long_df, _ = extract_coa_rows(raw)
    wide = coa_wide(long_df)
    assert "Total Room Department Revenue" in wide.columns
    assert wide.shape[0] == 12
    assert wide["Total Room Department Revenue"].notna().all()


def test_forecast_coa_future_only_and_models():
    f1 = _build_synthetic_workbook(2023)
    f2 = _build_synthetic_workbook(2024)
    combined, _ = extract_coa_multi([f1, f2])
    wide = coa_wide(combined)
    future_df, model_report, flags = forecast_coa(wide, horizon=6)

    assert len(future_df) == 6
    last_actual = wide.index.max()
    assert (pd.to_datetime(future_df["Date"]) > last_actual).all()

    rev_model = model_report["Total Room Department Revenue"]["model"]
    assert rev_model == "SeasonalNaive(12)"

    bad_debt_model = model_report["Bad Debt Expense"]["model"]
    assert bad_debt_model == "ZeroFill"
    assert (future_df["Bad Debt Expense"].to_numpy() == 0).all()


def test_build_coa_rows_emits_lineage():
    raw = _build_synthetic_workbook(2024)
    long_df, _ = extract_coa_rows(raw)
    wide = coa_wide(long_df)
    future_df, model_report, flags = forecast_coa(wide, horizon=3)
    section_lookup = (long_df[~long_df["is_section"]]
                      .drop_duplicates("account")
                      .set_index("account")["section"].to_dict())
    rows = build_coa_rows(wide, future_df, model_report, flags, section_lookup)
    actual_rows = [r for r in rows if r["source_type"] == "actual"]
    forecast_rows = [r for r in rows if r["source_type"] == "forecast"]
    assert actual_rows and forecast_rows
    sample = forecast_rows[0]
    for key in ("month", "account", "value", "source_type",
                "forecast_model", "confidence_flag", "validation_notes"):
        assert key in sample
