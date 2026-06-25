"""
Tests for the root-cause Fairfield/Marriott row-label P&L pipeline.

These build a synthetic raw workbook that mirrors the real structure: the FIRST
column holds row labels (e.g. 'Total Operating Revenue'), months are columns,
and room/total revenue is seasonal across 2022-2024.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pnl_extraction import extract_pnl_rows, normalize_label, MISSING_SOURCE_DATA  # noqa: E402
from services.pnl_forecasting import forecast_future_months, MODEL_SEASONAL_NAIVE  # noqa: E402
from services.pnl_pipeline import build_pnl_forecast  # noqa: E402


def _seasonal_series(years=3, base=100000.0, amp=0.35):
    """Monthly seasonal revenue: summer-peaking."""
    vals = []
    for y in range(years):
        for m in range(1, 13):
            season = 1 + amp * np.sin((m - 3) / 12 * 2 * np.pi)
            vals.append(round(base * season * (1 + 0.03 * y), 2))
    return vals


def _fairfield_raw_workbook():
    """
    Build a header-less wide DataFrame (dtype=str) like a Fairfield P&L export.
    Row 0 = month headers; col 0 = row labels.
    """
    months = [pd.Timestamp(2022, 1, 1) + pd.DateOffset(months=i) for i in range(36)]
    month_labels = [m.strftime("%b, %Y") for m in months]

    room = _seasonal_series(amp=0.35, base=120000.0)
    fb = [round(v, 2) for v in np.full(36, 4000.0)]          # near-constant (limited-service)
    other = [round(0.04 * r, 2) for r in room]
    total = [round(room[i] + fb[i] + other[i], 2) for i in range(36)]
    uoe = [round(0.55 * t, 2) for t in total]
    gop = [round(total[i] - uoe[i], 2) for i in range(36)]
    noi = [round(g * 0.5 - 8000, 2) for g in gop]            # distinct from GOP

    rows = [
        [""] + month_labels,
        ["Total Room Department Revenue"] + [str(v) for v in room],
        ["Total FB and Minor Operating Dept Revenue"] + [str(v) for v in fb],
        ["Other Operated Departments Revenue"] + [str(v) for v in other],
        ["Total Operating Revenue"] + [str(v) for v in total],
        ["Gross Operating Profit"] + [str(v) for v in gop],
        ["Net Income loss"] + [str(v) for v in noi],
    ]
    return pd.DataFrame(rows), {"room": room, "fb": fb, "total": total, "gop": gop, "noi": noi}


def test_normalize_label():
    assert normalize_label("  Total  Operating, Revenue ") == "total operating revenue"
    assert normalize_label('="Net Income (loss)"') == "net income loss"


def test_extracts_financials_from_row_labels():
    raw, truth = _fairfield_raw_workbook()
    monthly_df, lineage, debug = extract_pnl_rows(raw)
    for metric in ["Room_Revenue", "FB_Revenue", "Total_Revenue", "GOP", "NOI"]:
        assert metric in monthly_df.columns, f"{metric} not extracted"
    # Values match source (within rounding).
    assert np.allclose(monthly_df["Room_Revenue"].to_numpy(float), truth["room"], atol=1.0)
    assert np.allclose(monthly_df["NOI"].to_numpy(float), truth["noi"], atol=1.0)
    # NOI original label preserved in lineage.
    assert lineage["NOI"]["source_row_label"] == "Net Income loss"


def test_missing_kpis_are_null_not_fabricated():
    raw, _ = _fairfield_raw_workbook()
    _, lineage, _ = extract_pnl_rows(raw)
    for kpi in ["Occupancy_Pct", "ADR", "RevPAR", "Rooms_Sold", "Rooms_Available"]:
        assert lineage[kpi]["status"] == MISSING_SOURCE_DATA
        assert lineage[kpi]["source_row_label"] is None


def test_historical_actuals_never_replaced():
    raw, truth = _fairfield_raw_workbook()
    result = build_pnl_forecast(raw, horizon=12)
    actual_rows = [r for r in result["rows"]
                   if r["source_type"] == "actual" and r["metric"] == "Room_Revenue"]
    # 36 historical months of actuals, matching source.
    assert len(actual_rows) == 36
    vals = [r["value"] for r in actual_rows]
    assert np.allclose(vals, truth["room"], atol=1.0)


def test_seasonal_revenue_uses_seasonal_naive_and_is_not_flat():
    raw, _ = _fairfield_raw_workbook()
    monthly_df, _, _ = extract_pnl_rows(raw)
    future_df, report, flags = forecast_future_months(
        monthly_df, ["Room_Revenue", "Total_Revenue", "FB_Revenue"], horizon=12
    )
    assert report["Room_Revenue"]["model"] == MODEL_SEASONAL_NAIVE
    assert report["Total_Revenue"]["model"] == MODEL_SEASONAL_NAIVE
    # Non-flat.
    assert np.ptp(future_df["Room_Revenue"].to_numpy(float)) > 1.0
    assert np.ptp(future_df["Total_Revenue"].to_numpy(float)) > 1.0
    assert flags == []


def test_future_hierarchy_reconciles():
    raw, _ = _fairfield_raw_workbook()
    result = build_pnl_forecast(raw, horizon=12)
    resid = result["reconciliation"]["future_residuals"]
    assert resid["max"] <= 1e-6
    assert result["validations"]["future_hierarchy_reconciles"]["passed"]


def test_noi_not_forced_equal_to_gop():
    raw, truth = _fairfield_raw_workbook()
    result = build_pnl_forecast(raw, horizon=12)
    # Pick a forecast month and confirm NOI != GOP (they are distinct lines).
    fc = {(r["metric"], r["month"]): r["value"] for r in result["rows"]
          if r["source_type"] == "forecast"}
    months = sorted({m for (_, m) in fc})
    sample = months[0]
    assert fc.get(("NOI", sample)) != fc.get(("GOP", sample))
