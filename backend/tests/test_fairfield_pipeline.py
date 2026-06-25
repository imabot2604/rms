"""
Lightweight regression / validation tests for the Fairfield-style Excel-bound
forecasting pipeline. These are intentionally dependency-light: they exercise
parsing, naming, reconciliation alignment, missing-month behavior, near-constant
fallback, and deterministic date parsing.

Run with: pytest backend/tests -q   (from the backend/ directory or with
backend/ on PYTHONPATH).
"""

import io
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.data_processing import (  # noqa: E402
    _map_columns_to_schema,
    _parse_month_series,
    derive_accounting_identities,
)
from services.models import (  # noqa: E402
    select_best_model,
    is_near_constant,
    forecast_excel_months,
    SIMPLE_MODELS,
)
from services.excel_timeline import coverage_frame, DQ_MISSING_MONTH  # noqa: E402
from services.reconciliation import (  # noqa: E402
    reconcile_topdown,
    reconciliation_error,
    ReconciliationError,
)


def _fairfield_frame(months=24, start="2022-01-01", drop_month=None):
    """Build a synthetic Fairfield-style canonical frame."""
    dates = pd.date_range(start=start, periods=months, freq="MS")
    rng = np.random.default_rng(7)
    avail = np.full(months, 124 * 30.0)
    occ = np.clip(0.7 + 0.1 * np.sin(np.arange(months) / 12 * 2 * np.pi) + rng.normal(0, 0.02, months), 0.3, 0.98)
    sold = (occ * avail).round()
    adr = 120 + rng.normal(0, 5, months)
    room_rev = sold * adr
    fb_rev = np.full(months, 5000.0) + rng.normal(0, 50, months)  # near-constant
    other_rev = room_rev * 0.05
    total_rev = room_rev + fb_rev + other_rev
    gop = total_rev * 0.4
    noi = gop * 0.6
    # Inject December NOI anomalies (year-end charges).
    for i, d in enumerate(dates):
        if d.month == 12:
            noi[i] -= total_rev[i] * 0.5
    df = pd.DataFrame({
        "Date": dates, "Rooms_Available": avail, "Rooms_Sold": sold,
        "Occupancy_Pct": occ, "ADR": adr, "RevPAR": occ * adr,
        "Room_Revenue": room_rev, "FB_Revenue": fb_rev, "Other_Revenue": other_rev,
        "Total_Revenue": total_rev, "Total_UOE": total_rev * 0.2,
        "GOP": gop, "NOI": noi,
    })
    if drop_month is not None:
        df = df[df["Date"] != pd.Timestamp(drop_month)].reset_index(drop=True)
    return df


def test_usali_statblock_labels_map_to_schema():
    # USALI stat-block KPIs appear as ROW LABELS -> after pivot become columns.
    labels = ["Occupancy %", "ADR", "RevPAR", "Rooms Sold", "Rooms Available"]
    mapping = _map_columns_to_schema(labels)
    targets = set(mapping.values())
    assert {"Occupancy_Pct", "ADR", "RevPAR", "Rooms_Sold", "Rooms_Available"} <= targets


def test_noi_naming_variants_normalized():
    for label in ["Net Income (loss)", "Net Income loss", "Net Operating Income", "NOI"]:
        mapping = _map_columns_to_schema([label])
        assert "NOI" in mapping.values(), f"{label} did not map to NOI"


def test_deterministic_month_parsing():
    s = pd.Series(["Jan, 2024", "Feb, 2024", "Dec, 2024"])
    parsed = _parse_month_series(s)
    assert list(parsed.dt.month) == [1, 2, 12]
    assert list(parsed.dt.year) == [2024, 2024, 2024]
    assert parsed.isna().sum() == 0


def test_total_uoe_derivation_when_missing():
    df = _fairfield_frame(12).drop(columns=["Total_UOE"])
    out = derive_accounting_identities(df)
    assert "Total_UOE" in out.columns
    assert out["Total_UOE"].notna().any()


def test_near_constant_detection_and_model_fallback():
    flat = np.full(24, 5000.0) + np.random.default_rng(1).normal(0, 1, 24)
    assert is_near_constant(flat)
    model, _, _ = select_best_model(flat, node="FB_Revenue")
    assert model in SIMPLE_MODELS


def test_missing_month_zero_filled_and_flagged():
    df = _fairfield_frame(12, drop_month="2022-05-01")
    coverage, _, sequence = coverage_frame(df, value_col="Occupancy_Pct")
    missing_rows = coverage[coverage["is_missing_filled"]]
    assert len(missing_rows) == 1
    r = missing_rows.iloc[0]
    assert r["actual"] == 0.0
    assert r["dq_flag"] == DQ_MISSING_MONTH
    table = forecast_excel_months(coverage, coverage.loc[~coverage["is_missing_filled"], "actual"].to_numpy(), value_col="Occupancy_Pct")
    mt = table[table["is_missing_filled"]].iloc[0]
    assert mt["forecast"] == 0.0 and mt["lower"] == 0.0 and mt["upper"] == 0.0


def test_revenue_reconciliation_error_near_zero():
    df = _fairfield_frame(24)
    arrays = {}
    index = None
    for node in ["Room_Revenue", "FB_Revenue", "Other_Revenue", "Total_Revenue", "Total_UOE", "GOP", "NOI"]:
        coverage, _, _ = coverage_frame(df, value_col=node)
        index = list(coverage["label"]) if index is None else index
        table = forecast_excel_months(coverage, coverage.loc[~coverage["is_missing_filled"], "actual"].to_numpy(), value_col=node)
        arrays[node] = table["forecast"].to_numpy()
    reconciled = reconcile_topdown(arrays, month_index=index)
    errors = reconciliation_error(reconciled)
    assert errors["max"] < 1e-6


def test_reconciliation_alignment_guardrail():
    arrays = {"Total_Revenue": np.ones(12), "Room_Revenue": np.ones(11)}
    with pytest.raises(ReconciliationError):
        reconcile_topdown(arrays)


def test_seasonal_naive_fallback_same_month_mean():
    from services.models import _fit_predict_in_sample
    y = np.array([
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
        20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0
    ])
    train_idx = np.arange(24)
    preds = _fit_predict_in_sample(y, train_idx, np.array([0]), "SeasonalNaive")
    assert preds[0] == 15.0


def test_noi_december_outlier_handling():
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    actuals = []
    for d in dates:
        if d.month == 12:
            actuals.append(-2000.0)
        else:
            actuals.append(-100.0)
            
    coverage = pd.DataFrame({
        "month": dates,
        "label": [d.strftime("%b %Y") for d in dates],
        "actual": actuals,
        "is_missing_filled": False,
        "dq_flag": "OK",
        "dq_reason": ""
    })
    
    table = forecast_excel_months(coverage, np.array(actuals), value_col="NOI")
    
    assert "outlier_adjustment" in table.columns
    assert table["outlier_adjustment"].iloc[0] == -2000.0
    
    dec_rows = table[table["month"].dt.month == 12]
    for val in dec_rows["forecast"]:
        assert val < -1500.0
        
    non_dec_rows = table[table["month"].dt.month != 12]
    for val in non_dec_rows["forecast"]:
        assert -150.0 < val < -50.0

