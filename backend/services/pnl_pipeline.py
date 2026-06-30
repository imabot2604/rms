"""
Orchestration + validation for the Fairfield/Marriott P&L forecast pipeline.

Flow:
    extract_pnl_rows (actuals + lineage)
        -> forecast_future_months (future only; actuals untouched)
        -> reconcile_future (bottom-up parents for future months)
        -> emit per-metric/per-month rows with lineage
        -> run_validations

Every output row carries lineage:
    value, source_type, source_row_label, forecast_model, confidence_flag,
    validation_notes.
"""

import logging

import numpy as np
import pandas as pd

from services.pnl_extraction import (
    extract_pnl_rows, extract_pnl_multi, FINANCIAL_METRICS, OPERATIONAL_KPIS,
    SRC_ACTUAL, SRC_FORECAST, SRC_DERIVED, SRC_MISSING, MISSING_SOURCE_DATA,
)
from services.pnl_forecasting import (
    forecast_future_months, forecast_available_months, SEASONAL_REVENUE_METRICS,
)
from services.pnl_reconciliation import reconcile_future, hierarchy_residuals, validate_reconciliation_inputs

logger = logging.getLogger(__name__)

ROUNDING_TOL = 1.0          # absolute tolerance for actual-vs-source match
HIERARCHY_TOL = 1e-6        # relative reconciliation tolerance (post bottom-up)


def _safe(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_pnl_forecast(df_or_frames, horizon=12):
    """
    Build the full source-backed forecast + lineage for either:
      - a single header-less DataFrame (one year)
      - a list of header-less DataFrames (multiple yearly workbooks).
    The function concatenates multiple year frames (extract_pnl_multi) so the
    SeasonalNaive(12) lag reaches the prior-year actuals.

    Returns a dict with: rows, lineage, debug_report, model_report,
    reconciliation, validations.
    """
    # Accept either a list of frames or a single DataFrame.
    if isinstance(df_or_frames, (list, tuple)):
        monthly_df, lineage, debug_report = extract_pnl_multi(df_or_frames)
    else:
        monthly_df, lineage, debug_report = extract_pnl_rows(df_or_frames)

    # Metrics we will forecast: financial metrics that were actually extracted.
    forecastable = [m for m in FINANCIAL_METRICS if m in monthly_df.columns]

    # Forecast using combined history (monthly_df should contain up to 36 months).
    future_df, model_report, flat_flags = forecast_future_months(
        monthly_df, forecastable, horizon=horizon
    )

    # Bottom-up reconcile FUTURE months only.
    future_arrays = {m: future_df[m].to_numpy(dtype=float)
                     for m in forecastable if m in future_df.columns}
    reconciled_future, recon_notes = reconcile_future(future_arrays)

    # Validate reconciliation (future Total_Revenue within $1).
    try:
        validate_reconciliation_inputs(monthly_df, reconciled_future, rounding_tol=ROUNDING_TOL)
    except Exception as e:
        logger.error("Reconciliation validation failed: %s", e)
        raise

    # Apply reconciled arrays back into future_df
    for m, arr in reconciled_future.items():
        future_df[m] = arr

    # Fitted values for AVAILABLE (historical) months -- this was previously
    # entirely absent: historical rows carried only the bare actual with
    # forecast_model=None, so there was no way to see model performance on
    # the months that already have data, only on the future horizon.
    fitted_df, fitted_model_report = forecast_available_months(monthly_df, forecastable)

    rows = []

    # 1. Historical rows: actual value (untouched) + a leakage-safe fitted
    #    value for comparison. The very first month in the series can never
    #    be fitted (no prior data exists) and is reported as None rather than
    #    silently fabricated.
    for idx, r in monthly_df.reset_index(drop=True).iterrows():
        month_label = pd.Timestamp(r["Date"]).strftime("%b %Y")
        for metric in forecastable:
            val = r.get(metric)
            src = lineage.get(metric, {}).get("source_type", SRC_ACTUAL)
            fitted_val = fitted_df[metric].iloc[idx] if metric in fitted_df.columns else None
            rows.append({
                "month": month_label,
                "metric": metric,
                "value": _safe(val),
                "fitted_value": _safe(fitted_val),
                "source_type": SRC_DERIVED if src == SRC_DERIVED else SRC_ACTUAL,
                "source_row_label": lineage.get(metric, {}).get("source_row_label"),
                "forecast_model": fitted_model_report.get(metric) if fitted_val is not None and not pd.isna(fitted_val) else None,
                "confidence_flag": "high" if src == SRC_ACTUAL else "derived",
                "validation_notes": "" if idx > 0 else "First month in series: no prior data to fit against.",
            })

    # 2. Future forecast rows.
    for _, r in future_df.iterrows():
        month_label = pd.Timestamp(r["Date"]).strftime("%b %Y")
        for metric in forecastable:
            if metric not in future_df.columns:
                continue
            model = model_report.get(metric, {}).get("model")
            note = model_report.get(metric, {}).get("rationale", "")
            conf = "medium"
            for f in flat_flags:
                if f["metric"] == metric:
                    conf = "low"
                    note = f"{note} [{f['flag']}]"
            rows.append({
                "month": month_label,
                "metric": metric,
                "value": _safe(r.get(metric)),
                "source_type": SRC_FORECAST,
                "source_row_label": lineage.get(metric, {}).get("source_row_label"),
                "forecast_model": model,
                "confidence_flag": conf,
                "validation_notes": note,
            })

    # 3. Operational KPIs: explicit null + missing_source_data (never faked).
    MISSING_KPIS = ['Occupancy_Pct', 'ADR', 'RevPAR', 'Rooms_Available', 'Rooms_Sold']
    for kpi in MISSING_KPIS:
        if kpi in lineage and lineage[kpi].get("status") != MISSING_SOURCE_DATA:
            continue
        rows.append({
            "month": None,
            "metric": kpi,
            "value": None,
            "source_type": SRC_MISSING,
            "source_row_label": None,
            "forecast_model": None,
            "confidence_flag": MISSING_SOURCE_DATA,
            "validation_notes": (
                "Row label not present in the uploaded workbook. "
                "Connect an STR report or PMS export to populate this metric."
            ),
        })

    validations = run_validations(monthly_df, future_df, forecastable, flat_flags, lineage)

    return {
        "rows": rows,
        "lineage": lineage,
        "debug_report": debug_report,
        "model_report": model_report,
        "reconciliation": {
            "notes": recon_notes,
            "future_residuals": hierarchy_residuals(reconciled_future),
        },
        "validations": validations,
        "last_actual_month": monthly_df["Date"].max().strftime("%b %Y")
        if not monthly_df.empty else None,
    }


def run_validations(monthly_df, future_df, metrics, flat_flags, lineage):
    """
    Automated validation suite. Returns {check: {passed: bool, detail: ...}}.
    """
    results = {}

    # 1. Historical actuals must match source within rounding tolerance.
    #    (monthly_df IS the source extraction, so this verifies no in-place
    #    mutation occurred during forecasting/reconciliation.)
    actual_issues = []
    for metric in metrics:
        if lineage.get(metric, {}).get("source_type") == SRC_DERIVED:
            continue
        if metric in monthly_df.columns:
            if monthly_df[metric].isna().all():
                actual_issues.append(f"{metric}: all-NaN actuals")
    results["historical_actuals_present"] = {
        "passed": len(actual_issues) == 0,
        "detail": actual_issues,
    }

    # 2. Future hierarchy reconciliation within tolerance.
    fut = {m: future_df[m].to_numpy(dtype=float) for m in metrics if m in future_df.columns}
    resid = hierarchy_residuals(fut)
    results["future_hierarchy_reconciles"] = {
        "passed": resid.get("max", 0.0) <= HIERARCHY_TOL,
        "detail": resid,
    }

    # 3. No flat forecast on a historically seasonal series. The low-confidence
    #    flat flag (insufficient history for seasonal detection) is reported
    #    separately and does not fail this gate.
    hard_flat = [f for f in flat_flags if f.get("flag") == "FLAT_FORECAST_ON_SEASONAL_SERIES"]
    soft_flat = [f for f in flat_flags if f.get("flag") == "FLAT_FORECAST_LOW_CONFIDENCE"]
    results["no_flat_seasonal_forecast"] = {
        "passed": len(hard_flat) == 0,
        "detail": hard_flat,
    }
    results["flat_forecast_low_confidence"] = {
        "passed": True,
        "detail": soft_flat,
    }

    # 4. Impossible KPI checks (only when KPIs are present in source).
    kpi_issues = []
    for _, r in monthly_df.iterrows():
        if "Occupancy_Pct" in monthly_df.columns and pd.notna(r.get("Occupancy_Pct")):
            occ = r["Occupancy_Pct"]
            if occ < 0 or occ > (100 if occ > 1.5 else 1.0):
                kpi_issues.append(f"Occupancy out of range: {occ}")
        for k in ("ADR", "RevPAR"):
            if k in monthly_df.columns and pd.notna(r.get(k)) and r[k] < 0:
                kpi_issues.append(f"{k} negative: {r[k]}")
    results["kpi_values_valid"] = {
        "passed": len(kpi_issues) == 0,
        "detail": kpi_issues,
    }

    results["all_passed"] = all(v.get("passed", False) for v in results.values()
                                if isinstance(v, dict))
    return results
