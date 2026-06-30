"""
Per-line forecasting for the full Chart of Accounts (COA).

Where ``pnl_forecasting`` is narrow (handful of headline metrics, with
forced SeasonalNaive for the strongly-seasonal revenue lines), this module
forecasts EVERY COA leaf account that has enough history.

Rules (mirrors the property-aware defaults in ``pnl_forecasting``):
  * Forecast strictly future months -- never overwrite history.
  * Strongly seasonal series with >=12m of history -> SeasonalNaive(12).
  * Near-zero / sparse series -> zero-fill.
  * Otherwise -> simple-average baseline.
  * If a forecast collapses to a flat line on a clearly seasonal history,
    raise a ``FLAT_FORECAST_ON_SEASONAL_SERIES`` flag (the row is still
    emitted; downstream reporting decides how to surface it).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from services.pnl_forecasting import (
    is_near_zero_sparse,
    is_seasonal,
    _seasonal_naive_forecast,
    MODEL_SEASONAL_NAIVE,
    MODEL_ZERO_FILL,
    MODEL_SIMPLE_AVERAGE,
    MODEL_LAST_VALUE,
)

logger = logging.getLogger(__name__)


def _choose_model(history):
    y = np.asarray(history, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return MODEL_ZERO_FILL, "Empty history; zero-fill."
    if is_near_zero_sparse(y):
        return MODEL_ZERO_FILL, "Near-zero sparse account."
    if y.size >= 12 and is_seasonal(y):
        return MODEL_SEASONAL_NAIVE, "Detected 12-month seasonality."
    if y.size >= 2:
        return MODEL_SIMPLE_AVERAGE, "No strong seasonality; simple average baseline."
    return MODEL_LAST_VALUE, "Single observation; carry forward."


def _forecast_series(history, horizon):
    model, rationale = _choose_model(history)
    y = np.asarray(history, dtype=float)
    y = y[~np.isnan(y)]
    if model == MODEL_ZERO_FILL:
        preds = np.zeros(horizon, dtype=float)
    elif model == MODEL_SEASONAL_NAIVE:
        preds = _seasonal_naive_forecast(y, horizon, season=12)
    elif model == MODEL_LAST_VALUE:
        preds = np.full(horizon, y[-1] if y.size else 0.0, dtype=float)
    else:  # simple average
        base = float(np.nanmean(y)) if y.size else 0.0
        preds = np.full(horizon, base, dtype=float)
    return preds, model, rationale


def forecast_coa(wide_df: pd.DataFrame, horizon: int = 12):
    """
    Forecast every account column in ``wide_df``.

    Args:
        wide_df: DataFrame indexed by month-start ``Date`` with one column per
                 COA account (numeric values, NaN-friendly).
        horizon: number of future months to produce.

    Returns:
        (future_df, model_report, flags)
        future_df:    DataFrame with a ``Date`` column + one column per account
                      for the future months only.
        model_report: {account: {"model": str, "rationale": str}}.
        flags:        list of validation flags
                      (e.g. FLAT_FORECAST_ON_SEASONAL_SERIES).
    """
    if wide_df is None or wide_df.empty:
        raise ValueError("wide_df must be a non-empty COA wide frame.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    dates = pd.to_datetime(wide_df.index)
    last_actual = dates.max()
    future_dates = [last_actual + relativedelta(months=i)
                    for i in range(1, horizon + 1)]

    out = {"Date": future_dates}
    model_report = {}
    flags = []

    for account in wide_df.columns:
        history = wide_df[account].to_numpy(dtype=float)
        preds, model, rationale = _forecast_series(history, horizon)
        out[account] = preds
        model_report[account] = {"model": model, "rationale": rationale}

        hist_clean = history[~np.isnan(history)]
        if hist_clean.size >= 12 and is_seasonal(hist_clean):
            if np.ptp(preds) < 1e-6:
                flags.append({
                    "account": account,
                    "flag": "FLAT_FORECAST_ON_SEASONAL_SERIES",
                    "detail": "Forecast is flat but history is seasonal.",
                })

    future_df = pd.DataFrame(out)
    return future_df, model_report, flags


def build_coa_rows(wide_df: pd.DataFrame, future_df: pd.DataFrame,
                   model_report: dict, flags: list,
                   section_lookup: dict | None = None):
    """
    Emit one row per (account, month) in lineage form suitable for the API.
    """
    section_lookup = section_lookup or {}
    flag_by_account = {f["account"]: f for f in flags}
    rows = []

    for account in wide_df.columns:
        section = section_lookup.get(account)
        for ts, val in wide_df[account].items():
            rows.append({
                "month": pd.Timestamp(ts).strftime("%b %Y"),
                "account": account,
                "section": section,
                "value": None if pd.isna(val) else float(val),
                "source_type": "actual",
                "forecast_model": None,
                "confidence_flag": "high",
                "validation_notes": "",
            })

    for _, r in future_df.iterrows():
        month_label = pd.Timestamp(r["Date"]).strftime("%b %Y")
        for account in wide_df.columns:
            if account not in future_df.columns:
                continue
            info = model_report.get(account, {})
            note = info.get("rationale", "")
            conf = "medium"
            if account in flag_by_account:
                conf = "low"
                note = f"{note} [{flag_by_account[account]['flag']}]"
            val = r.get(account)
            rows.append({
                "month": month_label,
                "account": account,
                "section": section_lookup.get(account),
                "value": None if pd.isna(val) else float(val),
                "source_type": "forecast",
                "forecast_model": info.get("model"),
                "confidence_flag": conf,
                "validation_notes": note,
            })

    return rows
