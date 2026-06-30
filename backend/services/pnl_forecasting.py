"""
Layer 3: property-aware forecasting for Fairfield/Marriott P&L metrics.

Key rules (root-cause fixes):
  * Forecast ONLY months strictly after the last actual month. Historical
    actuals are returned untouched and are never overwritten by model output.
  * Strongly seasonal revenue series (Room_Revenue, Total_Revenue) default to
    SeasonalNaive with 12-month seasonality so they never collapse to a flat
    line. ARIMA and other models that flatten visibly seasonal series are
    explicitly NOT used for these metrics.
  * True near-zero sparse series (e.g. limited-service F&B) use zero-fill.
  * A per-metric model-selection report records the chosen model + rationale.
"""

import logging

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# Metrics that are strongly seasonal for this property and must use a seasonal
# model by default.
SEASONAL_REVENUE_METRICS = {"Room_Revenue", "Total_Revenue"}

# FORCE set: these metrics must use SeasonalNaive(12) regardless of other logic
FORCE_SEASONAL_NAIVE = {"Room_Revenue", "Total_Revenue", "GOP", "FB_Revenue"}

MODEL_SEASONAL_NAIVE = "SeasonalNaive(12)"
MODEL_ZERO_FILL = "ZeroFill"
MODEL_SIMPLE_AVERAGE = "SimpleAverage"
MODEL_LAST_VALUE = "LastValue"


def is_near_zero_sparse(series, zero_frac_threshold=0.6):
    """True if most observations are ~0 (true sparse series)."""
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return True
    return float(np.mean(np.abs(y) < 1e-9)) >= zero_frac_threshold


def is_seasonal(series, min_cycles=2, strength_threshold=0.10):
    """
    Detect meaningful 12-month seasonality.

    Requires at least ``min_cycles`` years of data and a month-of-year signal
    whose spread is non-trivial relative to the series level.
    """
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    if y.size < 12 * min_cycles:
        return False
    level = np.mean(np.abs(y))
    if level < 1e-9:
        return False
    # Month-of-year means relative to overall level.
    n_full = (y.size // 12) * 12
    if n_full < 12:
        return False
    reshaped = y[-n_full:].reshape(-1, 12)
    monthly_means = reshaped.mean(axis=0)
    spread = (monthly_means.max() - monthly_means.min()) / level
    return spread >= strength_threshold


def _seasonal_naive_forecast(history, horizon, season=12):
    """Repeat the value from `season` months prior (last full cycle)."""
    y = np.asarray(history, dtype=float)
    preds = []
    extended = list(y)
    for h in range(horizon):
        ref = len(extended) - season
        preds.append(extended[ref] if ref >= 0 else extended[-1])
        extended.append(preds[-1])
    return np.asarray(preds, dtype=float)


def select_model(metric, history):
    """
    Choose a forecasting model for a metric, returning (model_name, rationale).

    Property-aware defaults take precedence over generic selection.
    """
    y = np.asarray(history, dtype=float)
    y = y[~np.isnan(y)]

    # Forced seasonal naive metrics
    if metric in FORCE_SEASONAL_NAIVE:
        if y.size >= 12:
            return MODEL_SEASONAL_NAIVE, (
                f"{metric} is forced to SeasonalNaive(12) for this property class "
                "to preserve seasonality and avoid flat ARIMA-like behaviour."
            )
        return MODEL_LAST_VALUE, "Insufficient history (<12m) for forced seasonality."

    if is_near_zero_sparse(y):
        return MODEL_ZERO_FILL, "Near-zero sparse series; zero-fill is appropriate."

    # Strongly seasonal revenue series -> SeasonalNaive(12) by default.
    if metric in SEASONAL_REVENUE_METRICS:
        if y.size >= 12:
            return MODEL_SEASONAL_NAIVE, (
                f"{metric} is a strongly seasonal revenue series for this "
                f"property; SeasonalNaive(12) preserves seasonality (ARIMA "
                f"excluded to avoid flat forecasts)."
            )
        return MODEL_LAST_VALUE, "Insufficient history (<12m) for seasonality."

    # Other metrics: seasonal naive if seasonal, else simple average.
    if is_seasonal(y):
        return MODEL_SEASONAL_NAIVE, "Detected 12-month seasonality."
    return MODEL_SIMPLE_AVERAGE, "No strong seasonality; simple average baseline."


def _forecast_one(metric, history, horizon):
    model, rationale = select_model(metric, history)
    y = np.asarray(history, dtype=float)
    y = y[~np.isnan(y)]
    if model == MODEL_ZERO_FILL:
        preds = np.zeros(horizon)
    elif model == MODEL_SEASONAL_NAIVE:
        preds = _seasonal_naive_forecast(y, horizon, season=12)
    elif model == MODEL_LAST_VALUE:
        preds = np.full(horizon, y[-1] if y.size else 0.0)
    else:  # SimpleAverage
        base = float(np.nanmean(y)) if y.size else 0.0
        preds = np.full(horizon, base)

    # Flat-forecast detector: SimpleAverage and LastValue are flat BY
    # DEFINITION whenever the underlying series isn't seasonal, so this is
    # expected, not an error condition. Surface it as a non-fatal flag (so the
    # caller can mark the forecast low-confidence) instead of raising -- a
    # hard raise here previously killed the ENTIRE pipeline for any metric
    # with too little history for seasonal detection (<24 months), which is
    # the common case, not the exception (e.g. it crashed every run for the
    # uploaded "Test Hospitality" workbook, which has exactly 12 months).
    is_flat = False
    if preds.size and y.size:
        pred_mean = float(np.nanmean(preds))
        pred_std = float(np.nanstd(preds))
        hist_mean = float(np.nanmean(np.abs(y))) if np.any(np.abs(y) > 0) else 0.0
        hist_std = float(np.nanstd(y))
        pred_rel = (pred_std / pred_mean) if pred_mean != 0 else float('inf')
        hist_rel = (hist_std / hist_mean) if hist_mean != 0 else 0.0
        if pred_rel < 0.02 and hist_rel > 0.15:
            is_flat = True

    return preds, model, rationale, is_flat


def fit_in_sample(metric, history):
    """
    Produce a leakage-safe FITTED value for every available (historical)
    month, so available months can be compared model-vs-actual the same way
    future months are -- not just echoed back as bare actuals.

    For month i, the fitted value uses ONLY data strictly before i (an
    expanding-window estimate), or the same-month-prior-year actual once 12
    months of history exist and the metric is seasonal/forced-seasonal. The
    first month can never be fitted (no prior data) and is returned as NaN --
    that is a real, honest limitation, not a bug to paper over.

    Returns (fitted: np.ndarray same length as history, model: str).
    """
    y = np.asarray(history, dtype=float)
    n = y.size
    fitted = np.full(n, np.nan)
    if n == 0:
        return fitted, MODEL_LAST_VALUE

    clean = y[~np.isnan(y)]
    if is_near_zero_sparse(clean):
        fitted[:] = 0.0
        fitted[np.isnan(y)] = np.nan
        return fitted, MODEL_ZERO_FILL

    use_seasonal = metric in FORCE_SEASONAL_NAIVE or is_seasonal(clean)
    model = MODEL_SEASONAL_NAIVE if use_seasonal else MODEL_SIMPLE_AVERAGE

    for i in range(n):
        if i == 0:
            continue  # no prior data to fit on -- left as NaN, by design
        if use_seasonal and i >= 12 and not np.isnan(y[i - 12]):
            fitted[i] = y[i - 12]
        else:
            prior = y[:i]
            prior = prior[~np.isnan(prior)]
            if prior.size:
                fitted[i] = float(np.mean(prior))
    return fitted, model


def forecast_available_months(monthly_df, metrics):
    """
    Companion to forecast_future_months: instead of months strictly after the
    last actual, this produces a FITTED value for each month that already has
    an actual, using fit_in_sample (no leakage -- each month's fitted value
    only looks at strictly earlier months).

    Returns (fitted_df, model_report) where fitted_df has 'Date' + one fitted
    column per metric, aligned 1:1 with monthly_df's rows.
    """
    if "Date" not in monthly_df.columns or monthly_df.empty:
        raise ValueError("monthly_df must contain a non-empty 'Date' column.")

    out = {"Date": list(monthly_df["Date"])}
    model_report = {}
    for metric in metrics:
        if metric not in monthly_df.columns:
            continue
        history = monthly_df[metric].to_numpy(dtype=float)
        fitted, model = fit_in_sample(metric, history)
        out[metric] = fitted
        model_report[metric] = model

    return pd.DataFrame(out), model_report


def forecast_future_months(monthly_df, metrics, horizon=12):
    """
    Forecast each metric for ``horizon`` months strictly AFTER the last actual.

    Args:
        monthly_df: DataFrame with a 'Date' column (month start) + metric cols.
                    These rows are ACTUALS and are never modified here.
        metrics:    canonical metric names to forecast (must exist in df).
        horizon:    number of future months to forecast.

    Returns:
        (future_df, model_report, flags)
        future_df:    DataFrame with 'Date' + forecasted metric columns for the
                      future months only.
        model_report: {metric: {"model": str, "rationale": str}}.
        flags:        list of validation flags (e.g. flat-forecast on seasonal).
    """
    if "Date" not in monthly_df.columns or monthly_df.empty:
        raise ValueError("monthly_df must contain a non-empty 'Date' column.")

    last_actual = pd.to_datetime(monthly_df["Date"]).max()
    future_dates = [last_actual + relativedelta(months=i) for i in range(1, horizon + 1)]

    out = {"Date": future_dates}
    model_report = {}
    flags = []

    for metric in metrics:
        if metric not in monthly_df.columns:
            continue
        history = monthly_df[metric].to_numpy(dtype=float)
        preds, model, rationale, is_flat = _forecast_one(metric, history, horizon)
        out[metric] = preds
        model_report[metric] = {"model": model, "rationale": rationale}

        # Flat-forecast guard on a historically seasonal series.
        hist_clean = history[~np.isnan(history)]
        if metric in SEASONAL_REVENUE_METRICS or is_seasonal(hist_clean):
            if np.ptp(preds) < 1e-6 and hist_clean.size >= 12:
                flags.append({
                    "metric": metric,
                    "flag": "FLAT_FORECAST_ON_SEASONAL_SERIES",
                    "detail": "Forecast is flat but history is seasonal.",
                })
        elif is_flat:
            flags.append({
                "metric": metric,
                "flag": "FLAT_FORECAST_LOW_CONFIDENCE",
                "detail": (
                    f"{model} produced a flat forecast while {metric}'s history is "
                    "variable. Usually means there isn't enough history yet for "
                    "seasonal detection (needs 24+ months); not a hard error."
                ),
            })

    future_df = pd.DataFrame(out)
    return future_df, model_report, flags
