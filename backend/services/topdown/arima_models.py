"""ARIMA/ETS forecasting for smooth expense lines.

Stable, slow-moving cost lines (property taxes, insurance, management fees,
utilities) are well modelled by ARIMA/ETS rather than Prophet. This module
wraps statsmodels SARIMAX as a generic ARIMA forecaster and exposes a helper
that takes a long ``(Date, node, actual)`` frame (from ``data_loader``) and
returns forecasts in the standard schema:

    date, node, forecast, lower, upper

A seasonal-naive fallback is used if statsmodels is unavailable or fitting
fails, mirroring the Prophet module's resilience.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_arima(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    """Fit a SARIMAX model on a 1-D series and return the fitted result.

    ``enforce_stationarity`` / ``enforce_invertibility`` are disabled so short
    monthly hotel series fit robustly.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_arima(fitted_model, steps):
    """Forecast ``steps`` ahead, returning (mean, lower, upper) series."""
    fc = fitted_model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int()
    lower, upper = conf.iloc[:, 0], conf.iloc[:, 1]
    return mean, lower, upper


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


def _fallback(dates: pd.DataFrame, node: str, horizon: int) -> pd.DataFrame:
    """Seasonal-naive fallback: per-calendar-month mean (or overall mean)."""
    last_date = dates["Date"].max()
    future = _future_dates(last_date, horizon)
    by_month = dates.assign(m=dates["Date"].dt.month).groupby("m")["actual"].mean()
    overall = dates["actual"].mean()
    point = np.array([by_month.get(d.month, overall) for d in future])
    return pd.DataFrame(
        {
            "date": future,
            "node": node,
            "forecast": point,
            "lower": point * 0.9,
            "upper": point * 1.1,
        }
    )


def forecast_cost_node(
    node_name: str,
    long_df: pd.DataFrame,
    horizon: int,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 0, 12),
) -> pd.DataFrame:
    """Forecast a single cost node from a long (Date, node, actual) frame.

    Extracts the node's series, fits ARIMA, and returns the standard schema
    ``date, node, forecast, lower, upper``. Falls back to seasonal-naive on
    failure or insufficient data.
    """
    sub = (
        long_df[long_df["node"] == node_name]
        .dropna(subset=["actual"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if sub.empty:
        logger.warning("No data for cost node %s; skipping.", node_name)
        return pd.DataFrame(columns=["date", "node", "forecast", "lower", "upper"])

    if len(sub) < 6:
        logger.warning("Too few points for ARIMA on %s; using fallback.", node_name)
        return _fallback(sub, node_name, horizon)

    series = pd.Series(sub["actual"].values, index=pd.DatetimeIndex(sub["Date"]))
    try:
        fitted = fit_arima(series, order=order, seasonal_order=seasonal_order)
        mean, lower, upper = forecast_arima(fitted, horizon)
        future = _future_dates(sub["Date"].max(), horizon)
        return pd.DataFrame(
            {
                "date": future,
                "node": node_name,
                "forecast": np.asarray(mean),
                "lower": np.asarray(lower),
                "upper": np.asarray(upper),
            }
        )
    except Exception as exc:  # noqa: BLE001 - POC fallback
        logger.warning("ARIMA failed for %s (%s); using fallback.", node_name, exc)
        return _fallback(sub, node_name, horizon)


def forecast_cost_nodes(
    node_names, long_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Forecast multiple cost nodes and return a combined long frame."""
    out = [forecast_cost_node(n, long_df, horizon) for n in node_names]
    out = [d for d in out if not d.empty]
    if not out:
        return pd.DataFrame(columns=["date", "node", "forecast", "lower", "upper"])
    return pd.concat(out, ignore_index=True)
