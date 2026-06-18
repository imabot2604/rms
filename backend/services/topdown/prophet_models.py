"""Top-level Prophet forecasting for the top-down reconciliation POC.

Trains one monthly Prophet model per top-level driver
(TotalOperatingRevenue, GOP, NOI) with yearly seasonality only, and produces
forecasts for a configurable horizon.

Output schema (long form):
    date, node, forecast, lower, upper
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import hierarchy

logger = logging.getLogger(__name__)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _fit_one(series: pd.DataFrame, node: str, periods: int) -> pd.DataFrame:
    """Fit a single Prophet model on a (ds, y) frame and forecast ``periods``.

    Falls back to a seasonal-naive mean if Prophet is unavailable or fails.
    Returns only the future rows in the long schema.
    """
    try:
        from prophet import Prophet

        if len(series) < 4:
            raise ValueError(f"Not enough points to fit Prophet for {node}")

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(series)
        future = model.make_future_dataframe(periods=periods, freq="MS")
        fc = model.predict(future).tail(periods)
        return pd.DataFrame(
            {
                "date": fc["ds"].values,
                "node": node,
                "forecast": fc["yhat"].values,
                "lower": fc["yhat_lower"].values,
                "upper": fc["yhat_upper"].values,
            }
        )
    except Exception as exc:  # noqa: BLE001 - POC fallback
        logger.warning("Prophet failed for %s (%s); using seasonal-naive fallback.", node, exc)
        return _fallback(series, node, periods)


def _fallback(series: pd.DataFrame, node: str, periods: int) -> pd.DataFrame:
    """Seasonal-naive fallback: repeat the same calendar month from last year."""
    last_date = series["ds"].max()
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    by_month = series.assign(m=series["ds"].dt.month).groupby("m")["y"].mean()
    overall = series["y"].mean()
    point = np.array([by_month.get(d.month, overall) for d in future_dates])
    return pd.DataFrame(
        {
            "date": future_dates,
            "node": node,
            "forecast": point,
            "lower": point * 0.9,
            "upper": point * 1.1,
        }
    )


def forecast_top_level(
    train_df: pd.DataFrame, periods: int = 12
) -> pd.DataFrame:
    """Forecast all top-level driver nodes and return a combined long frame.

    ``train_df`` is the data-quality-clean canonical frame. Each driver uses its
    canonical source column as the Prophet target ``y``.
    """
    out = []
    for node in hierarchy.TOP_LEVEL_FORECAST_NODES:
        col = hierarchy.source_of(node)
        if col not in train_df.columns or train_df[col].isna().all():
            logger.warning("Skipping top-level node %s: column %s missing.", node, col)
            continue
        series = (
            train_df[["Date", col]]
            .dropna()
            .rename(columns={"Date": "ds", col: "y"})
            .reset_index(drop=True)
        )
        out.append(_fit_one(series, node, periods))
    if not out:
        raise ValueError("No top-level series could be forecast.")
    return pd.concat(out, ignore_index=True)
