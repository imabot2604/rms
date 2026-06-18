"""Tree-based ML forecasting for ADR and event-driven nodes.

Volatile, regressor-sensitive lines (ADR, banquet/restaurant revenue) are
modelled with a RandomForest using calendar, trend, lag and optional external
regressors (comp index, OTB pace, events) already produced by the RMS
simulation (see backend/services/data_processing.py).

All forecasts conform to the standard schema:

    date, node, forecast, lower, upper

``lower``/``upper`` are approximated from RandomForest tree dispersion; they may
be coarse and are intended as indicative bands for the POC.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional external regressors carried alongside a node's series, if present.
OPTIONAL_REGRESSORS = ["Comp_Index", "OTB_Pace_Index", "Event_Flag"]
BASE_FEATURES = ["month", "year", "trend", "lag1"]


def build_features(df: pd.DataFrame, target_node: str):
    """Build a supervised matrix for one node from a long frame.

    ``df`` is long format with ``Date``, ``node``, ``actual`` and may carry
    extra regressor columns merged in by the caller. Returns (X, y, sub) where
    ``sub`` is the aligned, sorted, NaN-dropped node frame.
    """
    sub = df[df["node"] == target_node].copy()
    sub = sub.sort_values("Date")
    sub["month"] = sub["Date"].dt.month
    sub["year"] = sub["Date"].dt.year
    sub["trend"] = range(len(sub))
    sub["lag1"] = sub["actual"].shift(1)

    feature_cols = list(BASE_FEATURES)
    for reg in OPTIONAL_REGRESSORS:
        if reg in sub.columns and not sub[reg].isna().all():
            feature_cols.append(reg)

    sub = sub.dropna(subset=feature_cols + ["actual"])
    X = sub[feature_cols]
    y = sub["actual"]
    return X, y, sub


def fit_random_forest(X, y):
    """Fit a RandomForestRegressor (200 trees, fixed seed)."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X, y)
    return model


def forecast_random_forest(model, X_future):
    """Point forecast from a fitted RandomForest."""
    return model.predict(X_future)


def _rf_interval(model, X_row) -> tuple[float, float]:
    """Approximate a prediction band from per-tree spread."""
    try:
        preds = np.array([est.predict(X_row)[0] for est in model.estimators_])
        return float(preds.mean() - preds.std()), float(preds.mean() + preds.std())
    except Exception:  # noqa: BLE001
        return (np.nan, np.nan)


def forecast_node(
    df: pd.DataFrame, target_node: str, horizon: int
) -> pd.DataFrame:
    """Recursive multi-step RandomForest forecast for one node.

    Lag features are rolled forward using prior predictions; optional regressors
    are held flat at their last observed value (a simple POC assumption).
    Returns the standard schema ``date, node, forecast, lower, upper``.
    """
    X, y, sub = build_features(df, target_node)
    if len(X) < 6:
        logger.warning("Too few points for RF on %s; skipping.", target_node)
        return pd.DataFrame(columns=["date", "node", "forecast", "lower", "upper"])

    model = fit_random_forest(X, y)
    feature_cols = list(X.columns)

    last_date = sub["Date"].max()
    last_trend = int(sub["trend"].max())
    last_actual = float(sub["actual"].iloc[-1])
    flat_reg = {
        reg: float(sub[reg].iloc[-1])
        for reg in OPTIONAL_REGRESSORS
        if reg in feature_cols
    }

    rows = []
    prev = last_actual
    for step in range(1, horizon + 1):
        fdate = last_date + pd.offsets.MonthBegin(step)
        feat = {
            "month": fdate.month,
            "year": fdate.year,
            "trend": last_trend + step,
            "lag1": prev,
        }
        feat.update(flat_reg)
        X_row = pd.DataFrame([[feat[c] for c in feature_cols]], columns=feature_cols)
        point = float(model.predict(X_row)[0])
        lo, hi = _rf_interval(model, X_row)
        rows.append(
            {"date": fdate, "node": target_node, "forecast": point, "lower": lo, "upper": hi}
        )
        prev = point

    return pd.DataFrame(rows)
