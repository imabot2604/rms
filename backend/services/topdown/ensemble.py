"""Ensemble layer combining Prophet + ARIMA + ML per node.

Computes validation MAPE for each base model on a holdout window and combines
them with inverse-MAPE weights, producing a single best forecast per node while
still leveraging every base model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-9


def compute_mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, robust to zeros in y_true."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < EPS, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def weighted_ensemble(forecasts_dict: dict, y_true):
    """Inverse-MAPE weighted ensemble.

    ``forecasts_dict`` maps model_name -> predicted series aligned to y_true.
    Returns (ensemble_series, weights). Models with non-finite MAPE are dropped.
    """
    mape = {}
    for name, y_pred in forecasts_dict.items():
        m = compute_mape(y_true, y_pred)
        if np.isfinite(m) and m > 0:
            mape[name] = m
    if not mape:
        # Equal weights fallback.
        names = list(forecasts_dict)
        w = {n: 1.0 / len(names) for n in names}
        ens = sum(w[n] * np.asarray(forecasts_dict[n], dtype=float) for n in names)
        return ens, w

    weights = {name: 1.0 / m for name, m in mape.items()}
    total = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}
    ensemble = sum(
        weights[name] * np.asarray(forecasts_dict[name], dtype=float) for name in weights
    )
    return ensemble, weights


def _holdout_split(series_df: pd.DataFrame, holdout: int):
    """Split a sorted (Date, actual) frame into train / holdout tails."""
    series_df = series_df.sort_values("Date").reset_index(drop=True)
    if len(series_df) <= holdout + 4:
        return None, None
    return series_df.iloc[:-holdout], series_df.iloc[-holdout:]


def ensemble_node(
    node: str,
    actual_df: pd.DataFrame,
    base_forecast_fns: dict,
    horizon: int,
    holdout: int = 6,
) -> pd.DataFrame:
    """Build an inverse-MAPE ensemble forecast for one node.

    Parameters
    ----------
    node : str
        COA node name.
    actual_df : DataFrame
        Long (Date, node, actual) history for the node.
    base_forecast_fns : dict
        model_name -> callable(train_long_df, steps) returning a long forecast
        frame (date, node, forecast, ...). Each callable must accept the
        training history and a horizon and forecast the same node.
    horizon : int
        Final forecast horizon (months).
    holdout : int
        Number of trailing months used to score models.

    Returns a long frame (date, node, forecast, lower, upper) for the ensemble,
    with model weights attached as a ``weights`` attribute on the frame.
    """
    node_hist = actual_df[actual_df["node"] == node][["Date", "node", "actual"]].dropna()
    train, hold = _holdout_split(node_hist, holdout)

    weights = None
    if train is not None:
        preds = {}
        for name, fn in base_forecast_fns.items():
            try:
                fc = fn(train, len(hold))
                fc = fc.sort_values("date").head(len(hold))
                if len(fc) == len(hold):
                    preds[name] = fc["forecast"].values
            except Exception as exc:  # noqa: BLE001
                logger.warning("Holdout scoring failed for %s/%s: %s", node, name, exc)
        if preds:
            _, weights = weighted_ensemble(preds, hold["actual"].values)
            logger.info("Ensemble weights for %s: %s", node, weights)

    # Final forecasts on full history.
    final = {}
    for name, fn in base_forecast_fns.items():
        try:
            fc = fn(node_hist, horizon).sort_values("date").head(horizon)
            if len(fc) == horizon:
                final[name] = fc["forecast"].values
                last_dates = fc["date"].values
        except Exception as exc:  # noqa: BLE001
            logger.warning("Final forecast failed for %s/%s: %s", node, name, exc)

    if not final:
        return pd.DataFrame(columns=["date", "node", "forecast", "lower", "upper"])

    if weights is None:
        weights = {n: 1.0 / len(final) for n in final}
    # Restrict weights to models that produced final forecasts, renormalize.
    weights = {n: w for n, w in weights.items() if n in final}
    if not weights:
        weights = {n: 1.0 / len(final) for n in final}
    wsum = sum(weights.values())
    weights = {n: w / wsum for n, w in weights.items()}

    ensemble = sum(weights[n] * final[n] for n in weights)
    out = pd.DataFrame(
        {
            "date": last_dates,
            "node": node,
            "forecast": ensemble,
            "lower": np.nan,
            "upper": np.nan,
        }
    )
    out.attrs["weights"] = weights
    return out
