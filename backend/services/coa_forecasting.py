"""
Per-node forecasting over a generic COA tree (see coa_extraction.py).

Every node -- whether a GL-level leaf ("40010.000 . Transient - Best
Available Rate") or a department rollup ("Total Room Department Revenue")
-- gets BOTH:
  * a fitted value for every month it already has an actual for (no
    leakage: each month's fit only looks at strictly earlier months), and
  * a forecast for `horizon` months after the last actual.

Reconciliation is bottom-up and exact by construction: a verified rollup
node's fitted/forecast value is the sum of its children's fitted/forecast
values, not an independently modeled number that then needs to be
reconciled against them. Only true leaves (and unverified rollups, which
are treated as leaves) are modeled directly.

MODEL SELECTION uses the full candidate suite from services.models:
  Prophet, ARIMA, ETS, Random_Forest, XGBoost, SeasonalNaive
with holdout validation (last ~25% of observed points) to pick the
lowest-MAPE model per node. Near-constant / mostly-zero series fall back
to SimpleAverage or ZeroFill to avoid overfit.
"""

import logging
import numpy as np

from services.models import (
    select_best_model as _select_best_model,
    _fit_predict_in_sample,
    SIMPLE_MODELS,
)

logger = logging.getLogger(__name__)

MODEL_DERIVED_SUM = "DerivedSum(children)"


def _fit_and_forecast_leaf(values, months, node_label, horizon):
    """
    Fit + forecast a single leaf using the full model suite from
    services.models with holdout-validation model selection.

    Candidate models: Prophet, ARIMA, ETS, Random_Forest, XGBoost,
    SeasonalNaive.  Near-constant / mostly-zero series fall back to
    SimpleAverage or ZeroFill.

    Returns (fitted, forecast, model_name).

    fitted     -- np.ndarray same length as values; fitted[i] uses ONLY
                  values[:i] (expanding window).  First data month is NaN.
    forecast   -- np.ndarray length = horizon
    model_name -- str, winning model from holdout validation
    """
    y = np.asarray(values, dtype=float)
    n = y.size

    # Strip leading NaNs so model selection works on the real data window.
    first_valid = 0
    while first_valid < n and np.isnan(y[first_valid]):
        first_valid += 1
    if first_valid >= n:
        return np.full(n, np.nan), np.zeros(horizon), "ZeroFill"

    y_trimmed = y[first_valid:]
    months_trimmed = months[first_valid:] if months else None

    # ----- Model selection via holdout validation (services.models) -----
    best_model, best_mape_val, all_scores = _select_best_model(
        y_trimmed, months=months_trimmed, node=node_label
    )

    # ----- FITTED values (expanding window, no leakage) ------------------
    fitted = np.full(n, np.nan)

    if best_model in SIMPLE_MODELS - {"SeasonalNaive"}:
        if best_model == "ZeroFill":
            fitted[first_valid:] = 0.0
        else:  # SimpleAverage
            for i in range(first_valid + 1, n):
                prior = y[first_valid:i]
                prior = prior[~np.isnan(prior)]
                fitted[i] = float(np.mean(prior)) if prior.size else 0.0
    else:
        # Prophet / ARIMA / ETS / RF / XGBoost / SeasonalNaive
        full_idx = np.arange(n)
        for i in range(first_valid + 1, n):
            train_idx = full_idx[first_valid:i]
            test_idx = np.array([i])
            pred = _fit_predict_in_sample(y, train_idx, test_idx, best_model)
            if pred is not None and len(pred) == 1:
                fitted[i] = float(pred[0])
        # Fill any gaps with expanding-window mean.
        for i in range(first_valid + 1, n):
            if np.isnan(fitted[i]):
                prior = y[first_valid:i]
                prior = prior[~np.isnan(prior)]
                fitted[i] = float(np.mean(prior)) if prior.size else 0.0

    # ----- FORECAST for horizon months beyond last actual ---------------
    if best_model in SIMPLE_MODELS - {"SeasonalNaive"}:
        if best_model == "ZeroFill":
            forecast = np.zeros(horizon)
        else:
            base = float(np.mean(y_trimmed[~np.isnan(y_trimmed)])) if y_trimmed.size else 0.0
            forecast = np.full(horizon, base)
    else:
        train_idx = np.arange(first_valid, n)
        test_idx = np.arange(n, n + horizon)
        preds = _fit_predict_in_sample(y, train_idx, test_idx, best_model)
        if preds is not None and len(preds) == horizon:
            forecast = np.asarray(preds, dtype=float)
        elif preds is not None and len(preds) > 0:
            forecast = np.full(horizon, float(preds[-1]))
            forecast[:len(preds)] = preds
        else:
            base = float(np.mean(y_trimmed[~np.isnan(y_trimmed)])) if y_trimmed.size else 0.0
            forecast = np.full(horizon, base)
            best_model = f"{best_model}(fallback:SimpleAverage)"

    return fitted, forecast, best_model


def forecast_coa_tree(nodes, roots, months, horizon=12):
    """
    Walk the tree bottom-up and attach fitted/forecast arrays + model name
    to every node. Uses the FULL model suite with holdout validation per
    leaf. Mutates CoaNode objects in place.
    """
    n_months = len(months)
    future_labels = _future_month_labels(months[-1], horizon) if months else []

    depth_cache = {}

    def _depth(node):
        if node.id in depth_cache:
            return depth_cache[node.id]
        d = 0 if not node.children else 1 + max(_depth(c) for c in node.children)
        depth_cache[node.id] = d
        return d

    processing_order = sorted(nodes.values(), key=_depth)

    for node in processing_order:
        if node.children:
            child_fitted = np.array([c.fitted for c in node.children])
            child_forecast = np.array([c.forecast for c in node.children])
            node.fitted = np.nansum(child_fitted, axis=0)
            all_nan_mask = np.all(np.isnan(child_fitted), axis=0)
            node.fitted[all_nan_mask] = np.nan
            node.forecast = np.sum(child_forecast, axis=0)
            node.model = MODEL_DERIVED_SUM
        else:
            node.fitted, node.forecast, node.model = _fit_and_forecast_leaf(
                node.values, months, node.label, horizon
            )

    rows = []
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        for i, month in enumerate(months):
            rows.append({
                "node_id": node.id,
                "label": node.label,
                "parent_id": node.parent.id if node.parent else None,
                "is_leaf": len(node.children) == 0,
                "month": month.strftime("%b %Y"),
                "period": "historical",
                "actual": _safe(node.values[i]),
                "fitted": _safe(node.fitted[i]),
                "forecast": None,
                "model": node.model,
                "holdout_mape": None,
            })
        for j, label in enumerate(future_labels):
            rows.append({
                "node_id": node.id,
                "label": node.label,
                "parent_id": node.parent.id if node.parent else None,
                "is_leaf": len(node.children) == 0,
                "month": label,
                "period": "future",
                "actual": None,
                "fitted": None,
                "forecast": _safe(node.forecast[j]),
                "model": node.model,
                "holdout_mape": None,
            })
    return rows


def reconciliation_diagnostics(nodes):
    diagnostics = []
    for node in nodes.values():
        if not node.children:
            continue
        child_sum = np.nansum(np.array([c.values for c in node.children]), axis=0)
        resid = np.nanmax(np.abs(node.values - child_sum)) if node.values.size else 0.0
        diagnostics.append({
            "node_id": node.id, "label": node.label,
            "max_residual": _safe(resid),
        })
    return diagnostics


def _future_month_labels(last_month, horizon):
    out = []
    y, m = last_month.year, last_month.month
    for _ in range(horizon):
        m += 1
        if m > 12:
            m = 1
            y += 1
        out.append(f"{['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]} {y}")
    return out


def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)
