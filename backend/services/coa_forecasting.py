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
"""

import logging

import numpy as np

from services.pnl_forecasting import (
    is_near_zero_sparse, is_seasonal, _seasonal_naive_forecast,
    MODEL_SEASONAL_NAIVE, MODEL_ZERO_FILL, MODEL_SIMPLE_AVERAGE, MODEL_LAST_VALUE,
)

logger = logging.getLogger(__name__)

MODEL_DERIVED_SUM = "DerivedSum(children)"


def _select_generic_model(values):
    """Model selection with no metric-name special-casing -- works for any
    arbitrary COA label, unlike pnl_forecasting.select_model which keys off
    a fixed set of canonical metric names."""
    y = np.asarray(values, dtype=float)
    clean = y[~np.isnan(y)]
    if is_near_zero_sparse(clean):
        return MODEL_ZERO_FILL
    if clean.size >= 24 and is_seasonal(clean):
        return MODEL_SEASONAL_NAIVE
    if clean.size >= 1:
        return MODEL_SIMPLE_AVERAGE
    return MODEL_LAST_VALUE


def _fit_and_forecast_leaf(values, horizon):
    """Returns (fitted: np.ndarray same length as values, forecast: np.ndarray
    length horizon, model: str). No leakage: fitted[i] only uses values[:i]."""
    y = np.asarray(values, dtype=float)
    n = y.size
    model = _select_generic_model(y)
    fitted = np.full(n, np.nan)

    for i in range(1, n):
        if model == MODEL_ZERO_FILL:
            fitted[i] = 0.0
        elif model == MODEL_SEASONAL_NAIVE and i >= 12 and not np.isnan(y[i - 12]):
            fitted[i] = y[i - 12]
        else:
            prior = y[:i]
            prior = prior[~np.isnan(prior)]
            if prior.size:
                fitted[i] = float(np.mean(prior))

    clean = y[~np.isnan(y)]
    if model == MODEL_ZERO_FILL:
        forecast = np.zeros(horizon)
    elif model == MODEL_SEASONAL_NAIVE:
        forecast = _seasonal_naive_forecast(clean, horizon, season=12)
    elif model == MODEL_LAST_VALUE or clean.size == 0:
        forecast = np.full(horizon, clean[-1] if clean.size else 0.0)
    else:
        forecast = np.full(horizon, float(np.mean(clean)))

    return fitted, forecast, model


def forecast_coa_tree(nodes, roots, months, horizon=12):
    """
    Walk the tree bottom-up (children before parents -- guaranteed by sheet
    order, since a rollup row always appears after the rows it sums) and
    attach fitted/forecast arrays + model name to every node.

    Mutates the CoaNode objects in `nodes` in place by setting:
        node.fitted    -- np.ndarray, len(months), NaN for first month
        node.forecast  -- np.ndarray, len(horizon)
        node.model     -- str

    Returns a flat list of per-node, per-month dict rows ready for JSON.
    """
    n_months = len(months)
    future_labels = _future_month_labels(months[-1], horizon) if months else []

    # Process children before parents. Sorting by node_id only happens to
    # work when IDs are row numbers (a rollup row always has a higher row
    # number than what it sums); it silently breaks for other ID schemes,
    # e.g. the path-tuple IDs used by extract_coa_tree_multi for merged
    # multi-year trees. Compute an explicit depth (0 = leaf, 1 + max(child
    # depth) otherwise) and process deepest-first instead, which is correct
    # for any node ID type.
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
            # If EVERY child is NaN at a given month (e.g. month 0), keep NaN
            # rather than fabricating a 0.
            all_nan_mask = np.all(np.isnan(child_fitted), axis=0)
            node.fitted[all_nan_mask] = np.nan
            node.forecast = np.sum(child_forecast, axis=0)
            node.model = MODEL_DERIVED_SUM
        else:
            node.fitted, node.forecast, node.model = _fit_and_forecast_leaf(node.values, horizon)

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
                "fitted": _safe(node.fitted[i]) if i > 0 else None,
                "forecast": None,
                "model": node.model if i > 0 else None,
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
            })
    return rows


def reconciliation_diagnostics(nodes):
    """Max |actual - sum(children actual)| per verified-rollup node, so a
    reconciliation problem is visible rather than silently absorbed."""
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
        out.append(f"{['', 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]} {y}")
    return out


def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)
