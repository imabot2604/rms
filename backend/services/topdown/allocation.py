"""Top-down allocation and reconciliation for the POC.

Computes smoothed historical allocation shares from a parent node to its
children, then allocates a parent forecast down to children so that children
sum exactly back to the parent (within tolerance) at every level.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import hierarchy

logger = logging.getLogger(__name__)


def smoothed_shares(
    train_df: pd.DataFrame, parent: str, window: int = 12
) -> dict[str, float]:
    """Return normalized average child shares of ``parent`` over the last window.

    For each month, share_child = child_value / parent_value. Shares are
    averaged over the trailing ``window`` months and then renormalized so they
    sum to exactly 1.0 (guaranteeing reconciliation).
    """
    children = hierarchy.children_of(parent)
    parent_col = hierarchy.source_of(parent)
    if parent_col not in train_df.columns:
        raise ValueError(f"Parent column {parent_col} missing for {parent}.")

    recent = train_df.sort_values("Date").tail(window)
    parent_vals = pd.to_numeric(recent[parent_col], errors="coerce")

    raw: dict[str, float] = {}
    for child in children:
        col = hierarchy.source_of(child)
        if col not in train_df.columns:
            continue
        child_vals = pd.to_numeric(recent[col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(parent_vals > 0, child_vals / parent_vals, np.nan)
        mean_ratio = np.nanmean(ratio)
        raw[child] = float(mean_ratio) if np.isfinite(mean_ratio) else 0.0

    total = sum(raw.values())
    if total <= 0:
        # Even split if no usable history.
        n = len(raw) or 1
        return {child: 1.0 / n for child in raw}
    return {child: val / total for child, val in raw.items()}


def allocate(
    parent_forecast: pd.DataFrame, shares: dict[str, float], parent: str
) -> pd.DataFrame:
    """Allocate a parent forecast frame down to children using ``shares``.

    ``parent_forecast`` is a long frame (date, node, forecast, lower, upper)
    filtered to a single parent node. Returns a long frame for the children.
    """
    pf = parent_forecast[parent_forecast["node"] == parent]
    rows = []
    for _, r in pf.iterrows():
        for child, share in shares.items():
            rows.append(
                {
                    "date": r["date"],
                    "node": child,
                    "forecast": r["forecast"] * share,
                    "lower": r["lower"] * share,
                    "upper": r["upper"] * share,
                    "share": share,
                    "parent": parent,
                }
            )
    return pd.DataFrame(rows)


def allocate_recursive(
    top_forecast: pd.DataFrame, train_df: pd.DataFrame, parent: str, window: int = 12
) -> pd.DataFrame:
    """Allocate a parent down through every descendant level.

    Walks the hierarchy breadth-first from ``parent``, allocating each node's
    forecast to its children using smoothed shares. Returns a combined long
    frame of all allocated descendant nodes.
    """
    results = []
    frontier = [parent]
    forecasts_by_node = {parent: top_forecast[top_forecast["node"] == parent]}

    while frontier:
        node = frontier.pop(0)
        children = hierarchy.children_of(node)
        children = [c for c in children if hierarchy.source_of(c) in train_df.columns]
        if not children:
            continue
        shares = smoothed_shares(train_df, node, window)
        # Keep only children that actually received a share.
        shares = {c: s for c, s in shares.items() if c in children}
        if not shares:
            continue
        allocated = allocate(forecasts_by_node[node], shares, node)
        results.append(allocated)
        for child in shares:
            forecasts_by_node[child] = allocated[allocated["node"] == child]
            frontier.append(child)

    if not results:
        return pd.DataFrame(columns=["date", "node", "forecast", "lower", "upper", "share", "parent"])
    return pd.concat(results, ignore_index=True)


def reconciliation_error(
    parent_forecast: pd.DataFrame, child_forecast: pd.DataFrame, parent: str
) -> pd.DataFrame:
    """Return per-date reconciliation error: parent - sum(children).

    A correctly reconciled allocation yields errors at ~0 (numerical tolerance).
    """
    pf = (
        parent_forecast[parent_forecast["node"] == parent]
        .set_index("date")["forecast"]
        .rename("parent")
    )
    children = hierarchy.children_of(parent)
    cf = (
        child_forecast[child_forecast["node"].isin(children)]
        .groupby("date")["forecast"]
        .sum()
        .rename("children_sum")
    )
    merged = pd.concat([pf, cf], axis=1).fillna(0.0)
    merged["error"] = merged["parent"] - merged["children_sum"]
    return merged.reset_index()
