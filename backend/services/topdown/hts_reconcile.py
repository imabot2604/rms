"""Formal hierarchical reconciliation using scikit-hts.

This module turns the COA defined in ``hierarchy.py`` into the structure
scikit-hts expects, attaches base forecasts, and runs a reconciliation method
(bottom-up, top-down, middle-out, or OLS/MinT) so every node in the tree is
mutually consistent.

Docs: https://scikit-hts.readthedocs.io

Falls back gracefully (returns the unreconciled base forecasts) if scikit-hts
is not installed, so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import hierarchy

logger = logging.getLogger(__name__)


def build_tree_spec(root: str = "TotalOperatingRevenue") -> Dict[str, List[str]]:
    """Build the scikit-hts ``nodes`` dict: parent -> [children] under a root.

    Only the revenue sub-tree under ``root`` is included (the part that sums
    cleanly), since scikit-hts requires a strict additive hierarchy.
    """
    spec: Dict[str, List[str]] = {}

    def walk(node: str) -> None:
        kids = [
            c
            for c in hierarchy.children_of(node)
            if hierarchy.source_of(c) is not None
        ]
        if kids:
            spec[node] = kids
            for c in kids:
                walk(c)

    walk(root)
    return spec


def _all_nodes(tree_spec: Dict[str, List[str]], root: str) -> List[str]:
    nodes = {root}
    for parent, kids in tree_spec.items():
        nodes.add(parent)
        nodes.update(kids)
    return sorted(nodes)


def build_wide_history(
    actual_long: pd.DataFrame, tree_spec: Dict[str, List[str]], root: str
) -> pd.DataFrame:
    """Build a wide history frame (index=Date, columns=nodes) for scikit-hts."""
    nodes = _all_nodes(tree_spec, root)
    wide = (
        actual_long[actual_long["node"].isin(nodes)]
        .pivot_table(index="Date", columns="node", values="actual")
        .sort_index()
    )
    return wide


def reconcile(
    actual_long: pd.DataFrame,
    base_forecasts: pd.DataFrame,
    horizon: int,
    method: str = "OLS",
    root: str = "TotalOperatingRevenue",
) -> pd.DataFrame:
    """Reconcile base forecasts across the hierarchy with scikit-hts.

    Parameters
    ----------
    actual_long : long (Date, node, actual) history.
    base_forecasts : long (date, node, forecast[, lower, upper]) base forecasts
        for the nodes in the tree (independent per-node predictions).
    horizon : forecast horizon.
    method : 'OLS' (MinT/OLS), 'BU' (bottom-up), 'TD' (top-down), 'MO'
        (middle-out).

    Returns reconciled forecasts in the standard long schema.
    """
    tree_spec = build_tree_spec(root)
    nodes = _all_nodes(tree_spec, root)

    try:
        from hts import HTSRegressor  # type: ignore

        wide_hist = build_wide_history(actual_long, tree_spec, root)
        # scikit-hts learns base models itself; we use auto-arima per node.
        reg = HTSRegressor(model="auto_arima", revision_method=method, n_jobs=0)
        reg = reg.fit(wide_hist, nodes={root: tree_spec.get(root, [])} | tree_spec)
        pred = reg.predict(steps_ahead=horizon)

        future_dates = pd.date_range(
            wide_hist.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        out = []
        for node in nodes:
            if node not in pred.columns:
                continue
            vals = pred[node].tail(horizon).values
            out.append(
                pd.DataFrame(
                    {
                        "date": future_dates,
                        "node": node,
                        "forecast": vals,
                        "lower": np.nan,
                        "upper": np.nan,
                    }
                )
            )
        if out:
            logger.info("scikit-hts reconciliation complete (method=%s).", method)
            return pd.concat(out, ignore_index=True)
        logger.warning("scikit-hts produced no usable columns; returning base.")
        return base_forecasts.copy()

    except ImportError:
        logger.warning(
            "scikit-hts not installed; returning unreconciled base forecasts. "
            "Install with `pip install scikit-hts` to enable formal reconciliation."
        )
        return base_forecasts.copy()
    except Exception as exc:  # noqa: BLE001
        logger.warning("scikit-hts reconciliation failed (%s); returning base.", exc)
        return base_forecasts.copy()
