"""
COA (Chart of Accounts) hierarchy and top-down reconciliation.

Forecasts are produced per node (account). To keep totals consistent across the
COA hierarchy, child node forecasts must sum to their parent totals for every
month. This module performs top-down reconciliation: the parent (top) forecast
is treated as authoritative and child forecasts are scaled proportionally so
that the children sum to the parent for each month.

This preserves the existing reconciliation contract while operating strictly on
the months represented by the uploaded Excel (no extra horizons).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Canonical hospitality COA hierarchy: parent -> direct children.
# Mirrors the canonical schema produced by data_processing._map_columns_to_schema.
COA_HIERARCHY = {
    "Total_Revenue": ["Room_Revenue", "FB_Revenue"],
    "GOP": ["Total_Revenue", "Total_UOE"],
    "NOI": ["GOP"],
}


def reconcile_topdown(node_forecasts, hierarchy=None):
    """
    Top-down reconciliation across the COA hierarchy.

    Args:
        node_forecasts: dict {node_name: np.ndarray of monthly forecast values}.
                        All arrays must share the same length (the Excel months).
        hierarchy:      parent -> [children] mapping. Defaults to COA_HIERARCHY.

    Returns:
        dict of reconciled forecasts (same keys/lengths). Parent totals are
        authoritative; children are scaled proportionally to sum to the parent
        per month. Nodes absent from the inputs are skipped gracefully.
    """
    if hierarchy is None:
        hierarchy = COA_HIERARCHY

    reconciled = {k: np.asarray(v, dtype=float).copy() for k, v in node_forecasts.items()}

    for parent, children in hierarchy.items():
        if parent not in reconciled:
            continue
        present_children = [c for c in children if c in reconciled]
        if not present_children:
            continue

        parent_vals = reconciled[parent]
        n = len(parent_vals)
        child_stack = np.vstack([reconciled[c] for c in present_children])
        child_sum = child_stack.sum(axis=0)

        for t in range(n):
            total = child_sum[t]
            target = parent_vals[t]
            if abs(total) > 1e-9:
                scale = target / total
                for ci, c in enumerate(present_children):
                    reconciled[c][t] = child_stack[ci, t] * scale
            elif len(present_children) > 0:
                # No signal from children: distribute parent equally.
                share = target / len(present_children)
                for c in present_children:
                    reconciled[c][t] = share

    return reconciled
