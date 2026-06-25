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
#
# Fairfield/USALI total revenue includes revenue lines beyond Rooms and F&B, so
# Other_Revenue is an explicit child to make top-down scaling mathematically
# correct (Room + F&B + Other == Total_Revenue).
COA_HIERARCHY = {
    "NOI": ["GOP"],
    "GOP": ["Total_Revenue", "Total_UOE"],
    "Total_Revenue": ["Room_Revenue", "FB_Revenue", "Other_Revenue"],
}


class ReconciliationError(ValueError):
    """Raised when parent/child series cannot be safely reconciled."""


def _validate_alignment(node_forecasts, month_index=None):
    """
    Fail loudly if node arrays are not aligned on the same month index.

    All node arrays must share an identical length. When ``month_index`` is
    supplied, its length must match too. This prevents reconciling parent and
    child series that were fitted over inconsistent in-sample regions.
    """
    lengths = {k: len(np.asarray(v)) for k, v in node_forecasts.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        raise ReconciliationError(
            f"Node forecast arrays have mismatched lengths and cannot be "
            f"reconciled on a common month index: {lengths}"
        )
    if month_index is not None and unique_lengths:
        n = next(iter(unique_lengths))
        if len(month_index) != n:
            raise ReconciliationError(
                f"month_index length {len(month_index)} does not match node "
                f"forecast length {n}."
            )


def reconcile_topdown(node_forecasts, hierarchy=None, month_index=None):
    """
    Top-down reconciliation across the COA hierarchy.

    Args:
        node_forecasts: dict {node_name: np.ndarray of monthly forecast values}.
                        All arrays MUST share the same length and be aligned to
                        the same month index (the Excel months).
        hierarchy:      parent -> [children] mapping. Defaults to COA_HIERARCHY.
        month_index:    optional shared month index used to validate alignment.

    Returns:
        dict of reconciled forecasts (same keys/lengths). Parent totals are
        authoritative; children are scaled proportionally to sum to the parent
        per month. Nodes absent from the inputs are skipped gracefully.

    Raises:
        ReconciliationError: if node arrays are not aligned on a common index.
    """
    if hierarchy is None:
        hierarchy = COA_HIERARCHY

    # Reconcile ONLY on aligned forecast arrays. Fail loudly otherwise.
    _validate_alignment(node_forecasts, month_index=month_index)

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


def reconciliation_error(reconciled, hierarchy=None):
    """
    Compute the maximum absolute monthly residual |parent - sum(children)|
    after reconciliation, across all hierarchy relationships present.

    Returns a dict {parent: max_abs_residual} plus an overall 'max' key. Values
    near zero indicate consistent, correctly reconciled totals.
    """
    if hierarchy is None:
        hierarchy = COA_HIERARCHY

    errors = {}
    overall = 0.0
    for parent, children in hierarchy.items():
        if parent not in reconciled:
            continue
        present = [c for c in children if c in reconciled]
        if not present:
            continue
        child_sum = np.sum([reconciled[c] for c in present], axis=0)
        resid = np.abs(np.asarray(reconciled[parent], dtype=float) - child_sum)
        m = float(np.max(resid)) if resid.size else 0.0
        errors[parent] = m
        overall = max(overall, m)
    errors["max"] = overall
    return errors
