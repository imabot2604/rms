"""
Layer 4: hierarchical reconciliation for Fairfield/Marriott P&L.

Identities enforced:
    Total_Revenue = Room_Revenue + FB_Revenue + Other_Revenue (children present)
    GOP           = Total_Revenue - Total_UOE
    NOI           = source Net Income loss  (NOT forced equal to GOP)

Rules (root-cause fixes):
  * HISTORICAL months use ACTUAL parent and child source values and are left
    untouched (no model substitution).
  * FUTURE months are reconciled BOTTOM-UP: parents are recomputed from the
    forecast children rather than mixing incompatible model outputs.
  * No blind scale factor is ever applied across mismatched model ranges.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

REVENUE_CHILDREN = ["Room_Revenue", "FB_Revenue", "Other_Revenue"]


class HierarchyError(ValueError):
    """Raised when reconciliation inputs are structurally invalid."""


def reconcile_future(forecast_by_metric):
    """
    Bottom-up reconciliation of FUTURE forecast months.

    Args:
        forecast_by_metric: dict {metric: np.ndarray} all aligned to the same
                            future month index.

    Returns:
        (reconciled, notes)
        reconciled: dict with parents recomputed from children where children
                    exist. Total_Revenue := sum(present revenue children);
                    GOP := Total_Revenue - Total_UOE (if Total_UOE present).
                    NOI is preserved as its own forecast.
        notes:      list of human-readable reconciliation notes.
    """
    # Validate alignment.
    lengths = {k: len(np.asarray(v)) for k, v in forecast_by_metric.items()}
    if len(set(lengths.values())) > 1:
        raise HierarchyError(f"Misaligned forecast lengths: {lengths}")

    out = {k: np.asarray(v, dtype=float).copy() for k, v in forecast_by_metric.items()}
    notes = []

    # Total_Revenue from present revenue children (bottom-up).
    present_children = [c for c in REVENUE_CHILDREN if c in out]
    if present_children:
        recomputed_total = np.sum([out[c] for c in present_children], axis=0)
        if "Total_Revenue" in out:
            notes.append(
                "Total_Revenue recomputed from children "
                f"({' + '.join(present_children)}) for future months."
            )
        out["Total_Revenue"] = recomputed_total

    # GOP = Total_Revenue - Total_UOE (only if both available).
    if "Total_Revenue" in out and "Total_UOE" in out:
        out["GOP"] = out["Total_Revenue"] - out["Total_UOE"]
        notes.append("GOP recomputed as Total_Revenue - Total_UOE for future months.")

    # NOI is intentionally NOT forced to equal GOP; it keeps its own forecast.
    if "NOI" in out:
        notes.append("NOI preserved from its own source-backed forecast (not forced = GOP).")

    return out, notes


def hierarchy_residuals(metric_to_values):
    """
    Compute reconciliation residuals for a set of aligned arrays.

    Returns {check_name: max_abs_residual, ..., 'max': overall_max}.
    """
    errors = {}
    overall = 0.0

    present_children = [c for c in REVENUE_CHILDREN if c in metric_to_values]
    if "Total_Revenue" in metric_to_values and present_children:
        child_sum = np.sum([metric_to_values[c] for c in present_children], axis=0)
        resid = np.abs(np.asarray(metric_to_values["Total_Revenue"], dtype=float) - child_sum)
        errors["Total_Revenue=sum(children)"] = float(np.max(resid)) if resid.size else 0.0
        overall = max(overall, errors["Total_Revenue=sum(children)"])

    if all(k in metric_to_values for k in ("GOP", "Total_Revenue", "Total_UOE")):
        expected_gop = np.asarray(metric_to_values["Total_Revenue"], dtype=float) \
            - np.asarray(metric_to_values["Total_UOE"], dtype=float)
        resid = np.abs(np.asarray(metric_to_values["GOP"], dtype=float) - expected_gop)
        errors["GOP=Total_Revenue-Total_UOE"] = float(np.max(resid)) if resid.size else 0.0
        overall = max(overall, errors["GOP=Total_Revenue-Total_UOE"])

    errors["max"] = overall
    return errors
