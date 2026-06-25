"""
Concise Fairfield diagnostic summary.

Given a canonical DataFrame (e.g. Store.master_df from an upload, or the sample
Fairfield frame), prints the selected model per node, the holdout MAPE per node,
and the maximum reconciliation error after top-down scaling.

Usage (programmatic):
    from diagnostics import diagnostic_summary
    summary = diagnostic_summary(df)

This mirrors the validation workflow from the diagnostic report: train on the
uploaded history and report per-node reliability + accounting consistency.
"""

import logging

import numpy as np

from services.excel_timeline import coverage_frame
from services.models import forecast_excel_months
from services.reconciliation import reconcile_topdown, reconciliation_error, COA_HIERARCHY

logger = logging.getLogger(__name__)

DIAG_NODES = [
    "Occupancy_Pct", "ADR", "RevPAR", "Rooms_Sold", "Rooms_Available",
    "Room_Revenue", "FB_Revenue", "Other_Revenue", "Total_Revenue",
    "Total_UOE", "GOP", "NOI",
]


def diagnostic_summary(df, print_summary=True):
    """Build (and optionally print) the per-node + reconciliation diagnostic."""
    arrays = {}
    selected = {}
    mapes = {}
    skipped = []
    shared_index = None

    for node in DIAG_NODES:
        if node not in df.columns or df[node].isna().all():
            skipped.append(node)
            continue
        coverage, _, _ = coverage_frame(df, value_col=node)
        idx = list(coverage["label"])
        if shared_index is None:
            shared_index = idx
        elif idx != shared_index:
            skipped.append(node)
            continue
        observed = coverage.loc[~coverage["is_missing_filled"], "actual"].to_numpy(dtype=float)
        table = forecast_excel_months(coverage, observed, value_col=node)
        arrays[node] = table["forecast"].to_numpy(dtype=float)
        sel = [m for m in table["selected_model"].tolist() if m != "none(missing)"]
        selected[node] = sel[0] if sel else "none"
        mapes[node] = float(table["holdout_mape"].iloc[0])

    reconciled = reconcile_topdown(arrays, COA_HIERARCHY, month_index=shared_index)
    errors = reconciliation_error(reconciled, COA_HIERARCHY)

    summary = {
        "selected_model": selected,
        "holdout_mape": mapes,
        "max_reconciliation_error": errors.get("max"),
        "reconciliation_error_by_parent": {k: v for k, v in errors.items() if k != "max"},
        "skipped_nodes": skipped,
    }

    if print_summary:
        print("=== Fairfield diagnostic summary ===")
        print(f"{'node':<24}{'model':<16}{'holdout_mape':>14}")
        for node in selected:
            mv = mapes[node]
            mtxt = "n/a" if mv is None or np.isinf(mv) else f"{mv:.4f}"
            print(f"{node:<24}{selected[node]:<16}{mtxt:>14}")
        if skipped:
            print(f"skipped (absent/misaligned): {', '.join(skipped)}")
        print(f"max reconciliation error: {summary['max_reconciliation_error']:.6g}")

    return summary
