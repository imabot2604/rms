"""
Regression tests for the generic COA hierarchy extraction + forecasting.

Uses a small synthetic workbook with a known, verifiable rollup structure
(rather than property-specific fixtures) so it stays valid for any property.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.coa_extraction import extract_coa_tree  # noqa: E402
from services.coa_forecasting import forecast_coa_tree, reconciliation_diagnostics  # noqa: E402


def _synthetic_workbook():
    """
    Two leaf GL accounts roll up into a department total, which itself rolls
    up into a higher-level total -- mirrors the real shape (GL leaf ->
    department subtotal -> Total Operating Revenue) without being tied to
    any specific property's wording.
    """
    months = [pd.Timestamp(2023, 1, 1) + pd.DateOffset(months=i) for i in range(12)]
    month_labels = [m.strftime("%b, %Y") for m in months]

    leaf_a = [1000.0 + 50 * i for i in range(12)]
    leaf_b = [500.0 + 20 * i for i in range(12)]
    dept_total = [leaf_a[i] + leaf_b[i] for i in range(12)]
    other_dept = [300.0 + 10 * i for i in range(12)]
    grand_total = [dept_total[i] + other_dept[i] for i in range(12)]

    rows = [
        [""] + month_labels,
        ["40010.000 . Account A"] + [str(v) for v in leaf_a],
        ["40020.000 . Account B"] + [str(v) for v in leaf_b],
        ["Total Room Department Revenue"] + [str(v) for v in dept_total],
        ["Total Other Revenue"] + [str(v) for v in other_dept],
        ["Total Operating Revenue"] + [str(v) for v in grand_total],
    ]
    return pd.DataFrame(rows), {
        "leaf_a": leaf_a, "leaf_b": leaf_b, "dept_total": dept_total,
        "other_dept": other_dept, "grand_total": grand_total,
    }


def test_hierarchy_reconstructed_from_numbers():
    raw, truth = _synthetic_workbook()
    nodes, roots, months, debug = extract_coa_tree(raw)
    assert debug["value_row_count"] == 5

    dept = next(n for n in nodes.values() if n.label == "Total Room Department Revenue")
    assert dept.rollup_verified
    assert {c.label for c in dept.children} == {"40010.000 . Account A", "40020.000 . Account B"}

    grand = next(n for n in nodes.values() if n.label == "Total Operating Revenue")
    assert grand.rollup_verified
    assert {c.label for c in grand.children} == {"Total Room Department Revenue", "Total Other Revenue"}


def test_leaf_actuals_match_source():
    raw, truth = _synthetic_workbook()
    nodes, roots, months, debug = extract_coa_tree(raw)
    leaf_a = next(n for n in nodes.values() if n.label == "40010.000 . Account A")
    assert np.allclose(leaf_a.values, truth["leaf_a"], atol=0.01)


def test_rollups_reconcile_exactly():
    raw, truth = _synthetic_workbook()
    nodes, roots, months, debug = extract_coa_tree(raw)
    recon = reconciliation_diagnostics(nodes)
    assert len(recon) == 2  # two verified rollups: dept total and grand total
    assert all((d["max_residual"] or 0.0) < 1e-6 for d in recon)


def test_fitted_has_no_leakage_and_forecast_is_populated():
    raw, truth = _synthetic_workbook()
    nodes, roots, months, debug = extract_coa_tree(raw)
    rows = forecast_coa_tree(nodes, roots, months, horizon=3)

    leaf_a_rows = [r for r in rows if r["label"] == "40010.000 . Account A"]
    historical = [r for r in leaf_a_rows if r["period"] == "historical"]
    future = [r for r in leaf_a_rows if r["period"] == "future"]

    assert len(historical) == 12
    assert len(future) == 3
    # First month can never be fitted (no prior data).
    assert historical[0]["fitted"] is None
    # Later months ARE fitted.
    assert historical[-1]["fitted"] is not None
    # Future months have a forecast and no actual.
    assert all(r["forecast"] is not None and r["actual"] is None for r in future)


def test_parent_forecast_equals_sum_of_children_forecast():
    raw, truth = _synthetic_workbook()
    nodes, roots, months, debug = extract_coa_tree(raw)
    rows = forecast_coa_tree(nodes, roots, months, horizon=3)

    dept = next(n for n in nodes.values() if n.label == "Total Room Department Revenue")
    parent_future = [r["forecast"] for r in rows
                      if r["node_id"] == dept.id and r["period"] == "future"]
    child_sum_future = np.sum(
        [[r["forecast"] for r in rows if r["node_id"] == c.id and r["period"] == "future"]
         for c in dept.children],
        axis=0,
    )
    assert np.allclose(parent_future, child_sum_future, atol=1e-6)


def _synthetic_two_year_workbook_with_label_collision(year):
    """
    Reproduces the real-world bug found against this property's actual
    2023/2024/2025 workbooks: TWO unrelated root-level rows share the exact
    same label ("Total Departmental Expenses") within a single year -- one
    is a verified rollup of a small department-level summary, the other is
    an unrelated, much larger total elsewhere in the same sheet that does
    NOT roll up from anything nearby. A naive label-only join across years
    must not let the second occurrence silently overwrite the first.
    """
    months = [pd.Timestamp(year, 1, 1) + pd.DateOffset(months=i) for i in range(12)]
    month_labels = [m.strftime("%b, %Y") for m in months]

    room = [200.0 + 5 * year + i for i in range(12)]
    fb = [100.0 + 2 * year + i for i in range(12)]
    dept_total = [room[i] + fb[i] for i in range(12)]
    unrelated_total = [9000.0 + 50 * year + i for i in range(12)]  # does NOT equal any nearby sum

    rows = [
        [""] + month_labels,
        ["Room Department"] + [str(v) for v in room],
        ["Food and Beverages Department"] + [str(v) for v in fb],
        ["Total Departmental Expenses"] + [str(v) for v in dept_total],
        ["Unrelated Section Header"] + ["" for _ in range(12)],
        ["Total Departmental Expenses"] + [str(v) for v in unrelated_total],
    ]
    return pd.DataFrame(rows), {"dept_total": dept_total, "unrelated_total": unrelated_total}


def test_multi_year_label_collision_does_not_overwrite():
    from services.coa_extraction import extract_coa_tree_multi

    raw_2023, truth_2023 = _synthetic_two_year_workbook_with_label_collision(2023)
    raw_2024, truth_2024 = _synthetic_two_year_workbook_with_label_collision(2024)

    nodes, roots, months, debug = extract_coa_tree_multi([raw_2023, raw_2024])

    first = nodes[("total departmental expenses",)]
    second = nodes[("total departmental expenses", "__occurrence_2")]

    assert len(first.children) == 2  # the verified rollup keeps its children
    assert len(second.children) == 0  # the unrelated total stays unverified, not merged in

    assert np.allclose(first.values[:12], truth_2023["dept_total"], atol=0.01)
    assert np.allclose(second.values[:12], truth_2023["unrelated_total"], atol=0.01)
    assert np.allclose(first.values[12:], truth_2024["dept_total"], atol=0.01)
    assert np.allclose(second.values[12:], truth_2024["unrelated_total"], atol=0.01)

    # The collision must be reported, not silently absorbed.
    assert len(debug["label_collisions"]) >= 2  # one per year
