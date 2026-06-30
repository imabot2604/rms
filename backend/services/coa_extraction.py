"""
Generic Chart-of-Accounts (COA) hierarchy extraction for row-label P&L
workbooks.

Why this exists
----------------
``pnl_extraction.py`` only recognizes ~10 canonical buckets (Room_Revenue,
FB_Revenue, Total_Revenue, GOP, NOI, ...). Real workbooks like a hotel's
monthly Income Statement contain hundreds of line items below that --
department-level subtotals (Transient Revenue, Group Revenue, Banquet & 
Catering, Restaurant One/Two, Lounge, Room Service, Gift Shop ...) and
GL-coded leaf accounts (e.g. "40010.000 . Transient - Best Available Rate").
None of that is forecastable today. This module builds the FULL multi-level
tree directly from the numbers, instead of a fixed alias list, so it works
on any property's export, not one specific template.

Algorithm
---------
A "Total X" row is recognized as a rollup of the most recent CONTIGUOUS run
of not-yet-claimed sibling rows whose values sum to it, within tolerance.
This is discovered bottom-up, in sheet order:

  for each value-bearing row, in order:
      if its label looks like a rollup ("Total ...", "Gross ...", etc.):
          try matching it against a growing backward window of unclaimed
          rows immediately above it; the smallest window that sums to the
          rollup's own values (within tolerance) becomes its children.
          if no window matches, it is left as a standalone node and the
          mismatch is recorded (not hidden) as an "unverified rollup" --
          common where a workbook's true total spans non-contiguous
          sections (e.g. revenue-only rollups interleaved with expense
          rows for the same department).
      push the row onto the pending stack either way, so a LATER, higher
      rollup can still claim it.

This makes no assumption about row wording, GL numbering scheme, or
property type -- it only assumes the workbook's own totals add up, which is
an accounting identity, not a property-specific convention.
"""

import logging

import numpy as np
import pandas as pd

from services.pnl_extraction import _find_label_and_month_columns, normalize_label

logger = logging.getLogger(__name__)

ABS_TOL = 1.0          # absolute dollar tolerance for a rollup match
REL_TOL = 0.005         # additional relative tolerance (0.5%) for large totals

_ROLLUP_PREFIXES = ("total ", "grand total ", "gross operating", "net operating",
                     "net income")


def _is_rollup_label(label):
    l = label.strip().lower()
    return l.startswith(_ROLLUP_PREFIXES)


def _clean_numeric(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return np.nan
        s = s.replace(",", "").replace("$", "")
        neg = s.startswith("(") and s.endswith(")")
        if neg:
            s = s[1:-1]
        v = float(s)
        return -v if neg else v
    except (TypeError, ValueError):
        return np.nan


class CoaNode:
    __slots__ = ("id", "row", "label", "values", "children", "parent",
                 "rollup_label", "rollup_verified", "fitted", "forecast", "model")

    def __init__(self, node_id, row, label, values):
        self.id = node_id
        self.row = row
        self.label = label
        self.values = values  # np.ndarray, one value per month, aligned to `months`
        self.children = []    # list of CoaNode
        self.parent = None    # CoaNode or None
        self.rollup_label = _is_rollup_label(label)
        self.rollup_verified = False  # True once children are confirmed to sum to it
        self.fitted = None    # set by coa_forecasting.forecast_coa_tree
        self.forecast = None  # set by coa_forecasting.forecast_coa_tree
        self.model = None     # set by coa_forecasting.forecast_coa_tree

    def to_dict(self):
        return {
            "id": self.id,
            "row": self.row,
            "label": self.label,
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [c.id for c in self.children],
            "is_leaf": len(self.children) == 0,
            "rollup_label": self.rollup_label,
            "rollup_verified": self.rollup_verified,
        }


def extract_coa_tree(df_raw, max_scan=40):
    """
    Build the full COA hierarchy from a header-less raw workbook frame.

    Returns:
        nodes:   dict {node_id: CoaNode} for every value-bearing row.
        roots:   list of node_id with no parent (top-level nodes).
        months:  list of pd.Timestamp, sorted ascending.
        debug:   {"unverified_rollups": [...], "value_row_count": int,
                  "root_count": int}
    """
    header_idx, label_col, month_col_map = _find_label_and_month_columns(df_raw, max_scan=max_scan)
    if header_idx is None or not month_col_map:
        raise ValueError("Could not locate a monthly header row with month columns.")

    month_positions = sorted(month_col_map)
    months = [month_col_map[p] for p in month_positions]

    nodes = {}
    pending = []          # stack of CoaNode not yet claimed by a parent
    unverified_rollups = []

    for r in range(header_idx + 1, len(df_raw)):
        raw_label = df_raw.iat[r, label_col]
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            continue
        label = str(raw_label).strip()

        values = np.array(
            [_clean_numeric(df_raw.iat[r, p]) if p < df_raw.shape[1] else np.nan
             for p in month_positions],
            dtype=float,
        )
        if np.all(np.isnan(values)):
            continue  # pure section header with no data; not a COA node

        node = CoaNode(node_id=r, row=r, label=label, values=values)
        nodes[node.id] = node

        if node.rollup_label:
            matched = False
            for k in range(1, len(pending) + 1):
                window = pending[-k:]
                window_sum = np.nansum(np.array([w.values for w in window]), axis=0)
                mask = ~np.isnan(values)
                if not mask.any():
                    break
                diff = np.abs(window_sum[mask] - values[mask])
                tol = ABS_TOL + REL_TOL * np.abs(values[mask])
                if np.all(diff <= tol):
                    node.children = window
                    for w in window:
                        w.parent = node
                    node.rollup_verified = True
                    pending = pending[:-k]
                    matched = True
                    break
            if not matched:
                unverified_rollups.append({"row": r, "label": label})

        pending.append(node)

    roots = [n.id for n in pending]
    debug = {
        "unverified_rollups": unverified_rollups,
        "value_row_count": len(nodes),
        "root_count": len(roots),
        "months_detected": [m.strftime("%b %Y") for m in months],
    }
    return nodes, roots, months, debug


def _node_path(node, cache):
    """Normalized-label path from the tree root down to `node`, e.g.
    ("total room department revenue", "total transient revenue",
    "40010.000 . transient - best available rate"). Used as the cross-year
    join key instead of row position."""
    if node.id in cache:
        return cache[node.id]
    if node.parent is None:
        path = (normalize_label(node.label),)
    else:
        path = _node_path(node.parent, cache) + (normalize_label(node.label),)
    cache[node.id] = path
    return path


def extract_coa_tree_multi(raw_frames):
    """
    Extract + merge COA trees from MULTIPLE yearly workbooks of the SAME
    property into one continuous multi-year tree.

    Real-world charts of accounts DRIFT year to year: GL accounts get
    renumbered, renamed, added, or retired. Confirmed against this
    property's actual 2023/2024/2025 workbooks -- dozens of leaf-account
    differences between any two consecutive years (e.g. "43260.000 . Dues &
    Subscriptions" in 2023 becomes "43250.000 . Decorations" in 2024 at a
    similar-but-not-identical row position). A row-position merge across
    years would silently splice unrelated accounts together. Instead, nodes
    are joined across years by their PATH -- the chain of ancestor labels
    from the tree root down to that node, normalized for matching -- which
    is robust to row reordering and renumbering as long as the wording
    itself stayed stable. Where the wording changed (a genuinely
    renamed/retired/new account), the old and new lines intentionally do
    NOT merge: that is the honest answer, not a bug to paper over. Such
    accounts simply appear as single-year-only nodes, and any rollup whose
    children weren't fully present in every year is flagged in `debug`
    rather than silently treated as reconciled.

    Returns the same (nodes, roots, months, debug) shape as
    extract_coa_tree, except `months` spans all years combined and each
    node's `.values` is length len(months) with NaN for any month its path
    did not exist in that year's workbook.
    """
    if not raw_frames:
        raise ValueError("No workbook frames provided.")

    per_year = [extract_coa_tree(f) for f in raw_frames]
    # Sort years chronologically by their first month, regardless of upload order.
    per_year.sort(key=lambda t: t[2][0] if t[2] else pd.Timestamp.max)

    all_months = []
    for _, _, months, _ in per_year:
        all_months.extend(months)
    all_months = sorted(set(all_months))
    month_index = {m: i for i, m in enumerate(all_months)}
    n_total = len(all_months)

    path_label = {}        # path -> display label (latest year that had it wins)
    path_values = {}       # path -> np.ndarray length n_total, NaN-initialized
    path_parent = {}       # path -> parent path or None
    path_rollup_label = {}  # path -> bool, label looked like a rollup in ANY year
    path_rollup_verified_years = {}  # path -> count of years it verified as a rollup
    path_years_present = {}  # path -> set of year indices where this path had a row
    cross_year_mismatches = []
    label_collisions = []  # same normalized base path used by >1 unrelated row

    for year_idx, (nodes, roots, months, debug) in enumerate(per_year):
        cache = {}
        # Many real workbooks reuse the exact same wording for unrelated rows
        # in different sections (confirmed: this property's actual export has
        # THREE separate root-level rows literally labeled "Room Department" --
        # one under Operating Revenue, one under Departmental Expenses, one
        # under Total Departmental Income -- plus TWO separate root-level rows
        # both literally labeled "Total Departmental Expenses", one in the
        # Summary block and one in the Detail block, holding DIFFERENT
        # numbers). A pure normalized-label path would silently merge these
        # and overwrite one section's data with another's. Since the section
        # ORDER is stable year to year for the same template (verified: row
        # order of these collisions is identical in 2023/2024/2025), each
        # occurrence is disambiguated by its 1st/2nd/3rd... appearance within
        # the year, in sheet order, and that occurrence index is carried into
        # the join key.
        #
        # Pass 1 (order-independent): resolve every node's occurrence-aware
        # path. Run as its own pass, separate from pass 2 below, because a
        # node's PARENT always has a higher row number (a rollup row always
        # appears after the rows it sums), so a single forward pass cannot
        # both assign this node's occurrence index AND already know its
        # parent's resolved path at the same time.
        occurrence_seen = {}
        node_path_resolved = {}
        for node in nodes.values():
            base_path = _node_path(node, cache)
            occurrence_seen[base_path] = occurrence_seen.get(base_path, 0) + 1
            occ = occurrence_seen[base_path]
            path = base_path if occ == 1 else base_path + (f"__occurrence_{occ}",)
            node_path_resolved[node.id] = path
            if occ > 1:
                label_collisions.append({
                    "label": node.label, "year_index": year_idx, "occurrence": occ,
                    "note": "Same label text used for more than one unrelated row "
                            "in this year's workbook; disambiguated by position.",
                })

        # Pass 2: now that every node in this year has a resolved path
        # (including parents), record values/parent links using lookups only.
        for node in nodes.values():
            path = node_path_resolved[node.id]
            parent_path = node_path_resolved.get(node.parent.id) if node.parent else None

            if path not in path_values:
                path_values[path] = np.full(n_total, np.nan)
                path_years_present[path] = set()
                path_rollup_verified_years[path] = 0

            for local_i, m in enumerate(months):
                path_values[path][month_index[m]] = node.values[local_i]

            path_label[path] = node.label  # later years overwrite -> latest wording wins
            path_years_present[path].add(year_idx)
            path_rollup_label[path] = path_rollup_label.get(path, False) or node.rollup_label
            if node.rollup_verified:
                path_rollup_verified_years[path] += 1

            if path in path_parent and path_parent[path] != parent_path:
                cross_year_mismatches.append({
                    "path": " > ".join(path), "year_index": year_idx,
                    "note": "Parent changed across years for this label path.",
                })
            path_parent[path] = parent_path

    # Build merged CoaNode objects, id = the path tuple itself (stable,
    # content-addressed -- unlike a row number, it means the same thing
    # across separate extraction runs).
    merged_nodes = {}
    for path in path_values:
        merged_nodes[path] = CoaNode(node_id=path, row=None, label=path_label[path],
                                      values=path_values[path])
    for path, node in merged_nodes.items():
        parent_path = path_parent.get(path)
        if parent_path is not None and parent_path in merged_nodes:
            node.parent = merged_nodes[parent_path]
            merged_nodes[parent_path].children.append(node)
        # A path "verifies" as a rollup for the merged tree only if it had
        # children attached AND verified in at least one underlying year;
        # partial-coverage rollups (children present in some years only)
        # are still wired up, but flagged below for transparency.
    for path, node in merged_nodes.items():
        node.rollup_label = path_rollup_label.get(path, False)
        node.rollup_verified = bool(node.children) and path_rollup_verified_years.get(path, 0) > 0

    incomplete_rollups = []
    n_years = len(per_year)
    for path, node in merged_nodes.items():
        if node.children:
            years_per_child = [path_years_present[c.id] for c in node.children]
            if any(len(yrs) < n_years for yrs in years_per_child):
                incomplete_rollups.append({
                    "label": node.label,
                    "path": " > ".join(path),
                    "note": "Not every child of this rollup was present in every year "
                            "(account renamed/added/retired across years); bottom-up "
                            "sums for the affected years will understate the true total.",
                })

    roots = [p for p, n in merged_nodes.items() if n.parent is None]
    debug = {
        "year_count": n_years,
        "year_month_ranges": [[m.strftime("%b %Y") for m in months] for _, _, months, _ in per_year],
        "months_detected": [m.strftime("%b %Y") for m in all_months],
        "value_row_count": len(merged_nodes),
        "root_count": len(roots),
        "cross_year_parent_mismatches": cross_year_mismatches,
        "incomplete_rollups": incomplete_rollups,
        "label_collisions": label_collisions,
    }
    return merged_nodes, roots, all_months, debug
