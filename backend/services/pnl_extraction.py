"""
Layer 1: P&L row-label extraction for Fairfield Inn / Marriott monthly P&L
workbooks.

ROOT-CAUSE NOTE
---------------
These workbooks are organized by ROW LABELS (e.g. "Total Room Department
Revenue", "Total Operating Revenue", "Gross Operating Profit", "Net Income
loss") with months as columns. The legacy pipeline expected KPIs such as
Occupancy_Pct / ADR / RevPAR / Rooms Sold / Rooms Available as COLUMN HEADERS,
which do not exist in these files. This module scans the first label column and
maps canonical metrics from the row names instead.

Design goals:
  * Explicit, alias-based canonical row map (no fragile guessing).
  * Fuzzy normalization (trim, lowercase, collapse spaces, safe punctuation
    removal) for matching only -- the ORIGINAL label is preserved for lineage.
  * Operational KPIs that are not present in the source are reported as
    ``missing_source_data`` and are NEVER fabricated from revenue.
  * A debug report listing matched rows, unmatched rows, duplicate candidate
    rows, and any requested metrics that were missing.

The output is a normalized monthly long/▢wide structure plus lineage metadata,
consumed by the forecasting and reconciliation layers.
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Source-of-truth status markers for metric lineage.
SRC_ACTUAL = "actual"
SRC_FORECAST = "forecast"
SRC_DERIVED = "derived"
SRC_MISSING = "missing"
MISSING_SOURCE_DATA = "missing_source_data"

# Canonical metrics that, for this property class (limited-service Fairfield),
# are accounting/revenue lines we expect from row labels.
FINANCIAL_METRICS = [
    "Room_Revenue", "FB_Revenue", "Other_Revenue",
    "Total_Revenue", "Total_UOE", "GOP", "NOI",
]

# Operational KPIs. We attempt row-label matching only; if absent they are
# reported missing_source_data and never inferred from revenue.
OPERATIONAL_KPIS = [
    "Occupancy_Pct", "ADR", "RevPAR", "Rooms_Available", "Rooms_Sold",
]

# Explicit canonical row map. Keys are canonical metric names; values are lists
# of normalized alias phrases. Matching is exact-on-normalized first, then
# containment, so specific aliases should be listed.
CANONICAL_ROW_MAP = {
    "Room_Revenue": [
        "total room department revenue", "total rooms department revenue",
        "room department revenue", "total room revenue", "rooms revenue",
        "room revenue",
    ],
    "FB_Revenue": [
        "total fb and minor operating dept revenue",
        "total f b and minor operating dept revenue",
        "total food and beverage revenue", "food and beverage revenue",
        "f b revenue", "fb revenue",
    ],
    "Other_Revenue": [
        "other operated departments revenue", "minor operated departments",
        "rentals and other income", "miscellaneous income", "other revenue",
        "other income",
    ],
    "Total_Revenue": [
        "total operating revenue", "total hotel revenue", "total revenue",
    ],
    "Total_UOE": [
        # Verified Fairfield label is 'Total Departmental Expenses'.
        "total departmental expenses", "total department expenses",
        "total undistributed operating expenses",
        "total undistributed expenses", "total uoe", "uoe",
    ],
    "GOP": [
        "gross operating profit", "gop",
    ],
    "NOI": [
        # Verified Fairfield label is 'Net Income loss' (no parens, lowercase
        # 'loss'). normalize_label strips ( ) / so all variants collapse to
        # 'net income loss'. Product expects canonical NOI; original label is
        # preserved in lineage.
        "net income loss", "net income (loss)", "net income/loss",
        "net operating income", "net income",
    ],
    # Operational KPIs (may legitimately be absent in Fairfield files).
    "Occupancy_Pct": ["occupancy percent", "occupancy %", "occupancy", "occ %"],
    "ADR": ["average daily rate", "adr"],
    "RevPAR": ["revenue per available room", "revpar"],
    "Rooms_Available": ["rooms available", "room nights available", "available rooms"],
    "Rooms_Sold": ["rooms sold", "room nights sold", "rooms occupied"],
}

_MONTH_TOKENS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def normalize_label(label):
    """
    Fuzzy-normalize a row label for matching ONLY.

    Trims, lowercases, collapses repeated whitespace, and removes punctuation
    that is safe to drop (&, commas, parentheses, slashes, periods). The caller
    keeps the ORIGINAL label for lineage/debugging.
    """
    if label is None:
        return ""
    s = str(label).strip().lower()
    # Strip ='"..."' export wrappers.
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    # Replace ampersand/punctuation with spaces (safe), keep alphanumerics.
    s = re.sub(r"[&/().,:;\"']", " ", s)
    s = re.sub(r"[^a-z0-9%\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _match_canonical(normalized):
    """
    Match a normalized label to a canonical metric.

    Returns (metric, match_kind) where match_kind is 'exact' or 'contains',
    or (None, None) if no alias matches. Exact matches win over containment.
    """
    # Exact normalized match first.
    for metric, aliases in CANONICAL_ROW_MAP.items():
        if normalized in aliases:
            return metric, "exact"
    # Containment fallback (longest alias first for specificity).
    best = None
    best_len = 0
    for metric, aliases in CANONICAL_ROW_MAP.items():
        for alias in aliases:
            if alias and alias in normalized and len(alias) > best_len:
                best = metric
                best_len = len(alias)
    if best is not None:
        return best, "contains"
    return None, None


def _find_label_and_month_columns(df_raw, max_scan=30):
    """
    Identify the header row containing month columns and the first label column.

    Returns (header_row_idx, label_col_idx, month_col_map) where month_col_map
    maps the raw column position -> parsed pd.Timestamp (month start).
    """
    from services.data_processing import _parse_month_series  # reuse parser

    for idx in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[idx]
        month_positions = {}
        for pos, val in enumerate(row):
            if pd.isna(val):
                continue
            sval = str(val).strip().lower()
            if ' - ' in sval or ' to ' in sval or 'total' in sval:
                continue
            if any(tok in sval for tok in _MONTH_TOKENS):
                month_positions[pos] = str(val).strip()
        if len(month_positions) >= 3:
            parsed = _parse_month_series(pd.Series(list(month_positions.values())))
            month_col_map = {}
            for (pos, _), ts in zip(month_positions.items(), parsed):
                if pd.notna(ts):
                    month_col_map[pos] = ts.replace(day=1).normalize()
            # Label column = first non-month, non-empty column on this row.
            label_col_idx = 0
            return idx, label_col_idx, month_col_map
    return None, None, {}


def extract_pnl_rows(df_raw):
    """
    Extract a normalized monthly P&L from a raw wide workbook frame.

    Args:
        df_raw: a header-less DataFrame (dtype=str) as read from the workbook.

    Returns:
        (monthly_df, lineage, debug_report)

        monthly_df: one row per month, columns = canonical metrics that were
                    found (financial). Index is the month (Timestamp).
        lineage:    dict {metric: {"source_row_label": str|None,
                                    "source_type": actual|derived|missing,
                                    "status": str}}.
        debug_report: dict with matched_rows, unmatched_rows,
                    duplicate_candidates, missing_metrics.
    """
    from services.data_processing import _clean_cell

    header_idx, label_col, month_col_map = _find_label_and_month_columns(df_raw)
    if header_idx is None or not month_col_map:
        raise ValueError(
            "Could not locate a monthly P&L header row with month columns."
        )

    months = [month_col_map[pos] for pos in sorted(month_col_map)]
    month_positions = sorted(month_col_map)

    # Accumulate canonical metric -> {month: value} from matched rows.
    metric_values = {}            # metric -> dict(month -> float)
    matched_rows = []             # (original_label, metric, kind)
    unmatched_rows = []           # original labels
    duplicate_candidates = {}     # metric -> [original labels]
    metric_source_label = {}      # metric -> first original label

    for r in range(header_idx + 1, len(df_raw)):
        raw_label = df_raw.iat[r, label_col]
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            continue
        original_label = str(raw_label).strip()
        normalized = normalize_label(original_label)
        if not normalized:
            continue

        metric, kind = _match_canonical(normalized)
        if metric is None:
            unmatched_rows.append(original_label)
            continue

        # Pull the month values for this row.
        row_vals = {}
        for pos in month_positions:
            cell = _clean_cell(df_raw.iat[r, pos]) if pos < df_raw.shape[1] else np.nan
            row_vals[month_col_map[pos]] = pd.to_numeric(
                pd.Series([cell]), errors="coerce"
            ).iloc[0]

        if metric in metric_values:
            # First occurrence wins; record duplicate candidates for debugging.
            duplicate_candidates.setdefault(metric, [metric_source_label[metric]])
            duplicate_candidates[metric].append(original_label)
            continue

        metric_values[metric] = row_vals
        metric_source_label[metric] = original_label
        matched_rows.append({"label": original_label, "metric": metric, "match": kind})

    # Build the monthly dataframe from matched FINANCIAL metrics (+ any matched
    # operational KPIs). Operational KPIs absent here are flagged later.
    data = {"Date": months}
    for metric, vals in metric_values.items():
        data[metric] = [vals.get(m, np.nan) for m in months]
    monthly_df = pd.DataFrame(data)

    # --- Lineage + honest missing handling. ---
    lineage = {}
    for metric in metric_values:
        lineage[metric] = {
            "source_row_label": metric_source_label[metric],
            "source_type": SRC_ACTUAL,
            "status": "matched",
        }

    # Derive Total_UOE only if no explicit UOE row was matched but we have
    # Total_Revenue and GOP. Never guess otherwise.
    if "Total_UOE" not in monthly_df.columns \
            and "Total_Revenue" in monthly_df.columns and "GOP" in monthly_df.columns:
        monthly_df["Total_UOE"] = monthly_df["Total_Revenue"] - monthly_df["GOP"]
        lineage["Total_UOE"] = {
            "source_row_label": None,
            "source_type": SRC_DERIVED,
            "status": "derived: Total_Revenue - GOP",
        }

    # Operational KPIs: present only if matched; otherwise missing_source_data.
    missing_metrics = []
    for kpi in OPERATIONAL_KPIS:
        if kpi not in monthly_df.columns:
            lineage[kpi] = {
                "source_row_label": None,
                "source_type": SRC_MISSING,
                "status": MISSING_SOURCE_DATA,
                "reason": "not present in source workbook. "
                          "I could not verify those KPI labels in the attached files.",
            }
            missing_metrics.append(kpi)

    debug_report = {
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "duplicate_candidates": duplicate_candidates,
        "missing_metrics": missing_metrics,
        "months_detected": [m.strftime("%b %Y") for m in months],
    }

    monthly_df = monthly_df.sort_values("Date").reset_index(drop=True)
    return monthly_df, lineage, debug_report
