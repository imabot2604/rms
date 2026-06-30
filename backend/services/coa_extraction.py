"""
Full Chart-of-Accounts (COA) extraction for Fairfield/Marriott monthly P&L
workbooks.

Where ``pnl_extraction`` is intentionally narrow -- it maps only the canonical
headline metrics (Room_Revenue, FB_Revenue, Total_Revenue, GOP, NOI, ...) --
this module extracts EVERY non-empty account row in the workbook's label
column and returns it as a long, monthly time series.

The output is the raw COA grid that downstream layers (forecasting, drill-down
reporting, variance analysis) iterate over. Lineage is preserved: every row
carries the original label, its normalized form, and the section/parent
heading it appears under (e.g. ``Room Department``, ``Undistributed Operating
Expenses``, ``Fixed Charges``).

Design rules:
  * Reuse the month-column detection from ``pnl_extraction`` so COA and
    headline pipelines share the exact same timeline.
  * Never invent values. Cells that are not parseable as numbers become NaN.
  * Heading-only rows (no numeric values in any month column) are kept as
    structural ``section`` rows with ``is_section=True`` and used only to
    classify subsequent leaf rows.
  * The original label text is preserved verbatim; ``normalized_label`` is for
    matching/dedup only.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from services.pnl_extraction import (
    _find_label_and_month_columns,
    normalize_label,
    CANONICAL_ROW_MAP,
    _match_canonical,
)

logger = logging.getLogger(__name__)


# Heuristics for recognising section/parent headings rather than leaf accounts.
_SECTION_HINTS = (
    "department", "departments", "revenue", "expenses",
    "undistributed", "fixed charges", "operating", "income", "statistics",
)


def _looks_like_section(label: str, row_vals) -> bool:
    """Heading rows have no numeric data for the month columns."""
    has_any_numeric = any(pd.notna(v) for v in row_vals.values())
    if has_any_numeric:
        return False
    norm = normalize_label(label)
    if not norm:
        return False
    # Pure heading lines often end with one of the structural hints.
    return any(hint in norm for hint in _SECTION_HINTS) or label.isupper()


def extract_coa_rows(df_raw: pd.DataFrame):
    """
    Extract every COA account row from a single header-less workbook frame.

    Returns:
        (long_df, debug_report)

        long_df columns: Date, account, original_label, normalized_label,
                         section, canonical_metric, is_section, value
        debug_report:    {"months_detected", "section_count", "leaf_count",
                          "canonical_hits"}
    """
    from services.data_processing import _clean_cell

    header_idx, label_col, month_col_map = _find_label_and_month_columns(df_raw)
    if header_idx is None or not month_col_map:
        raise ValueError(
            "Could not locate a monthly COA header row with month columns."
        )

    months = [month_col_map[pos] for pos in sorted(month_col_map)]
    month_positions = sorted(month_col_map)

    records = []
    current_section = None
    section_count = 0
    leaf_count = 0
    canonical_hits = 0

    start_row = max(header_idx + 1, 7)
    for r in range(start_row, len(df_raw)):
        raw_label = df_raw.iat[r, label_col]
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            continue
        original_label = str(raw_label).strip()
        norm = normalize_label(original_label)
        if not norm:
            continue

        row_vals = {}
        for pos in month_positions:
            cell = (_clean_cell(df_raw.iat[r, pos])
                    if pos < df_raw.shape[1] else np.nan)
            row_vals[month_col_map[pos]] = pd.to_numeric(
                pd.Series([cell]), errors="coerce"
            ).iloc[0]

        if _looks_like_section(original_label, row_vals):
            current_section = original_label
            section_count += 1
            # Emit one structural row so downstream consumers can render
            # the COA tree without re-parsing.
            records.append({
                "Date": months[0],
                "account": original_label,
                "original_label": original_label,
                "normalized_label": norm,
                "section": current_section,
                "canonical_metric": None,
                "is_section": True,
                "value": np.nan,
            })
            continue

        metric, _ = _match_canonical(norm)
        if metric is not None:
            canonical_hits += 1

        leaf_count += 1
        for m in months:
            records.append({
                "Date": m,
                "account": original_label,
                "original_label": original_label,
                "normalized_label": norm,
                "section": current_section,
                "canonical_metric": metric,
                "is_section": False,
                "value": row_vals.get(m, np.nan),
            })

    long_df = pd.DataFrame.from_records(records)
    if not long_df.empty:
        long_df["Date"] = pd.to_datetime(long_df["Date"])
        long_df = long_df.sort_values(["account", "Date"]).reset_index(drop=True)

    debug_report = {
        "months_detected": [m.strftime("%b %Y") for m in months],
        "section_count": section_count,
        "leaf_count": leaf_count,
        "canonical_hits": canonical_hits,
        "total_accounts": int(long_df["account"].nunique()) if not long_df.empty else 0,
    }
    return long_df, debug_report


def extract_coa_multi(raw_frames):
    """
    Extract and concatenate the full COA across multiple yearly workbooks.

    Months are deduplicated (first occurrence per ``account``+``Date`` wins).
    """
    if not raw_frames:
        raise ValueError("No workbook frames provided.")
    pieces = []
    sections = 0
    leaves = 0
    canonical = 0
    months_seen = []
    for frame in raw_frames:
        long_df, dbg = extract_coa_rows(frame)
        pieces.append(long_df)
        sections += dbg["section_count"]
        leaves += dbg["leaf_count"]
        canonical += dbg["canonical_hits"]
        months_seen.extend(dbg["months_detected"])

    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not combined.empty:
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = (combined
                    .drop_duplicates(subset=["account", "Date"], keep="first")
                    .sort_values(["account", "Date"])
                    .reset_index(drop=True))

    debug_report = {
        "section_count": sections,
        "leaf_count": leaves,
        "canonical_hits": canonical,
        "months_detected": sorted(set(months_seen)),
        "total_accounts": int(combined["account"].nunique()) if not combined.empty else 0,
    }
    return combined, debug_report


def coa_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a COA long frame to wide: index=Date, columns=account, values=value.
    Section rows are excluded.
    """
    if long_df.empty:
        return pd.DataFrame()
    leaves = long_df[~long_df["is_section"].astype(bool)]
    wide = (leaves
            .pivot_table(index="Date", columns="account",
                         values="value", aggfunc="first")
            .sort_index())
    wide.index.name = "Date"
    return wide
