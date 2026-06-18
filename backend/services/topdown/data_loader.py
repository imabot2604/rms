"""Data loading for the top-down reconciliation POC.

This module loads the monthly P&L / income-statement data into a single
canonical wide DataFrame indexed by month, then exposes a long-form node table
that aligns the raw columns with the COA hierarchy.

It reuses the existing ingestion pipeline
(``backend/services/data_processing.process_uploaded_files``) when uploaded
files are provided, and otherwise loads from a CSV/Excel path. For nodes whose
source column is missing, values are derived where the accounting identity
allows it (e.g. TotalDepartmentalExpenses = TotalOperatingRevenue - GOP).

The canonical schema follows backend/services/data_processing.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import hierarchy

logger = logging.getLogger(__name__)


def load_from_path(path: str) -> pd.DataFrame:
    """Load a monthly P&L file from disk into the canonical wide frame.

    Supports CSV and Excel. The file is expected to already match (or be
    reshaped into) the canonical schema with a ``Date`` column and one row per
    month. For raw wide-format exports, prefer the upload pipeline.
    """
    if path.lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return _standardize(df)


def load_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize an already-loaded DataFrame (e.g. the in-memory Store frame)."""
    return _standardize(df.copy())


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes, sort by month, and derive missing accounting lines."""
    if "Date" not in df.columns:
        raise ValueError("Input data must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Coerce every known source column to numeric.
    for node in hierarchy.NODES:
        col = hierarchy.source_of(node)
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _derive_missing(df)
    return df


def _derive_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill derivable accounting lines from identities when columns are absent.

    Identities used (standard hotel COA):
        TotalDepartmentalExpenses = TotalOperatingRevenue - GOP_before_UOE
        We approximate departmental expenses as Total_Revenue - GOP - Total_UOE
        only when neither departmental expense columns nor the total exist.
        OtherOperatingRevenue / MiscIncome residual:
            residual = Total_Revenue - Room_Revenue - FB_Revenue
    """
    cols = df.columns

    # Department expense total: residual of revenue, UOE and GOP if not present.
    if "Total_Dept_Expenses" not in cols and "Total_Revenue" in cols and "GOP" in cols:
        uoe = df["Total_UOE"] if "Total_UOE" in cols else 0.0
        df["Total_Dept_Expenses"] = df["Total_Revenue"] - df["GOP"] - uoe

    # Other operating revenue + misc income residual split (70/30) when absent.
    if "Total_Revenue" in cols and "Room_Revenue" in cols and "FB_Revenue" in cols:
        residual = df["Total_Revenue"] - df["Room_Revenue"].fillna(0) - df["FB_Revenue"].fillna(0)
        residual = residual.clip(lower=0)
        if "Other_Revenue" not in cols:
            df["Other_Revenue"] = residual * 0.7
        if "Misc_Income" not in cols:
            df["Misc_Income"] = residual * 0.3

    return df


def to_node_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form frame: one row per (Date, node) with the actual value.

    Only nodes whose source column exists (or was derived) are included.
    """
    records = []
    for node in hierarchy.NODES:
        col = hierarchy.source_of(node)
        if col and col in df.columns:
            sub = df[["Date", col]].rename(columns={col: "actual"})
            sub["node"] = node
            records.append(sub[["Date", "node", "actual"]])
    if not records:
        raise ValueError("No COA nodes could be mapped to the input columns.")
    return pd.concat(records, ignore_index=True).sort_values(["node", "Date"]).reset_index(drop=True)


def available_nodes(df: pd.DataFrame) -> list[str]:
    """List COA nodes that have data in the loaded frame."""
    return [
        node
        for node in hierarchy.NODES
        if hierarchy.source_of(node) and hierarchy.source_of(node) in df.columns
        and not df[hierarchy.source_of(node)].isna().all()
    ]
