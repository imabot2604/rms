"""Data-quality and occupancy constraint checks for the top-down POC.

Recomputes monthly Occupancy% from Rooms Sold / Rooms Available, enforces the
0%-100% range, and flags impossible months. Flagged months are excluded from
model training but retained for reporting.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def recompute_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute Occupancy_Pct = Rooms_Sold / Rooms_Available (0..1 scale).

    Standardizes Rooms_Available to a positive denominator and computes a fresh
    occupancy column ``Occupancy_Recomputed``. The original column is preserved.
    """
    df = df.copy()
    if "Rooms_Available" in df.columns and "Rooms_Sold" in df.columns:
        avail = pd.to_numeric(df["Rooms_Available"], errors="coerce")
        sold = pd.to_numeric(df["Rooms_Sold"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = np.where(avail > 0, sold / avail, np.nan)
        df["Occupancy_Recomputed"] = occ
    else:
        # Fall back to any existing occupancy column.
        df["Occupancy_Recomputed"] = pd.to_numeric(
            df.get("Occupancy_Pct", pd.Series(np.nan, index=df.index)), errors="coerce"
        )
    return df


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean ``dq_ok`` column and a ``dq_reason`` column.

    A month is flagged (dq_ok == False) when:
        - Rooms_Available <= 0
        - Rooms_Sold < 0
        - Occupancy_Recomputed > 1 (i.e. > 100%) or < 0
    """
    df = recompute_occupancy(df)
    reasons = pd.Series("", index=df.index, dtype=object)

    avail = pd.to_numeric(df.get("Rooms_Available"), errors="coerce")
    sold = pd.to_numeric(df.get("Rooms_Sold"), errors="coerce")
    occ = pd.to_numeric(df.get("Occupancy_Recomputed"), errors="coerce")

    if avail is not None:
        reasons = reasons.mask(avail <= 0, reasons + "rooms_available<=0;")
    if sold is not None:
        reasons = reasons.mask(sold < 0, reasons + "rooms_sold<0;")
    reasons = reasons.mask(occ > 1.0, reasons + "occupancy>100%;")
    reasons = reasons.mask(occ < 0.0, reasons + "occupancy<0%;")

    df["dq_reason"] = reasons
    df["dq_ok"] = reasons == ""

    flagged = df.loc[~df["dq_ok"]]
    if not flagged.empty:
        for _, row in flagged.iterrows():
            logger.warning(
                "Data-quality flag %s: %s",
                row["Date"].date() if pd.notna(row.get("Date")) else "?",
                row["dq_reason"],
            )
    return df


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the clean (dq_ok) rows for model training.

    The full frame (including flagged rows) should be kept separately for
    reporting.
    """
    flagged = flag_quality(df)
    clean = flagged.loc[flagged["dq_ok"]].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No data-quality-clean months available for training.")
    return clean
