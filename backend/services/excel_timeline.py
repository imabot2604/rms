"""
Excel-bound timeline + data-quality utilities for the RMS forecasting pipeline.

This module is ADDITIVE. It does not modify or replace the existing Prophet
top-down POC (services.models.run_ensemble). It provides the building blocks
for the "Excel-only month" behavior:

  * detect_excel_timeline:  read the exact monthly range and month labels
                            present in the uploaded workbook (no extra/future
                            months are ever invented).
  * build_expected_sequence: build the continuous monthly index between the
                            first and last month found in the Excel. Months
                            inside that range that are absent from the file are
                            treated as MISSING (to be zero-filled + flagged).
  * apply_quality_checks:   recompute occupancy from Rooms Sold / Rooms
                            Available, flag impossible occupancy (>100%, <0%),
                            negative rooms sold, and invalid rooms available.
  * coverage_frame:         join actuals onto the expected sequence and emit
                            zero-filled, strongly-flagged rows for missing
                            months.

All DQ flags/reasons are preserved so downstream reporting can surface them.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Strong, machine-detectable flag markers used across the pipeline.
DQ_OK = "OK"
DQ_MISSING_MONTH = "MISSING_MONTH"
DQ_IMPOSSIBLE_OCC = "IMPOSSIBLE_OCCUPANCY"
DQ_NEG_SOLD = "NEGATIVE_ROOMS_SOLD"
DQ_INVALID_AVAIL = "INVALID_ROOMS_AVAILABLE"
DQ_OCC_RECOMPUTED = "OCCUPANCY_RECOMPUTED"


def detect_excel_timeline(df):
    """
    Detect the exact monthly range and labels present in the uploaded data.

    Returns a dict describing only what the Excel actually contains. We never
    extend the range beyond the months represented in the file.
    """
    if "Date" not in df.columns:
        raise ValueError("Cannot detect Excel timeline: no 'Date' column present.")

    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot detect Excel timeline: no valid dates found.")

    # Normalize every observed date to its month start (MS) so that day-of-month
    # noise in the source file does not create spurious months.
    present_months = sorted({d.replace(day=1).normalize() for d in dates})

    return {
        "start": present_months[0],
        "end": present_months[-1],
        "present_months": present_months,
        "labels": [m.strftime("%b %Y") for m in present_months],
        "count": len(present_months),
    }


def build_expected_sequence(timeline):
    """
    Build the continuous monthly sequence between the first and last Excel month.

    The horizon is strictly bounded by the uploaded file: it starts at the first
    month in the workbook and ends at the last. No unrelated future horizons are
    produced. Months inside this range that the Excel omitted are returned in
    ``missing_months`` so they can be zero-filled and flagged.
    """
    start = timeline["start"]
    end = timeline["end"]
    expected = pd.date_range(start=start, end=end, freq="MS")

    present = set(timeline["present_months"])
    missing = [m for m in expected if m not in present]

    return {
        "expected_months": list(expected),
        "missing_months": missing,
        "labels": [m.strftime("%b %Y") for m in expected],
    }


def apply_quality_checks(df):
    """
    Recompute occupancy from Rooms Sold / Rooms Available and flag impossible
    values. Returns a copy of ``df`` with 'dq_flag' and 'dq_reason' columns.

    Preserves all existing data-quality semantics:
      * occupancy recomputed from Rooms Sold / Rooms Available when both exist
      * flag occupancy > 100% or < 0%
      * flag negative rooms sold
      * flag invalid (<= 0 or NaN) rooms available when sold rooms are present
    """
    out = df.copy()
    n = len(out)
    flags = [[] for _ in range(n)]
    reasons = [[] for _ in range(n)]

    has_sold = "Rooms_Sold" in out.columns
    has_avail = "Rooms_Available" in out.columns

    # 1. Recompute occupancy from Rooms Sold / Rooms Available where possible.
    if has_sold and has_avail:
        sold = pd.to_numeric(out["Rooms_Sold"], errors="coerce")
        avail = pd.to_numeric(out["Rooms_Available"], errors="coerce")
        recomputed = sold / avail.replace(0, np.nan)
        if "Occupancy_Pct" not in out.columns:
            out["Occupancy_Pct"] = np.nan
        for i in range(n):
            rc = recomputed.iloc[i]
            if pd.notna(rc):
                prev = out["Occupancy_Pct"].iloc[i]
                out.iat[i, out.columns.get_loc("Occupancy_Pct")] = rc
                # Note when the recomputed value materially differs from source.
                if pd.isna(prev) or abs(float(prev) - float(rc)) > 0.005:
                    flags[i].append(DQ_OCC_RECOMPUTED)
                    reasons[i].append(
                        "Occupancy recomputed from Rooms Sold / Rooms Available"
                    )

    # 2. Flag impossible / invalid values.
    for i in range(n):
        if "Occupancy_Pct" in out.columns:
            occ = out["Occupancy_Pct"].iloc[i]
            if pd.notna(occ):
                if occ > 1.0:
                    flags[i].append(DQ_IMPOSSIBLE_OCC)
                    reasons[i].append(f"Occupancy {occ:.2%} exceeds 100%")
                elif occ < 0.0:
                    flags[i].append(DQ_IMPOSSIBLE_OCC)
                    reasons[i].append(f"Occupancy {occ:.2%} is negative")

        if has_sold:
            sold_v = pd.to_numeric(pd.Series([out["Rooms_Sold"].iloc[i]]), errors="coerce").iloc[0]
            if pd.notna(sold_v) and sold_v < 0:
                flags[i].append(DQ_NEG_SOLD)
                reasons[i].append(f"Negative rooms sold ({sold_v})")

        if has_avail:
            avail_v = pd.to_numeric(pd.Series([out["Rooms_Available"].iloc[i]]), errors="coerce").iloc[0]
            sold_present = has_sold and pd.notna(
                pd.to_numeric(pd.Series([out["Rooms_Sold"].iloc[i]]), errors="coerce").iloc[0]
            )
            if sold_present and (pd.isna(avail_v) or avail_v <= 0):
                flags[i].append(DQ_INVALID_AVAIL)
                reasons[i].append("Invalid rooms available (<= 0 or missing)")

    out["dq_flag"] = [";".join(f) if f else DQ_OK for f in flags]
    out["dq_reason"] = ["; ".join(r) if r else "" for r in reasons]
    return out


def coverage_frame(df, value_col="Occupancy_Pct"):
    """
    Join the actuals onto the expected Excel month sequence.

    Every expected month appears exactly once. Months that are missing from the
    uploaded Excel are emitted as zero-filled rows with a strong
    ``MISSING_MONTH`` flag and ``is_missing_filled=True`` so reporting can show
    them distinctly. No months outside the Excel range are produced.
    """
    timeline = detect_excel_timeline(df)
    sequence = build_expected_sequence(timeline)

    checked = apply_quality_checks(df)
    checked = checked.copy()
    checked["_month"] = pd.to_datetime(checked["Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    rows = []
    for month in sequence["expected_months"]:
        match = checked[checked["_month"] == month]
        if match.empty:
            # Missing month inside the expected range -> zero-fill + strong flag.
            rows.append({
                "month": month,
                "label": month.strftime("%b %Y"),
                "actual": 0.0,
                "dq_flag": DQ_MISSING_MONTH,
                "dq_reason": "Month absent from uploaded Excel; zero-filled",
                "is_missing_filled": True,
            })
        else:
            r = match.iloc[0]
            actual = r.get(value_col)
            rows.append({
                "month": month,
                "label": month.strftime("%b %Y"),
                "actual": float(actual) if pd.notna(actual) else 0.0,
                "dq_flag": r.get("dq_flag", DQ_OK),
                "dq_reason": r.get("dq_reason", ""),
                "is_missing_filled": False,
            })

    coverage = pd.DataFrame(rows)
    return coverage, timeline, sequence
