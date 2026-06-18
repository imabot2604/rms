"""Orchestration, verification, and summary for the top-down Prophet POC.

Pipeline:
    1. Load + standardize monthly P&L data.
    2. Run data-quality checks (occupancy recompute + flags); split clean rows.
    3. Forecast top-level drivers with Prophet.
    4. Allocate TotalOperatingRevenue down the COA hierarchy via smoothed
       historical shares (reconciled at each level).
    5. Verify accounting consistency and print a summary.

This module has no notebook dependencies and can be run as a script:
    python -m backend.services.topdown.reconcile_poc --path data/pnl.csv
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import allocation, data_loader, hierarchy, prophet_models, quality

logger = logging.getLogger(__name__)

TOLERANCE = 1e-6


def run_poc(
    df: Optional[pd.DataFrame] = None,
    path: Optional[str] = None,
    periods: int = 12,
    window: int = 12,
) -> dict:
    """Run the end-to-end POC and return all artifacts in a dict.

    Provide either an in-memory frame (``df``) or a file ``path``.
    Returns keys:
        canonical, flagged, top_forecast, dept_forecast, room_subline_forecast,
        verification
    """
    if df is None and path is None:
        raise ValueError("Provide either df or path.")

    canonical = data_loader.load_from_frame(df) if df is not None else data_loader.load_from_path(path)

    # Data quality: keep full flagged frame for reporting, clean frame for training.
    flagged = quality.flag_quality(canonical)
    train_df = quality.training_frame(canonical)

    # Top-level Prophet forecasts.
    top_forecast = prophet_models.forecast_top_level(train_df, periods=periods)

    # Level 1 -> Level 2: allocate Total Operating Revenue to departments.
    dept_forecast = allocation.allocate_recursive(
        top_forecast, train_df, parent="TotalOperatingRevenue", window=window
    )

    # Example single-level further allocation: Rooms -> Transient/Group/Other.
    room_subline_forecast = dept_forecast[
        dept_forecast["parent"] == "RoomDeptRevenue"
    ].copy()

    verification = verify(top_forecast, dept_forecast, train_df)

    return {
        "canonical": canonical,
        "flagged": flagged,
        "top_forecast": top_forecast,
        "dept_forecast": dept_forecast,
        "room_subline_forecast": room_subline_forecast,
        "verification": verification,
    }


def verify(
    top_forecast: pd.DataFrame, dept_forecast: pd.DataFrame, train_df: pd.DataFrame
) -> pd.DataFrame:
    """Verify reconciliation at each parent level. Returns max abs error per parent.

    Checks every parent that has allocated children in ``dept_forecast`` plus
    the top revenue node.
    """
    combined = pd.concat([top_forecast, dept_forecast], ignore_index=True, sort=False)
    parents = sorted(
        p for p in dept_forecast["parent"].dropna().unique()
    )
    rows = []
    for parent in parents:
        err = allocation.reconciliation_error(combined, combined, parent)
        max_abs = float(err["error"].abs().max()) if not err.empty else 0.0
        rows.append(
            {
                "parent": parent,
                "max_abs_error": max_abs,
                "reconciled": max_abs <= TOLERANCE,
            }
        )
    return pd.DataFrame(rows)


def summarize(result: dict) -> None:
    """Print top-level forecasts, allocated department forecasts, and recon error."""
    top = result["top_forecast"]
    dept = result["dept_forecast"]
    ver = result["verification"]

    print("=" * 70)
    print("TOP-LEVEL FORECASTS (Prophet)")
    print("=" * 70)
    pivot_top = top.pivot_table(index="date", columns="node", values="forecast")
    print(pivot_top.round(2).to_string())

    print("\n" + "=" * 70)
    print("ALLOCATED DEPARTMENT FORECASTS (children of TotalOperatingRevenue)")
    print("=" * 70)
    depts = ["RoomDeptRevenue", "FBDeptRevenue", "OtherOperatingRevenue", "MiscIncome"]
    dpt = dept[dept["node"].isin(depts)]
    pivot_dept = dpt.pivot_table(index="date", columns="node", values="forecast")
    print(pivot_dept.round(2).to_string())

    print("\n" + "=" * 70)
    print("EXAMPLE FURTHER ALLOCATION: RoomDeptRevenue -> sub-lines")
    print("=" * 70)
    subs = result["room_subline_forecast"]
    if not subs.empty:
        pivot_sub = subs.pivot_table(index="date", columns="node", values="forecast")
        print(pivot_sub.round(2).to_string())
    else:
        print("(no room sub-line columns available in source data)")

    print("\n" + "=" * 70)
    print("RECONCILIATION ERROR (should be ~0 at every level)")
    print("=" * 70)
    print(ver.to_string(index=False))

    flagged = result["flagged"]
    bad = flagged.loc[~flagged["dq_ok"]] if "dq_ok" in flagged.columns else flagged.iloc[0:0]
    print("\nData-quality flagged months (excluded from training):", len(bad))


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Top-down Prophet reconciliation POC")
    parser.add_argument("--path", required=True, help="Path to monthly P&L CSV/Excel")
    parser.add_argument("--periods", type=int, default=12, help="Forecast horizon (months)")
    parser.add_argument("--window", type=int, default=12, help="Share-smoothing window (months)")
    args = parser.parse_args()

    result = run_poc(path=args.path, periods=args.periods, window=args.window)
    summarize(result)


if __name__ == "__main__":
    _main()
