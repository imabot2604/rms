"""Multi-mode reconciliation pipeline for the RMS forecasting stack.

Orchestrates the full modeling stack:

    * Prophet  -> top-level demand/profit drivers (TotalOperatingRevenue, GOP, NOI)
    * ARIMA    -> smooth expense lines (taxes, insurance, mgmt fees, utilities)
    * ML (RF)  -> ADR and event-driven revenue nodes
    * Ensemble -> inverse-MAPE blend on selected high-value nodes
    * Reconcile-> share-based top-down OR scikit-hts (selectable via --mode)

Every series passes through ``quality.py`` first: only dq_ok==True rows train
models, while all months (with dq_reason) remain visible for reporting.

Run:
    python -m backend.services.topdown.reconcile_pipeline --path data/pnl.csv --mode topdown_share
    python -m backend.services.topdown.reconcile_pipeline --path data/pnl.csv --mode hts_scikit
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import pandas as pd

from . import (
    allocation,
    arima_models,
    data_loader,
    ensemble,
    hierarchy,
    hts_reconcile,
    ml_models,
    prophet_models,
    quality,
)

logger = logging.getLogger(__name__)

MODES = ("topdown_share", "hts_scikit", "nixtla")


def _prophet_node_fn(canonical_train: pd.DataFrame):
    """Adapter so a node series can be Prophet-forecast for the ensemble."""
    def fn(node_hist: pd.DataFrame, steps: int) -> pd.DataFrame:
        node = node_hist["node"].iloc[0]
        col = hierarchy.source_of(node)
        wide = node_hist.rename(columns={"actual": col})[["Date", col]]
        # Reuse the generic single-series Prophet fit via prophet_models.
        series = wide.dropna().rename(columns={"Date": "ds", col: "y"})
        return prophet_models._fit_one(series, node, steps)
    return fn


def run_pipeline(
    df: Optional[pd.DataFrame] = None,
    path: Optional[str] = None,
    periods: int = 12,
    window: int = 12,
    mode: str = "topdown_share",
) -> dict:
    """Run the full stack and return all artifacts."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if df is None and path is None:
        raise ValueError("Provide either df or path.")

    canonical = (
        data_loader.load_from_frame(df) if df is not None else data_loader.load_from_path(path)
    )

    # Quality gate: flagged frame for reporting, clean frame for training.
    flagged = quality.flag_quality(canonical)
    train_df = quality.training_frame(canonical)
    long_actual = data_loader.to_node_long(train_df)

    consolidated = []

    # 1. Prophet top-level drivers.
    top_forecast = prophet_models.forecast_top_level(train_df, periods=periods)
    consolidated.append(top_forecast)

    # 2. ARIMA cost forecasting step (only nodes present in the data).
    cost_nodes = [n for n in hierarchy.COST_FORECAST_NODES if n in long_actual["node"].unique()]
    cost_forecast = arima_models.forecast_cost_nodes(cost_nodes, long_actual, periods)
    if not cost_forecast.empty:
        consolidated.append(cost_forecast)

    # 3. ML forecasts for ADR / event-driven nodes present in the data.
    ml_frames = []
    for node in hierarchy.ML_NODES:
        if node in long_actual["node"].unique():
            mf = ml_models.forecast_node(long_actual, node, periods)
            if not mf.empty:
                ml_frames.append(mf)
    ml_forecast = pd.concat(ml_frames, ignore_index=True) if ml_frames else pd.DataFrame()
    if not ml_forecast.empty:
        consolidated.append(ml_forecast)

    # 4. Ensemble on key high-value nodes (RoomDeptRevenue if available).
    ensemble_frames = []
    key_nodes = [n for n in ["RoomDeptRevenue", "ADR"] if n in long_actual["node"].unique()]
    base_fns = {
        "prophet": _prophet_node_fn(train_df),
        "arima": lambda h, s: arima_models.forecast_cost_node(h["node"].iloc[0], h, s),
        "rf": lambda h, s: ml_models.forecast_node(h, h["node"].iloc[0], s),
    }
    for node in key_nodes:
        ef = ensemble.ensemble_node(node, long_actual, base_fns, periods)
        if not ef.empty:
            ensemble_frames.append(ef)
    ensemble_forecast = (
        pd.concat(ensemble_frames, ignore_index=True) if ensemble_frames else pd.DataFrame()
    )

    # 5. Reconciliation.
    if mode == "topdown_share":
        reconciled = allocation.allocate_recursive(
            top_forecast, train_df, parent="TotalOperatingRevenue", window=window
        )
    elif mode == "hts_scikit":
        base = pd.concat(consolidated, ignore_index=True)
        reconciled = hts_reconcile.reconcile(
            long_actual, base, periods, method="OLS", root="TotalOperatingRevenue"
        )
    else:  # nixtla (placeholder)
        logger.warning("nixtla mode not yet implemented; falling back to topdown_share.")
        reconciled = allocation.allocate_recursive(
            top_forecast, train_df, parent="TotalOperatingRevenue", window=window
        )

    return {
        "mode": mode,
        "canonical": canonical,
        "flagged": flagged,
        "top_forecast": top_forecast,
        "cost_forecast": cost_forecast,
        "ml_forecast": ml_forecast,
        "ensemble_forecast": ensemble_forecast,
        "reconciled": reconciled,
    }


def summarize(result: dict) -> None:
    """Print a compact summary of every stage."""
    print("=" * 70)
    print(f"PIPELINE MODE: {result['mode']}")
    print("=" * 70)

    def _show(title, frame):
        print(f"\n--- {title} ---")
        if frame is None or frame.empty:
            print("(none)")
            return
        pivot = frame.pivot_table(index="date", columns="node", values="forecast")
        print(pivot.round(2).to_string())

    _show("Prophet top-level", result["top_forecast"])
    _show("ARIMA cost lines", result["cost_forecast"])
    _show("ML (RandomForest) nodes", result["ml_forecast"])
    _show("Ensemble (key nodes)", result["ensemble_forecast"])
    _show(f"Reconciled ({result['mode']})", result["reconciled"])

    flagged = result["flagged"]
    bad = flagged.loc[~flagged["dq_ok"]] if "dq_ok" in flagged.columns else flagged.iloc[0:0]
    print("\nData-quality flagged months (excluded from training):", len(bad))


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="RMS multi-mode reconciliation pipeline")
    parser.add_argument("--path", required=True, help="Path to monthly P&L CSV/Excel")
    parser.add_argument("--periods", type=int, default=12, help="Forecast horizon (months)")
    parser.add_argument("--window", type=int, default=12, help="Share-smoothing window (months)")
    parser.add_argument("--mode", choices=MODES, default="topdown_share", help="Reconciliation mode")
    args = parser.parse_args()

    result = run_pipeline(
        path=args.path, periods=args.periods, window=args.window, mode=args.mode
    )
    summarize(result)


if __name__ == "__main__":
    _main()
