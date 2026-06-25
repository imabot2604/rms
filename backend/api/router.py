from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.data_processing import process_uploaded_files, generate_otb_pace
from services.models import run_ensemble, forecast_excel_months
from services.excel_timeline import coverage_frame
from services.reconciliation import reconcile_topdown, reconciliation_error, COA_HIERARCHY
from services.pnl_pipeline import build_pnl_forecast
import io
import logging
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

api_router = APIRouter()


# In-memory store for simulation purposes
class Store:
    master_df = None
    raw_workbooks = []  # list of (filename, bytes) for row-label extraction


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_float(val):
    """Safely convert a value to a JSON-compatible float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


@api_router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Capture raw bytes for the row-label (P&L) extraction layer before the
        # legacy processor consumes the file streams.
        Store.raw_workbooks = []
        for f in files:
            data = await f.read()
            Store.raw_workbooks.append((getattr(f, "filename", "") or "", data))
            f.file.seek(0)

        # Process and concatenate files
        df = process_uploaded_files(files)

        # Generate synthetic OTB and Comp-Set
        df = generate_otb_pace(df)

        # Save to memory
        Store.master_df = df

        # Convert to records for the frontend
        records = df.copy()

        # Format explicitly for the frontend schema
        normalized = []
        for i, row in records.iterrows():
            date_val = row.get('Date')
            if date_val is None or (hasattr(date_val, 'isoformat') is False):
                continue

            normalized.append({
                "id": f"row-{i}",
                "date": date_val.isoformat(),
                "metrics": {
                    "roomsAvailable": _safe_float(row.get('Rooms_Available')),
                    "roomsSold": _safe_float(row.get('Rooms_Sold')),
                    "occupancy": _safe_float(row.get('Occupancy_Pct')),
                    "adr": _safe_float(row.get('ADR')),
                    "revpar": _safe_float(row.get('RevPAR')),
                    "roomRevenue": _safe_float(row.get('Room_Revenue')),
                    "fbRevenue": _safe_float(row.get('FB_Revenue')),
                    "gop": _safe_float(row.get('GOP')),
                    "noi": _safe_float(row.get('NOI')),
                    "totalRevenue": _safe_float(row.get('Total_Revenue')),
                },
                "features": {
                    "trend": _safe_float(row.get('Trend')),
                    "seasonality": _safe_float(row.get('Seasonality')),
                    "adr_lag": _safe_float(row.get('ADR_Lag')),
                    "comp_index": _safe_float(row.get('Comp_Index')),
                    "otb_pace_index": _safe_float(row.get('OTB_Pace_Index')),
                }
            })

        return {"status": "success", "count": len(normalized), "data": normalized}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@api_router.get("/status")
async def get_status():
    if Store.master_df is not None:
        cols = [c for c in Store.master_df.columns if not c.startswith('_')]
        return {
            "loaded": True,
            "rows": len(Store.master_df),
            "columns": cols
        }
    return {"loaded": False}


@api_router.get("/forecast")
async def get_forecast(horizon: int = 6):
    if Store.master_df is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet. Please upload hotel data first.")

    if horizon < 1 or horizon > 36:
        raise HTTPException(status_code=400, detail="Horizon must be between 1 and 36 months.")

    try:
        results = run_ensemble(Store.master_df, periods=horizon)
        return {"status": "success", "horizon": horizon, "forecast": results}
    except Exception as e:
        logger.exception(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


# Nodes (COA accounts) we forecast on the Excel-bound timeline. Occupancy is the
# primary demand driver; the remaining nodes participate in COA reconciliation.
EXCEL_FORECAST_NODES = [
    "Occupancy_Pct", "ADR", "RevPAR",
    "Rooms_Sold", "Rooms_Available",
    "Room_Revenue", "FB_Revenue", "Other_Revenue", "Total_Revenue",
    "Total_UOE", "GOP", "NOI",
]


@api_router.get("/forecast_excel")
async def get_forecast_excel():
    """
    Forecast ONLY for the months represented in the uploaded Excel.

    Behavior:
      * The forecast horizon is bounded by the Excel's own first/last month.
        No extra future months or unrelated horizons are created.
      * Missing months inside the expected range are included, zero-filled, and
        flagged with a strong MISSING_MONTH reason.
      * Data-quality checks (occupancy recompute, impossible occupancy, negative
        sold rooms, invalid available rooms) are applied and preserved.
      * Per node, the lowest-MAPE model is chosen via holdout validation.
      * Totals are reconciled top-down across the COA hierarchy.

    Output rows contain: month, node, actual, forecast, lower, upper, dq_flag,
    dq_reason, is_missing_filled.
    """
    if Store.master_df is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet. Please upload hotel data first.")

    df = Store.master_df

    try:
        node_forecast_arrays = {}
        per_node_tables = {}
        timeline_info = None
        sequence_info = None
        warnings = []
        shared_index = None  # the single month index every node aligns to

        for node in EXCEL_FORECAST_NODES:
            if node not in df.columns or df[node].isna().all():
                warnings.append({
                    "node": node,
                    "reason": "Not present or not derivable from the uploaded file.",
                })
                continue

            coverage, timeline, sequence = coverage_frame(df, value_col=node)
            timeline_info = timeline_info or timeline
            sequence_info = sequence_info or sequence

            # Establish/validate a single shared month index across all nodes so
            # reconciliation never mixes inconsistent in-sample regions.
            node_index = list(coverage["label"])
            if shared_index is None:
                shared_index = node_index
            elif node_index != shared_index:
                warnings.append({
                    "node": node,
                    "reason": "Month index misaligned with other nodes; skipped from reconciliation.",
                })
                continue

            # Observed (non-missing) actuals in month order for model selection.
            observed = coverage.loc[~coverage["is_missing_filled"], "actual"].to_numpy(dtype=float)

            node_table = forecast_excel_months(coverage, observed, value_col=node)
            per_node_tables[node] = node_table
            node_forecast_arrays[node] = node_table["forecast"].to_numpy(dtype=float)

        if not per_node_tables:
            raise ValueError("No forecastable numeric nodes found in the uploaded data.")

        # Top-down COA reconciliation on the SHARED, aligned month index.
        reconciled = reconcile_topdown(
            node_forecast_arrays, COA_HIERARCHY, month_index=shared_index
        )
        recon_errors = reconciliation_error(reconciled, COA_HIERARCHY)

        rows = []
        diagnostics = {"selected_model": {}, "holdout_mape": {}}
        for node, table in per_node_tables.items():
            recon_vals = reconciled.get(node)
            # Surface per-node model + MAPE in the diagnostic summary.
            sel_models = [m for m in table["selected_model"].tolist() if m != "none(missing)"]
            diagnostics["selected_model"][node] = sel_models[0] if sel_models else "none"
            diagnostics["holdout_mape"][node] = _safe_float(table["holdout_mape"].iloc[0])
            for i in range(len(table)):
                fc = float(recon_vals[i]) if recon_vals is not None else float(table["forecast"].iloc[i])
                rows.append({
                    "month": table["label"].iloc[i],
                    "node": node,
                    "actual": _safe_float(table["actual"].iloc[i]),
                    "forecast": _safe_float(fc),
                    "lower": _safe_float(table["lower"].iloc[i]),
                    "upper": _safe_float(table["upper"].iloc[i]),
                    "dq_flag": table["dq_flag"].iloc[i],
                    "dq_reason": table["dq_reason"].iloc[i],
                    "is_missing_filled": bool(table["is_missing_filled"].iloc[i]),
                    "selected_model": table["selected_model"].iloc[i],
                    "holdout_mape": _safe_float(table["holdout_mape"].iloc[i]),
                })

        return {
            "status": "success",
            "timeline": {
                "start": timeline_info["start"].isoformat(),
                "end": timeline_info["end"].isoformat(),
                "present_labels": timeline_info["labels"],
                "expected_labels": sequence_info["labels"],
                "missing_labels": [m.strftime("%b %Y") for m in sequence_info["missing_months"]],
            },
            "diagnostics": {
                "selected_model": diagnostics["selected_model"],
                "holdout_mape": diagnostics["holdout_mape"],
                "max_reconciliation_error": _safe_float(recon_errors.get("max")),
                "reconciliation_error_by_parent": {
                    k: _safe_float(v) for k, v in recon_errors.items() if k != "max"
                },
            },
            "warnings": warnings,
            "rows": rows,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Excel-bound forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@api_router.get("/forecast_pnl")
async def get_forecast_pnl(horizon: int = 12):
    """
    Source-backed Fairfield/Marriott P&L forecast using ROW-LABEL extraction.

    Historical months are returned as untouched actuals; only months strictly
    after the last actual are forecast (SeasonalNaive(12) for seasonal revenue).
    Future parents are recomputed bottom-up from children. Operational KPIs that
    are absent in the source are returned as null with missing_source_data.

    Each row carries lineage: value, source_type, source_row_label,
    forecast_model, confidence_flag, validation_notes.
    """
    if not Store.raw_workbooks:
        raise HTTPException(status_code=400,
                            detail="No workbook uploaded yet. Please upload a P&L file first.")
    if horizon < 1 or horizon > 36:
        raise HTTPException(status_code=400, detail="Horizon must be between 1 and 36 months.")

    try:
        # Read the first workbook as a header-less wide frame.
        filename, data = Store.raw_workbooks[0]
        if filename.lower().endswith((".csv", ".txt")):
            try:
                df_raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str)
                if df_raw.shape[1] <= 1:
                    df_raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str, sep="\t")
            except Exception:
                df_raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str, sep="\t")
        else:
            df_raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)

        result = build_pnl_forecast(df_raw, horizon=horizon)
        return {"status": "success", "horizon": horizon, **result}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"P&L forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"P&L forecasting error: {str(e)}")
