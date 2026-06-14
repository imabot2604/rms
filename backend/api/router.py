from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.data_processing import process_uploaded_files, generate_otb_pace
from services.models import run_ensemble
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

api_router = APIRouter()


# In-memory store for simulation purposes
class Store:
    master_df = None


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
