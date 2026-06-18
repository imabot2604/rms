# Top-down forecasting & reconciliation package

Forecast the hotel P&L across the full chart of accounts (COA) and keep every
node mutually consistent. The package layers several model families and two
reconciliation strategies, all behind a single pipeline entry point.

## Model stack

| Layer | Module | Used for |
|-------|--------|----------|
| **Prophet** | `prophet_models.py` | Major demand/profit lines: `TotalOperatingRevenue`, `GOP`, `NOI` (yearly seasonality only). |
| **ARIMA/ETS** | `arima_models.py` | Smooth expense lines: `PropertyTaxes`, `Insurance`, `ManagementFees`, `UtilitiesTotal` (SARIMAX wrapper). |
| **Tree-based ML** | `ml_models.py` | `ADR` and event-driven revenue (`BanquetRevenue`, `RestaurantLoungeRevenue`) using RandomForest with calendar/trend/lag + optional regressors (`Comp_Index`, `OTB_Pace_Index`, events). |
| **Ensemble** | `ensemble.py` | Inverse-MAPE blend of Prophet/ARIMA/RF on high-value nodes (`RoomDeptRevenue`, `ADR`), scored on a holdout window. |
| **Reconciliation** | `allocation.py`, `hts_reconcile.py` | Make all nodes sum-consistent across the COA tree. |

All base models emit the same long schema: `date, node, forecast, lower, upper`.

## Reconciliation modes

The orchestrator `reconcile_pipeline.py` exposes `--mode`:

- **`topdown_share`** (default) — share-based top-down allocation
  (`allocation.py`). Smoothed trailing-window shares are renormalized to sum to
  1, so children reconcile to their parent **exactly** at every level.
- **`hts_scikit`** — formal reconciliation via
  [`scikit-hts`](https://scikit-hts.readthedocs.io) (`hts_reconcile.py`),
  supporting bottom-up, top-down, middle-out and OLS/MinT. Falls back to the
  unreconciled base forecasts if `scikit-hts` is not installed.
- **`nixtla`** — placeholder for a future Nixtla `HierarchicalForecast`
  integration; currently falls back to `topdown_share`.

## Data-quality gate (non-negotiable)

`quality.py` is applied **before any model trains**:

- Occupancy% recomputed as `Rooms Sold / Rooms Available`, constrained to 0–100%.
- Months with `Rooms Available <= 0`, `Rooms Sold < 0`, or occupancy out of
  range are flagged (`dq_ok=False`, with a `dq_reason`).
- Only `dq_ok == True` rows train models; **all** months remain visible in
  reports and logs for auditing.

## Running

```bash
pip install -r backend/requirements.txt

# Share-based top-down reconciliation
python -m backend.services.topdown.reconcile_pipeline --path data/pnl.csv --mode topdown_share

# Formal scikit-hts reconciliation
python -m backend.services.topdown.reconcile_pipeline --path data/pnl.csv --mode hts_scikit
```

The original Prophet-only POC remains available via
`python -m backend.services.topdown.reconcile_poc --path data/pnl.csv`.

## Notes

- Nodes whose source column is absent from the data are skipped gracefully;
  extend `backend/services/data_processing.py` to emit the L2/L3 expense and
  sub-line columns to activate more of the tree.
- Every model has a deterministic fallback (seasonal-naive / equal-weight) so a
  missing optional library never breaks the pipeline.
