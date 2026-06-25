# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## Excel-bound forecasting

The forecasting pipeline is bounded by the uploaded workbook. When a user
uploads an Excel/CSV file, the system uses **only the months present in that
file's timeline**. It never invents or extends extra months beyond what the
file represents and never produces unrelated future horizons.

### Month detection and the expected sequence

1. `backend/services/excel_timeline.detect_excel_timeline` reads the exact
   monthly range and month labels present in the upload (normalized to month
   start, so day-of-month noise does not create spurious months).
2. `build_expected_sequence` builds the continuous monthly index between the
   **first and last** month found in the file. The forecast horizon is exactly
   this range, no more.

### Missing-month zero-fill

If a month inside the expected range is absent from the upload, it is still
included in the output but:

- its `actual`, `forecast`, `lower`, and `upper` values default to **0**,
- it is flagged with the strong reason `MISSING_MONTH`,
- the row carries `is_missing_filled = true` (`isMissingFilled` on the client)
  so reporting can render it distinctly.

### Preserved data-quality checks

`apply_quality_checks` keeps all existing DQ semantics and preserves the flags
in reporting:

- occupancy is **recomputed from Rooms Sold / Rooms Available** when both exist
  (flag `OCCUPANCY_RECOMPUTED`),
- impossible occupancy `> 100%` or `< 0%` is flagged (`IMPOSSIBLE_OCCUPANCY`),
- negative rooms sold is flagged (`NEGATIVE_ROOMS_SOLD`),
- invalid rooms available (`<= 0` or missing while sold rooms exist) is flagged
  (`INVALID_ROOMS_AVAILABLE`).

### COA top-down reconciliation

`backend/services/reconciliation` defines the Chart of Accounts hierarchy and
performs **top-down reconciliation**: parent totals are authoritative and child
node forecasts are scaled proportionally so children sum to their parent for
every month, keeping reported totals consistent.

### Model selection to minimize MAPE

Per node, `backend/services/models.select_best_model` runs **holdout
validation** on the uploaded monthly series (last ~25% of points, min 1 / max 6
as the holdout) across the models already implemented in this repo (Prophet,
ARIMA, ETS, Random Forest, XGBoost when available, and a seasonal-naive
baseline). The lowest-MAPE model per node is used by
`forecast_excel_months` to produce forecast / lower / upper for exactly the
Excel months. The existing Prophet top-down POC (`run_ensemble`) is preserved
and still available via `GET /api/forecast`.

### Output schema

`GET /api/forecast_excel` returns rows with:

`month`, `node` (COA account), `actual`, `forecast`, `lower`, `upper`,
`dq_flag`, `dq_reason`, and `is_missing_filled` (plus `selected_model` and
`holdout_mape` for transparency). The response also includes a `warnings` list
(any expected node that could not be produced is reported, never silently
skipped) and a `diagnostics` block (selected model + holdout MAPE per node and
the maximum reconciliation error).

## Fairfield / USALI handling

These behaviors make the pipeline correct and accounting-consistent for
Fairfield-style USALI wide-format workbooks. All changes are additive; the
existing Prophet top-down POC (`run_ensemble` / `GET /api/forecast`) is
preserved and top-down reconciliation remains the primary flow.

### USALI stat-block parsing

In Fairfield files, KPIs such as Occupancy, ADR, RevPAR, Rooms Sold, and Rooms
Available appear as **row labels** in the statistics block (months are
columns), not as column headers. After the wide-to-long melt and pivot, those
row labels become columns and are mapped into the canonical schema by
`data_processing._map_columns_to_schema`, so they are available downstream for
forecasting and DQ checks. Existing header-based formats remain supported.

### Canonical field naming (NOI and KPIs)

The schema mapping normalizes USALI variants to canonical names end to end.
In particular `Net Income (loss)`, `Net Income loss`, `Net Income`, and
`Net Operating Income` all map to the canonical node **`NOI`**, so the
forecasting pipeline no longer silently skips NOI due to a naming mismatch.

### Derived revenue / UOE nodes

Fairfield total revenue includes revenue lines beyond Rooms and F&B, so
`Room_Revenue + FB_Revenue` does not equal `Total_Revenue` on its own.
`derive_accounting_identities` derives:

- **`Other_Revenue`** = `Total_Revenue - (Room_Revenue + FB_Revenue)`, added as
  an explicit child of `Total_Revenue` so revenue reconciles exactly.
- **`Total_UOE`** (when absent) from `Total_Departmental_Income - GOP`, falling
  back to `Total_Revenue - GOP`, so the GOP hierarchy still reconciles.

Derived columns are built with a single batched `pd.concat` to avoid pandas
DataFrame fragmentation (the wide-ingestion fragmentation warning is gone).

Updated COA hierarchy:

```
Total_Revenue := Room_Revenue + FB_Revenue + Other_Revenue
GOP           := Total_Revenue + Total_UOE   (top-down authoritative parent)
NOI           := GOP
```

### Deterministic date parsing

Month labels like `Jan, 2024` are parsed with explicit formats
(`%b, %Y`, `%b %Y`, ...) before any generic fallback, removing repeated parse
warnings and making ingestion deterministic and faster.

### Near-constant series fallback

`models.is_near_constant` flags near-constant or mostly-zero series
(`std / mean(abs) < threshold`, or a high zero-fraction). For such nodes (for
example a flat Fairfield F&B revenue line) the pipeline does **not** run
XGBoost / Random Forest; it falls back to `SimpleAverage` or `ZeroFill`. The
choice is surfaced in `selected_model`.

### December NOI outlier handling

NOI carries severe December anomalies (depreciation, amortization, year-end
charges). `models.robust_clip_series` applies a median/MAD modified z-score and,
for NOI specifically, clips December months toward the non-December central
tendency. Only the **model input** is adjusted; raw actuals are preserved for
reporting. This materially reduces NOI holdout MAPE.

### Reconciliation alignment guarantees

Top-down reconciliation operates strictly on **aligned forecast arrays** over a
single shared month index. Every node's forecast is produced aligned 1:1 to the
full Excel month index, and `reconcile_topdown` validates alignment and raises
`ReconciliationError` if parent/child month indexes differ, instead of silently
producing wrong numbers. `reconciliation_error` reports the maximum residual
after scaling (near zero when consistent).

### Validation and tests

`backend/tests/test_fairfield_pipeline.py` provides lightweight regression
tests confirming: USALI stat-block parsing of the core KPIs, NOI naming
normalization, near-zero revenue reconciliation error after top-down scaling,
zero-filled + flagged missing months, near-constant F&B falling back to a
simple model, deterministic `Jan, 2024` parsing, and the reconciliation
alignment guardrail. `backend/diagnostics.py` (`diagnostic_summary`) prints the
selected model per node, holdout MAPE, and maximum reconciliation error,
mirroring the train-on-history / validate workflow from the diagnostic report.

## Source-backed P&L pipeline (row-label architecture)

### Root cause

Fairfield Inn / Marriott monthly P&L workbooks are organized by **row labels**
(e.g. `Total Room Department Revenue`, `Total Operating Revenue`,
`Gross Operating Profit`, `Net Income loss`) with **months as columns**. The
legacy detection expected operational KPIs (Occupancy, ADR, RevPAR, Rooms Sold,
Rooms Available) as column headers, which do not exist in these files. The
source-backed pipeline scans the first label column and maps canonical metrics
from row names instead. It is exposed via `GET /api/forecast_pnl` and is
additive (the existing `/forecast` and `/forecast_excel` endpoints are intact).

### Layers

1. **Extraction** (`services/pnl_extraction.py`) -- `extract_pnl_rows()` scans
   the label column, fuzzy-normalizes labels (trim, lowercase, collapse spaces,
   safe punctuation removal) for matching, and maps them via the explicit
   `CANONICAL_ROW_MAP`. The original label is preserved for lineage. Returns a
   monthly dataframe plus a debug report (matched / unmatched / duplicate
   candidate rows and missing requested metrics).
2. **Mapping** -- canonical row map with alias support:
   `Room_Revenue <- Total Room Department Revenue`,
   `FB_Revenue <- Total FB and Minor Operating Dept Revenue`,
   `Total_Revenue <- Total Operating Revenue`,
   `GOP <- Gross Operating Profit`,
   `NOI <- Net Income loss` (canonical name; original label kept in metadata),
   `Total_UOE` derived as `Total_Revenue - GOP` only when no explicit UOE row
   exists.
3. **Forecasting** (`services/pnl_forecasting.py`) -- forecasts ONLY months
   strictly after the last actual. `Room_Revenue` and `Total_Revenue` default
   to **SeasonalNaive(12)** so seasonal series never collapse to a flat line
   (ARIMA is explicitly excluded for these). True near-zero sparse series use
   zero-fill. A per-metric model-selection report records the rationale.
4. **Reconciliation** (`services/pnl_reconciliation.py`) -- historical actuals
   are untouched; future months are reconciled **bottom-up**
   (`Total_Revenue = Room + FB + Other`, `GOP = Total_Revenue - Total_UOE`).
   `NOI` keeps its own source-backed forecast and is **not** forced to equal
   GOP. No blind scale factors across mismatched model ranges.

### Honest missing KPIs

Operational KPIs absent from the source return explicit `null` with status
`missing_source_data` and the reason "not present in source workbook. I could
not verify those KPI labels in the attached files." They are never fabricated
from revenue.

### Metric lineage

Every output row carries: `value`, `source_type` (`actual` / `forecast` /
`derived` / `missing`), `source_row_label`, `forecast_model`,
`confidence_flag`, and `validation_notes`.

### Validations (`services/pnl_pipeline.run_validations`)

- historical actuals are present and unmodified (match source within rounding
  tolerance),
- future-period hierarchy equations reconcile within tolerance,
- flat forecast on a historically seasonal series is flagged,
- impossible KPI values are flagged if/when KPIs become available
  (occupancy outside 0-100%, ADR < 0, RevPAR < 0).

Tests live in `backend/tests/test_pnl_pipeline.py`.
