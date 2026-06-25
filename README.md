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
`holdout_mape` for transparency).
