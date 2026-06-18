"""Top-down Prophet reconciliation POC package.

Forecast the top of the P&L with Prophet, then allocate forecasts down the
chart-of-accounts (COA) hierarchy using smoothed historical shares so that all
department and sub-account totals stay accounting-consistent.

Modules:
    data_loader     -- load/normalize monthly P&L data into a canonical frame
    hierarchy       -- COA parent-child structure (3 levels)
    quality         -- occupancy recompute + data-quality flags
    prophet_models  -- top-level Prophet forecasts
    allocation      -- smoothed shares + top-down reconciliation
    reconcile_poc   -- orchestration, verification, summary
"""

from .reconcile_poc import run_poc, summarize

__all__ = ["run_poc", "summarize"]
