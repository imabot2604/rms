"""COA hierarchy definition for the top-down reconciliation POC.

The hierarchy is a parent-child mapping with three levels:

    Level 1 (top):  TotalOperatingRevenue, TotalDepartmentalExpenses, GOP, NOI,
                    NetIncome
    Level 2:        Department revenue / expense totals
    Level 3:        Major sub-lines of each department revenue

Each node maps to a dict describing its parent, level and the canonical source
column (if it exists directly in the loaded data). Nodes whose ``source`` is
``None`` are derived from their children.

The canonical column names follow the existing project schema
(see backend/services/data_processing.py): Total_Revenue, Room_Revenue,
FB_Revenue, GOP, NOI, Total_UOE, etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# --- Node metadata --------------------------------------------------------
# parent == None marks a root node.
# source == canonical column name in the loaded frame, or None when derived.
NODES: Dict[str, dict] = {
    # ---- Level 1 ----
    "TotalOperatingRevenue": {"parent": None, "level": 1, "source": "Total_Revenue"},
    "TotalDepartmentalExpenses": {"parent": None, "level": 1, "source": "Total_Dept_Expenses"},
    "GOP": {"parent": None, "level": 1, "source": "GOP"},
    "NOI": {"parent": None, "level": 1, "source": "NOI"},
    "NetIncome": {"parent": None, "level": 1, "source": "Net_Income"},

    # ---- Level 2: department revenue (children of TotalOperatingRevenue) ----
    "RoomDeptRevenue": {"parent": "TotalOperatingRevenue", "level": 2, "source": "Room_Revenue"},
    "FBDeptRevenue": {"parent": "TotalOperatingRevenue", "level": 2, "source": "FB_Revenue"},
    "OtherOperatingRevenue": {"parent": "TotalOperatingRevenue", "level": 2, "source": "Other_Revenue"},
    "MiscIncome": {"parent": "TotalOperatingRevenue", "level": 2, "source": "Misc_Income"},

    # ---- Level 2: department expense (children of TotalDepartmentalExpenses) ----
    "RoomDeptExpense": {"parent": "TotalDepartmentalExpenses", "level": 2, "source": "Room_Expense"},
    "FBDeptExpense": {"parent": "TotalDepartmentalExpenses", "level": 2, "source": "FB_Expense"},
    "OtherDeptExpense": {"parent": "TotalDepartmentalExpenses", "level": 2, "source": "Other_Expense"},

    # ---- Level 3: room revenue sub-lines (children of RoomDeptRevenue) ----
    "TotalTransient": {"parent": "RoomDeptRevenue", "level": 3, "source": "Transient_Revenue"},
    "TotalGroup": {"parent": "RoomDeptRevenue", "level": 3, "source": "Group_Revenue"},
    "TotalOtherRoomRevenue": {"parent": "RoomDeptRevenue", "level": 3, "source": "Other_Room_Revenue"},

    # ---- Level 3: F&B revenue sub-lines (children of FBDeptRevenue) ----
    "BanquetRevenue": {"parent": "FBDeptRevenue", "level": 3, "source": "Banquet_Revenue"},
    "RestaurantLoungeRevenue": {"parent": "FBDeptRevenue", "level": 3, "source": "Restaurant_Revenue"},
}

# Nodes that are forecast directly with Prophet (top-level drivers).
TOP_LEVEL_FORECAST_NODES: List[str] = ["TotalOperatingRevenue", "GOP", "NOI"]


def parent_of(node: str) -> Optional[str]:
    """Return the parent node name, or None for a root."""
    return NODES[node]["parent"]


def children_of(node: str) -> List[str]:
    """Return the list of direct children for a node."""
    return [name for name, meta in NODES.items() if meta["parent"] == node]


def source_of(node: str) -> Optional[str]:
    """Return the canonical source column for a node, or None if derived."""
    return NODES[node]["source"]


def level_of(node: str) -> int:
    """Return the hierarchy level (1=top) of a node."""
    return NODES[node]["level"]


def nodes_at_level(level: int) -> List[str]:
    """Return all node names at a given level."""
    return [name for name, meta in NODES.items() if meta["level"] == level]


def as_parent_child_table() -> List[dict]:
    """Return the hierarchy as a flat list of rows (node, parent, level, source).

    Useful for serialization or display.
    """
    return [
        {
            "node": name,
            "parent": meta["parent"],
            "level": meta["level"],
            "source": meta["source"],
        }
        for name, meta in NODES.items()
    ]
