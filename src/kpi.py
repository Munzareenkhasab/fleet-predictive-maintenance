import pandas as pd


# ============================================================
# SIMULATION COST ASSUMPTIONS
# ============================================================

INVENTORY_COST_PER_UNIT_PER_CYCLE = 200

PREVENTIVE_MAINTENANCE_COST = 25_000

AOG_COST = 150_000


def calculate_inventory_kpis(results_df):
    """
    Calculate inventory and operational KPIs
    from the fleet simulation results.
    """

    df = results_df.copy()

    # --------------------------------------------------------
    # Inventory carrying cost
    # --------------------------------------------------------

    df["inventory_carrying_cost"] = (
        df["target_stock"]
        * INVENTORY_COST_PER_UNIT_PER_CYCLE
    )

    total_inventory_cost = (
        df["inventory_carrying_cost"].sum()
    )

    # --------------------------------------------------------
    # Inventory utilization
    # --------------------------------------------------------

    total_stock = (
        df["target_stock"].sum()
    )

    average_inventory = (
        df["target_stock"].mean()
    )

    # Simulated component demand:
    # expected demand accumulated across
    # the fleet observation points.
    total_expected_demand = (
        df["expected_demand"].sum()
    )

    # --------------------------------------------------------
    # Inventory turnover
    # --------------------------------------------------------

    if average_inventory > 0:

        inventory_turnover = (
            total_expected_demand /
            average_inventory
        )

    else:

        inventory_turnover = 0

    # --------------------------------------------------------
    # Service-level proxy
    # --------------------------------------------------------

    # Inventory coverage ratio:
    # how much expected demand is covered
    # by recommended stock.

    total_capacity = (
        df["target_stock"].sum()
    )

    if total_expected_demand > 0:

        service_level = min(
            1.0,
            total_capacity /
            total_expected_demand
        )

    else:

        service_level = 1.0

    # --------------------------------------------------------
    # Peak inventory
    # --------------------------------------------------------

    peak_inventory = (
        df["target_stock"].max()
    )

    minimum_inventory = (
        df["target_stock"].min()
    )

    # --------------------------------------------------------
    # Average fleet risk
    # --------------------------------------------------------

    average_fleet_risk = (
        df[
            "average_failure_probability"
        ].mean()
    )

    # --------------------------------------------------------
    # Peak fleet risk
    # --------------------------------------------------------

    peak_fleet_risk = (
        df[
            "average_failure_probability"
        ].max()
    )

    return {
        "total_inventory_cost":
            total_inventory_cost,

        "average_inventory":
            average_inventory,

        "inventory_turnover":
            inventory_turnover,

        "service_level":
            service_level,

        "peak_inventory":
            peak_inventory,

        "minimum_inventory":
            minimum_inventory,

        "average_fleet_risk":
            average_fleet_risk,

        "peak_fleet_risk":
            peak_fleet_risk,

        "total_expected_demand":
            total_expected_demand
    }