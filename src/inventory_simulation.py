import pandas as pd


def simulate_inventory_fulfillment(
    simulation_df,
    initial_stock=10
):
    """
    Simulate inventory fulfillment across fleet
    observation points.

    Demand is simulated from the number of engines
    predicted to be in the critical failure window.
    """

    stock = initial_stock

    total_demand = 0
    fulfilled_demand = 0
    total_stockouts = 0
    total_inventory_used = 0
    total_carrying_cost = 0

    results = []

    for _, row in simulation_df.iterrows():

        # Recommended inventory for this fleet state
        target_stock = int(row["target_stock"])

        # Increase stock when predicted requirement rises
        if stock < target_stock:
            stock = target_stock

        stock_before_demand = stock

        # ----------------------------------------------------
        # Simulated actual demand
        # ----------------------------------------------------
        #
        # Each high-risk engine generates one component
        # demand event.
        #
        # This is explicitly a simulation assumption.
        #

        demand = int(row["high_risk_engines"])

        # ----------------------------------------------------
        # Fulfill demand
        # ----------------------------------------------------

        fulfilled = min(
            stock,
            demand
        )

        stock -= fulfilled

        stockout = demand - fulfilled

        # ----------------------------------------------------
        # Update totals
        # ----------------------------------------------------

        total_demand += demand

        fulfilled_demand += fulfilled

        total_stockouts += stockout

        total_inventory_used += fulfilled

        # $200 per unit per simulation cycle
        carrying_cost = (
            stock
            * 200
        )

        total_carrying_cost += carrying_cost

        # ----------------------------------------------------
        # Save cycle result
        # ----------------------------------------------------

        results.append({

            "cycle":
                row["cycle"],

            "target_stock":
                target_stock,

            "stock_before_demand":
                stock_before_demand,

            "actual_demand":
                demand,

            "fulfilled_demand":
                fulfilled,

            "stockout":
                stockout,

            "ending_stock":
                stock,

            "carrying_cost":
                carrying_cost
        })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    if total_demand > 0:

        fill_rate = (
            fulfilled_demand /
            total_demand
        )

    else:

        fill_rate = 1.0

    if total_demand > 0:

        stockout_rate = (
            total_stockouts /
            total_demand
        )

    else:

        stockout_rate = 0.0

    average_inventory = (
        results_df["ending_stock"].mean()
        if len(results_df) > 0
        else 0
    )

    if average_inventory > 0:

        inventory_turnover = (
            total_inventory_used /
            average_inventory
        )

    else:

        inventory_turnover = 0

    kpis = {

        "total_demand":
            total_demand,

        "fulfilled_demand":
            fulfilled_demand,

        "total_stockouts":
            total_stockouts,

        "fill_rate":
            fill_rate,

        "stockout_rate":
            stockout_rate,

        "inventory_used":
            total_inventory_used,

        "average_inventory":
            average_inventory,

        "inventory_turnover":
            inventory_turnover,

        "total_carrying_cost":
            total_carrying_cost
    }

    return results_df, kpis