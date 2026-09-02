import pandas as pd


AOG_COST = 150_000
PREVENTIVE_COST = 25_000


def calculate_business_value(
    actual,
    predicted,
    threshold
):

    # Reset indexes so comparison is strictly row-by-row
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)

    # Convert probability into maintenance decision
    decisions = (
        predicted >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # CLASSIFY BUSINESS OUTCOMES
    # --------------------------------------------------------

    true_failures = (
        (actual == 1) &
        (decisions == 1)
    ).sum()

    false_alarms = (
        (actual == 0) &
        (decisions == 1)
    ).sum()

    missed_failures = (
        (actual == 1) &
        (decisions == 0)
    ).sum()

    # --------------------------------------------------------
    # MAINTENANCE DECISIONS
    # --------------------------------------------------------

    preventive_actions = (
        true_failures +
        false_alarms
    )

    # Every predicted critical case triggers
    # preventive maintenance in our simulation.
    preventive_cost = (
        preventive_actions *
        PREVENTIVE_COST
    )

    # Missed critical cases become unscheduled
    # AOG events in the simulation.
    aog_cost = (
        missed_failures *
        AOG_COST
    )

    # --------------------------------------------------------
    # TOTAL AI-ASSISTED COST
    # --------------------------------------------------------

    ai_cost = (
        preventive_cost +
        aog_cost
    )

    # --------------------------------------------------------
    # REACTIVE BASELINE
    # --------------------------------------------------------

    # Assume every critical event becomes an
    # unscheduled AOG event without predictive maintenance.
    reactive_cost = (
        actual.sum() *
        AOG_COST
    )

    # --------------------------------------------------------
    # SAVINGS
    # --------------------------------------------------------

    cost_difference = (
        reactive_cost -
        ai_cost
    )

    savings_percentage = (
        cost_difference /
        reactive_cost *
        100
        if reactive_cost > 0
        else 0
    )

    # --------------------------------------------------------
    # RETURN BUSINESS KPIs
    # --------------------------------------------------------

    return {

        "reactive_cost":
            reactive_cost,

        "ai_cost":
            ai_cost,

        "cost_difference":
            cost_difference,

        "savings_percentage":
            savings_percentage,

        "preventive_actions":
            preventive_actions,

        "true_failures_detected":
            true_failures,

        "false_alarms":
            false_alarms,

        "missed_failures":
            missed_failures,

        "preventive_cost":
            preventive_cost,

        "aog_cost":
            aog_cost
    }