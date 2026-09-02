import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fleet Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("✈️ Fleet Predictive Maintenance & Inventory Optimization")

st.caption(
    "AI-driven predictive maintenance and dynamic spare inventory planning"
)


# ============================================================
# LOAD RESULTS
# ============================================================

try:

    simulation_df = pd.read_csv(
        "fleet_simulation_results.csv"
    )

    fulfillment_df = pd.read_csv(
        "inventory_fulfillment_results.csv"
    )

except FileNotFoundError:

    st.error(
        "Simulation files not found. "
        "Please run `python train.py` first."
    )

    st.stop()


# ============================================================
# CALCULATE DASHBOARD KPIs
# ============================================================

peak_inventory = int(
    simulation_df["target_stock"].max()
)

average_inventory = (
    simulation_df["target_stock"].mean()
)

average_risk = (
    simulation_df[
        "average_failure_probability"
    ].mean()
)

peak_risk = (
    simulation_df[
        "average_failure_probability"
    ].max()
)

total_demand = int(
    fulfillment_df["actual_demand"].sum()
)

fulfilled_demand = int(
    fulfillment_df["fulfilled_demand"].sum()
)

stockouts = int(
    fulfillment_df["stockout"].sum()
)

if total_demand > 0:

    fill_rate = (
        fulfilled_demand /
        total_demand
    )

else:

    fill_rate = 1.0


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Fleet Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Fleet Risk",
        f"{average_risk * 100:.2f}%"
    )

with col2:

    st.metric(
        "Peak Fleet Risk",
        f"{peak_risk * 100:.2f}%"
    )

with col3:

    st.metric(
        "Peak Recommended Stock",
        f"{peak_inventory} units"
    )

with col4:

    st.metric(
        "Simulated Fill Rate",
        f"{fill_rate * 100:.2f}%"
    )


# ============================================================
# FLEET RISK CHART
# ============================================================

st.subheader("📈 Fleet Failure Risk Over Lifecycle")

risk_chart = simulation_df[
    [
        "cycle",
        "average_failure_probability"
    ]
].set_index("cycle")

st.line_chart(
    risk_chart
)


# ============================================================
# INVENTORY CHART
# ============================================================

st.subheader("📦 Dynamic Inventory Recommendation")

inventory_chart = simulation_df[
    [
        "cycle",
        "target_stock"
    ]
].set_index("cycle")

st.line_chart(
    inventory_chart
)


# ============================================================
# FLEET HEALTH TABLE
# ============================================================

st.subheader("🔍 Fleet Simulation Details")

display_df = simulation_df[
    [
        "cycle",
        "engines_observed",
        "average_rul",
        "average_failure_probability",
        "high_risk_engines",
        "expected_demand",
        "safety_stock",
        "target_stock"
    ]
].copy()

display_df.columns = [
    "Cycle",
    "Engines Observed",
    "Average RUL",
    "Average Failure Risk",
    "High-Risk Engines",
    "Expected Demand",
    "Safety Stock",
    "Target Stock"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INVENTORY FULFILLMENT
# ============================================================

st.subheader("📦 Inventory Fulfillment")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Simulated Demand",
        total_demand
    )

with col2:

    st.metric(
        "Fulfilled Demand",
        fulfilled_demand
    )

with col3:

    st.metric(
        "Stockouts",
        stockouts
    )


# ============================================================
# INVENTORY DETAILS
# ============================================================

st.subheader("Inventory Simulation")

st.dataframe(
    fulfillment_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ASSUMPTIONS
# ============================================================

st.divider()

st.subheader("⚠️ Simulation Assumptions")

st.write(
    """
    This dashboard uses the NASA C-MAPSS turbofan degradation
    dataset and simulated operational cost assumptions.

    • Failure-risk threshold: 30 cycles RUL

    • Inventory carrying cost: $200 per unit per cycle

    • Preventive maintenance cost: $25,000

    • AOG / missed-failure cost: $150,000

    • One high-risk engine is treated as one simulated
      component-demand event.

    These values are simulation assumptions and should not
    be interpreted as actual airline MRO pricing or
    guaranteed operational savings.
    """
)