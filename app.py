import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fleet Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

try:
    simulation_df = pd.read_csv("fleet_simulation_results.csv")
    fulfillment_df = pd.read_csv("inventory_fulfillment_results.csv")

except FileNotFoundError:
    st.error(
        "Simulation files not found. Please run "
        "`python train.py` first."
    )
    st.stop()


# ============================================================
# CALCULATE METRICS
# ============================================================

peak_inventory = int(
    simulation_df["target_stock"].max()
)

average_inventory = (
    simulation_df["target_stock"].mean()
)

average_risk = (
    simulation_df["average_failure_probability"].mean()
)

peak_risk = (
    simulation_df["average_failure_probability"].max()
)

peak_risk_cycle = simulation_df.loc[
    simulation_df["average_failure_probability"].idxmax(),
    "cycle"
]

peak_inventory_cycle = simulation_df.loc[
    simulation_df["target_stock"].idxmax(),
    "cycle"
]

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
    fill_rate = fulfilled_demand / total_demand
else:
    fill_rate = 1.0


# ============================================================
# BUSINESS IMPACT - UNSEEN TEST SET
# ============================================================

REACTIVE_COST = 49_800_000
AI_COST = 17_975_000

SIMULATED_SAVINGS = (
    REACTIVE_COST - AI_COST
)

SAVINGS_PERCENTAGE = (
    SIMULATED_SAVINGS
    / REACTIVE_COST
    * 100
)

PREVENTIVE_ACTIONS = 407
MISSED_FAILURES = 52
FALSE_ALARMS = 127
TRUE_FAILURES_DETECTED = 280


# ============================================================
# INVENTORY KPIs
# ============================================================

INVENTORY_CARRYING_COST = 9_600
INVENTORY_TURNOVER = 5.01


# ============================================================
# HEADER
# ============================================================

st.title("✈️ Fleet Predictive Maintenance")

st.markdown(
    "**AI-driven maintenance prediction + dynamic spare inventory optimization**"
)

st.divider()


# ============================================================
# FLEET HEALTH STATUS
# ============================================================

if peak_risk < 0.10:

    status = "🟢 HEALTHY"

    recommendation = (
        "Fleet risk is currently low. Maintain normal monitoring "
        "and baseline spare inventory."
    )

elif peak_risk < 0.25:

    status = "🟡 WATCH"

    recommendation = (
        "Fleet degradation is increasing. Closely monitor "
        "high-risk engines and prepare additional spare inventory."
    )

else:

    status = "🔴 CRITICAL"

    recommendation = (
        "Significant fleet degradation detected. Prioritize "
        "high-risk engines for preventive maintenance and "
        "increase spare inventory readiness."
    )


# ============================================================
# CURRENT FLEET STATUS
# ============================================================

st.subheader("🚦 Current Fleet Status")

col1, col2 = st.columns([1, 3])

with col1:

    st.metric(
        "Fleet Status",
        status
    )

with col2:

    st.info(
        f"🧠 **AI Recommendation:** {recommendation}"
    )


# ============================================================
# TOP KPI CARDS
# ============================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Fleet Risk",
        f"{average_risk * 100:.2f}%"
    )

with col2:

    st.metric(
        "Peak Fleet Risk",
        f"{peak_risk * 100:.2f}%",
        f"Cycle {int(peak_risk_cycle)}"
    )

with col3:

    st.metric(
        "Peak Recommended Stock",
        f"{peak_inventory} units",
        f"Cycle {int(peak_inventory_cycle)}"
    )

with col4:

    st.metric(
        "Simulated Fill Rate",
        f"{fill_rate * 100:.2f}%"
    )


# ============================================================
# RISK TREND
# ============================================================

st.subheader("📈 Fleet Failure Risk Over Lifecycle")

risk_chart = simulation_df[
    [
        "cycle",
        "average_failure_probability"
    ]
].copy()

risk_chart["average_failure_probability"] *= 100

risk_chart = risk_chart.set_index("cycle")

st.line_chart(
    risk_chart
)

st.caption(
    "Average predicted probability of an engine entering the "
    "30-cycle critical RUL window."
)


# ============================================================
# HIGH RISK ENGINES
# ============================================================

st.subheader("🚨 High-Risk Engine Trend")

high_risk_chart = simulation_df[
    [
        "cycle",
        "high_risk_engines"
    ]
].set_index("cycle")

st.line_chart(
    high_risk_chart
)

st.caption(
    "Number of engines classified as high-risk under the "
    "selected simulation threshold."
)


# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

st.subheader("📦 Dynamic Spare Inventory Recommendation")

inventory_chart = simulation_df[
    [
        "cycle",
        "target_stock"
    ]
].set_index("cycle")

st.line_chart(
    inventory_chart
)

st.info(
    f"📦 **Inventory Decision:** The optimizer recommends a "
    f"peak stock level of **{peak_inventory} units** around "
    f"cycle **{int(peak_inventory_cycle)}**, reflecting "
    f"increased predicted fleet risk."
)


# ============================================================
# INVENTORY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Average Inventory",
        f"{average_inventory:.2f} units"
    )

with col2:

    st.metric(
        "Peak Inventory",
        f"{peak_inventory} units"
    )

with col3:

    st.metric(
        "Simulated Demand",
        total_demand
    )

with col4:

    st.metric(
        "Stockouts",
        stockouts
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.divider()

st.subheader("💰 Simulated Business Impact")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Reactive Cost",
        "$49.80M"
    )

with col2:

    st.metric(
        "AI-Assisted Cost",
        "$17.98M"
    )

with col3:

    st.metric(
        "Simulated Cost Reduction",
        f"{SAVINGS_PERCENTAGE:.2f}%"
    )

with col4:

    st.metric(
        "Preventive Actions",
        PREVENTIVE_ACTIONS
    )


st.success(
    f"💰 Under the defined simulation assumptions, the "
    f"AI-assisted strategy reduces modeled maintenance cost "
    f"by approximately **${SIMULATED_SAVINGS / 1_000_000:.2f}M** "
    f"compared with the reactive-maintenance baseline."
)


# ============================================================
# MAINTENANCE DETAILS
# ============================================================

st.subheader("🔧 Maintenance Decision Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Preventive Actions",
        PREVENTIVE_ACTIONS
    )

with col2:

    st.metric(
        "Missed Critical Cases",
        MISSED_FAILURES
    )

with col3:

    st.metric(
        "False Alarms",
        FALSE_ALARMS
    )

with col4:

    st.metric(
        "Critical Cases Detected",
        TRUE_FAILURES_DETECTED
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Unseen Test-Set Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Precision",
        "68.8%"
    )

with col2:

    st.metric(
        "Recall",
        "84.3%"
    )

with col3:

    st.metric(
        "F1 Score",
        "75.8%"
    )

with col4:

    st.metric(
        "ROC-AUC",
        "99.4%"
    )

st.info(
    "The model was evaluated on the unseen NASA C-MAPSS test set. "
    "A 0.10 decision threshold was selected to prioritize detection "
    "of critical engines and reduce missed failures."
)


# ============================================================
# FLEET SIMULATION TABLE
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

display_df["average_failure_probability"] *= 100

display_df.columns = [
    "Cycle",
    "Engines Observed",
    "Average RUL",
    "Failure Risk (%)",
    "High-Risk Engines",
    "Expected Demand",
    "Safety Stock",
    "Target Stock"
]

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# INVENTORY FULFILLMENT
# ============================================================

st.subheader("📦 Inventory Fulfillment Simulation")

fulfillment_display = fulfillment_df.copy()

st.dataframe(
    fulfillment_display,
    width="stretch",
    hide_index=True
)


# ============================================================
# INVENTORY KPIs
# ============================================================

st.subheader("📊 Inventory Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Carrying Cost",
        f"${INVENTORY_CARRYING_COST:,.0f}"
    )

with col2:

    st.metric(
        "Inventory Turnover",
        f"{INVENTORY_TURNOVER:.2f}"
    )

with col3:

    st.metric(
        "Fill Rate",
        f"{fill_rate * 100:.2f}%"
    )

with col4:

    st.metric(
        "Stockouts",
        stockouts
    )


# ============================================================
# ASSUMPTIONS
# ============================================================

st.divider()

with st.expander("⚠️ Simulation Assumptions & Limitations"):

    st.markdown(
        """
        **Dataset**

        The predictive model uses the NASA C-MAPSS turbofan
        engine degradation dataset. C-MAPSS is a simulated
        benchmark dataset rather than live airline operational data.

        **Model evaluation**

        The reported model metrics are calculated on the unseen
        C-MAPSS test set using a 0.10 decision threshold.

        **Maintenance assumptions**

        - Critical window: RUL ≤ 30 cycles
        - AOG / missed-failure cost: **$150,000**
        - Preventive maintenance cost: **$25,000**
        - Inventory carrying cost: **$200 per unit per cycle**

        **Inventory assumption**

        One high-risk engine is treated as one simulated
        component-demand event.

        **Inventory planning**

        The optimizer uses predicted failure probabilities,
        demand uncertainty, lead time, and a 95% target service
        level to determine recommended spare inventory.

        **Important**

        The financial results and inventory fill rate shown
        above are **simulation results under defined assumptions**.
        They are not actual airline savings, operational
        guarantees, or real-world MRO cost estimates.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fleet Predictive Maintenance & Inventory Optimization Platform "
    "| AI + Predictive Analytics + Decision Optimization"
)