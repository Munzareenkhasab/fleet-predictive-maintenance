import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from src.features import build_features

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fleetora",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "failure_classifier.pkl"

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

DECISION_THRESHOLD = 0.10
PREVENTIVE_COST = 25_000
AOG_COST = 150_000

# Existing project results
REACTIVE_COST = 49_800_000
AI_COST = 17_975_000
COST_REDUCTION = REACTIVE_COST - AI_COST
SAVINGS_PERCENTAGE = COST_REDUCTION / REACTIVE_COST * 100

PREVENTIVE_ACTIONS = 407
TRUE_FAILURES_DETECTED = 280
FALSE_ALARMS = 127
MISSED_FAILURES = 52

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_results():
    simulation_path = BASE_DIR / "fleet_simulation_results.csv"
    fulfillment_path = BASE_DIR / "inventory_fulfillment_results.csv"

    simulation = (
        pd.read_csv(simulation_path)
        if simulation_path.exists()
        else pd.DataFrame()
    )

    fulfillment = (
        pd.read_csv(fulfillment_path)
        if fulfillment_path.exists()
        else pd.DataFrame()
    )

    return simulation, fulfillment


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def get_model_feature_names(model):
    """Get the exact feature order stored in the trained LightGBM model."""
    if model is None:
        return []

    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())

    return []


def classify_risk(probability):
    if probability < 0.10:
        return "🟢 LOW", "Continue normal monitoring."

    if probability < 0.30:
        return "🟡 MEDIUM", "Monitor closely and prepare for maintenance."

    if probability < 0.70:
        return "🟠 HIGH", "Schedule preventive maintenance."

    return "🔴 CRITICAL", "Prioritize immediate maintenance review."


def maintenance_decision(probability):
    if probability >= DECISION_THRESHOLD:
        return "Preventive maintenance recommended"

    return "Continue operation and monitor"


def build_prediction_template():
    """Create a simple multi-cycle CSV template for new engine data."""
    rows = []

    for cycle in range(1, 16):

        row = {
            "unit": 101,
            "cycle": cycle,
        }

        for sensor in SENSOR_COLUMNS:
            row[sensor] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def validate_input(df):

    required = {"unit", "cycle", *SENSOR_COLUMNS}

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        return False, missing

    if df.empty:
        return False, ["The uploaded CSV is empty."]

    if df[
        ["unit", "cycle"] + SENSOR_COLUMNS
    ].isnull().any().any():

        return False, [
            "Missing values detected in required columns."
        ]

    return True, []


def predict_uploaded_data(uploaded_df, model):

    """
    Run the same feature-engineering pipeline used during training,
    then select the exact feature names stored by the trained model.
    """

    working = uploaded_df.copy()

    working["unit"] = pd.to_numeric(
        working["unit"],
        errors="coerce"
    )

    working["cycle"] = pd.to_numeric(
        working["cycle"],
        errors="coerce"
    )

    for sensor in SENSOR_COLUMNS:

        working[sensor] = pd.to_numeric(
            working[sensor],
            errors="coerce"
        )

    working = working.dropna(
        subset=["unit", "cycle"] + SENSOR_COLUMNS
    )

    if working.empty:
        raise ValueError(
            "No valid numeric engine records were found."
        )

    working = working.sort_values(
        ["unit", "cycle"]
    ).copy()

    # Same window used by the training pipeline.
    engineered = build_features(
        working,
        window=15
    )

    model_features = get_model_feature_names(model)

    if not model_features:
        raise ValueError(
            "Could not read feature names from the trained "
            "LightGBM model."
        )

    missing_features = [
        feature
        for feature in model_features
        if feature not in engineered.columns
    ]

    if missing_features:

        raise ValueError(
            "The uploaded data could not reproduce the model "
            "features. "
            f"Missing engineered features: "
            f"{missing_features[:8]}"
        )

    X_new = engineered[
        model_features
    ].copy()

    probabilities = model.predict_proba(
        X_new
    )[:, 1]

    result = working[
        ["unit", "cycle"]
    ].copy()

    result["failure_probability"] = probabilities

    result["risk"] = [
        classify_risk(float(p))[0]
        for p in probabilities
    ]

    result["maintenance_decision"] = [
        maintenance_decision(float(p))
        for p in probabilities
    ]

    # Latest available record for each engine.
    latest = (
        result
        .sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .copy()
    )

    latest["probability_percent"] = (
        latest["failure_probability"] * 100
    )

    latest["estimated_preventive_cost"] = np.where(
        latest["failure_probability"]
        >= DECISION_THRESHOLD,
        PREVENTIVE_COST,
        0,
    )

    latest["estimated_exposure_if_missed"] = np.where(
        latest["failure_probability"]
        >= DECISION_THRESHOLD,
        latest["failure_probability"] * AOG_COST,
        0,
    )

    return result, latest


# ============================================================
# LOAD EXISTING DASHBOARD DATA
# ============================================================

simulation_df, fulfillment_df = load_results()

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✈️ FLEETORA")

    st.caption(
        "AI-Powered Fleet Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "✈️ Fleet Operations",
            "⚠️ Risk & Maintenance",
            "📦 Inventory Optimization",
            "🔮 Predict New Engine",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        "Predict. Prevent. Optimize."
    )

    st.caption(
        "NASA C-MAPSS • LightGBM • Decision Optimization"
    )


# ============================================================
# HEADER
# ============================================================

st.title("✈️ FLEETORA")

st.markdown(
    "### AI-Powered Fleet Predictive Maintenance & Inventory Intelligence"
)

st.caption(
    "**Predict. Prevent. Optimize.**"
)

st.divider()


# ============================================================
# PAGE 1 — FLEET OPERATIONS
# ============================================================

if page == "✈️ Fleet Operations":

    st.subheader("✈️ Fleet Operations")

    if simulation_df.empty:

        st.warning(
            "Fleet simulation results are not available."
        )

        st.stop()

    latest = simulation_df.iloc[-1]

    latest_engines = int(
        latest["engines_observed"]
    )

    latest_risk = float(
        latest["average_failure_probability"]
    )

    latest_rul = float(
        latest["average_rul"]
    )

    latest_high_risk = int(
        latest["high_risk_engines"]
    )

    status, recommendation = classify_risk(
        latest_risk
    )

    st.info(
        f"**Current fleet status:** {status}  \n"
        f"{recommendation}"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Engines Monitored",
            latest_engines
        )

    with c2:
        st.metric(
            "High-Risk Engines",
            latest_high_risk
        )

    with c3:
        st.metric(
            "Average Fleet Risk",
            f"{latest_risk * 100:.2f}%"
        )

    with c4:
        st.metric(
            "Average RUL",
            f"{latest_rul:.1f} cycles"
        )

    st.divider()

    st.subheader(
        "📈 Fleet Risk Across Lifecycle"
    )

    risk_chart = simulation_df[
        [
            "cycle",
            "average_failure_probability"
        ]
    ].copy()

    risk_chart[
        "average_failure_probability"
    ] *= 100

    risk_chart = risk_chart.set_index(
        "cycle"
    )

    st.line_chart(
        risk_chart
    )

    st.caption(
        "Average predicted probability of an engine "
        "entering the critical RUL window."
    )

    st.subheader(
        "🚨 High-Risk Engine Trend"
    )

    high_risk_chart = simulation_df[
        [
            "cycle",
            "high_risk_engines"
        ]
    ].set_index(
        "cycle"
    )

    st.line_chart(
        high_risk_chart
    )

    st.subheader(
        "🔍 Fleet Snapshot"
    )

    fleet_display = simulation_df[
        [
            "cycle",
            "engines_observed",
            "average_rul",
            "average_failure_probability",
            "high_risk_engines",
        ]
    ].copy()

    fleet_display[
        "average_failure_probability"
    ] *= 100

    fleet_display.columns = [
        "Cycle",
        "Engines Observed",
        "Average RUL",
        "Failure Risk (%)",
        "High-Risk Engines",
    ]

    st.dataframe(
        fleet_display,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PAGE 2 — RISK & MAINTENANCE
# ============================================================

elif page == "⚠️ Risk & Maintenance":

    st.subheader(
        "⚠️ Risk & Maintenance Intelligence"
    )

    st.markdown(
        "This layer converts predicted engine failure risk "
        "into maintenance decisions and estimated business impact."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Reactive Maintenance Cost",
            "$49.80M"
        )

    with c2:
        st.metric(
            "AI-Assisted Cost",
            "$17.98M"
        )

    with c3:
        st.metric(
            "Modeled Cost Reduction",
            "$31.83M"
        )

    with c4:
        st.metric(
            "Simulated Savings",
            f"{SAVINGS_PERCENTAGE:.2f}%"
        )

    st.success(
        f"💰 Under the defined simulation assumptions, "
        f"the AI-assisted strategy reduces modeled maintenance "
        f"cost by approximately "
        f"**${COST_REDUCTION / 1_000_000:.2f}M** compared with "
        f"the reactive-maintenance baseline."
    )

    st.divider()

    st.subheader(
        "🔧 Maintenance Decisions"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Preventive Actions",
            PREVENTIVE_ACTIONS
        )

    with c2:
        st.metric(
            "Critical Cases Detected",
            TRUE_FAILURES_DETECTED
        )

    with c3:
        st.metric(
            "False Alarms",
            FALSE_ALARMS
        )

    with c4:
        st.metric(
            "Missed Critical Cases",
            MISSED_FAILURES
        )

    st.divider()

    st.subheader(
        "🤖 Unseen Test-Set Model Performance"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Precision",
            "68.8%"
        )

    with c2:
        st.metric(
            "Recall",
            "84.3%"
        )

    with c3:
        st.metric(
            "F1 Score",
            "75.8%"
        )

    with c4:
        st.metric(
            "ROC-AUC",
            "99.4%"
        )

    st.info(
        "The model was evaluated on an unseen NASA C-MAPSS "
        "test set. A 0.10 decision threshold was selected "
        "as a safety-focused classification threshold."
    )

    st.subheader(
        "📊 Failure Risk Trend"
    )

    if not simulation_df.empty:

        risk_chart = simulation_df[
            [
                "cycle",
                "average_failure_probability"
            ]
        ].copy()

        risk_chart[
            "average_failure_probability"
        ] *= 100

        risk_chart = risk_chart.set_index(
            "cycle"
        )

        st.line_chart(
            risk_chart
        )

    st.subheader(
        "🧠 Decision Logic"
    )

    st.markdown(
        """
        **LightGBM prediction → failure probability → decision threshold → maintenance action**

        - Probability below **0.10** → continue monitoring.
        - Probability at or above **0.10** → preventive maintenance decision.
        - Preventive maintenance assumption → **$25,000**.
        - Missed critical failure / AOG assumption → **$150,000**.
        """
    )

    with st.expander(
        "⚠️ Business assumptions"
    ):

        st.markdown(
            """
            - Critical window: RUL ≤ 30 cycles
            - AOG / missed-failure cost: **$150,000**
            - Preventive maintenance cost: **$25,000**
            - Financial results are simulated under defined assumptions.
            - They are not actual airline savings or operational guarantees.
            """
        )


# ============================================================
# PAGE 3 — INVENTORY OPTIMIZATION
# ============================================================

elif page == "📦 Inventory Optimization":

    st.subheader(
        "📦 Inventory Optimization"
    )

    st.markdown(
        "Dynamic spare-parts planning driven by predicted "
        "fleet failure risk."
    )

    if simulation_df.empty:

        st.warning(
            "Inventory simulation results are not available."
        )

        st.stop()

    latest = simulation_df.iloc[-1]

    peak_inventory = int(
        simulation_df["target_stock"].max()
    )

    average_inventory = (
        simulation_df["target_stock"].mean()
    )

    peak_inventory_cycle = simulation_df.loc[
        simulation_df["target_stock"].idxmax(),
        "cycle",
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Expected Spare Demand",
            f"{latest['expected_demand']:.2f}"
        )

    with c2:
        st.metric(
            "Safety Stock",
            f"{latest['safety_stock']:.2f}"
        )

    with c3:
        st.metric(
            "Recommended Target Stock",
            f"{int(latest['target_stock'])} units"
        )

    with c4:
        st.metric(
            "Peak Recommended Stock",
            f"{peak_inventory} units"
        )

    st.divider()

    st.subheader(
        "📈 Dynamic Inventory Recommendation"
    )

    inventory_chart = simulation_df[
        [
            "cycle",
            "target_stock"
        ]
    ].set_index(
        "cycle"
    )

    st.line_chart(
        inventory_chart
    )

    st.info(
        f"📦 The optimizer reaches a peak recommendation "
        f"of **{peak_inventory} units** around cycle "
        f"**{int(peak_inventory_cycle)}**, reflecting increased "
        f"predicted fleet risk."
    )

    st.subheader(
        "📊 Inventory Performance"
    )

    total_demand = (
        int(
            fulfillment_df[
                "actual_demand"
            ].sum()
        )
        if "actual_demand" in fulfillment_df.columns
        else 0
    )

    fulfilled_demand = (
        int(
            fulfillment_df[
                "fulfilled_demand"
            ].sum()
        )
        if "fulfilled_demand" in fulfillment_df.columns
        else 0
    )

    stockouts = (
        int(
            fulfillment_df[
                "stockout"
            ].sum()
        )
        if "stockout" in fulfillment_df.columns
        else 0
    )

    fill_rate = (
        fulfilled_demand / total_demand
        if total_demand > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Average Inventory",
            f"{average_inventory:.2f} units"
        )

    with c2:
        st.metric(
            "Carrying Cost",
            "$9,600"
        )

    with c3:
        st.metric(
            "Inventory Turnover",
            "5.01"
        )

    with c4:
        st.metric(
            "Fill Rate",
            f"{fill_rate * 100:.2f}%"
        )

    st.subheader(
        "📦 Fulfillment"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Simulated Demand",
            total_demand
        )

    with c2:
        st.metric(
            "Fulfilled Demand",
            fulfilled_demand
        )

    with c3:
        st.metric(
            "Stockouts",
            stockouts
        )

    with c4:

        stockout_rate = (
            stockouts / total_demand * 100
            if total_demand
            else 0
        )

        st.metric(
            "Stockout Rate",
            f"{stockout_rate:.2f}%"
        )

    st.divider()

    st.subheader(
        "🔍 Inventory Simulation Details"
    )

    display_df = simulation_df[
        [
            "cycle",
            "engines_observed",
            "average_rul",
            "average_failure_probability",
            "high_risk_engines",
            "expected_demand",
            "safety_stock",
            "target_stock",
        ]
    ].copy()

    display_df[
        "average_failure_probability"
    ] *= 100

    display_df.columns = [
        "Cycle",
        "Engines Observed",
        "Average RUL",
        "Failure Risk (%)",
        "High-Risk Engines",
        "Expected Demand",
        "Safety Stock",
        "Target Stock",
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    with st.expander(
        "⚙️ How the optimizer works"
    ):

        st.markdown(
            """
            The inventory module is a custom probabilistic
            inventory optimizer, not a separately trained ML model.

            **LightGBM failure probabilities**
            → expected spare demand
            → demand uncertainty
            → safety stock
            → target inventory.

            Current simulation:

            - Lead time = **5 cycles**
            - Service level = **95%**
            - Initial stock = **10 units**
            """
        )


# ============================================================
# PAGE 4 — PREDICT NEW ENGINE
# ============================================================

else:

    st.subheader(
        "🔮 Predict New Engine Risk"
    )

    st.markdown(
        """
        Upload new engine sensor history and run it through the
        **same feature-engineering pipeline** used during model
        development.

        The saved LightGBM classifier then returns a failure
        probability and maintenance recommendation.
        """
    )

    if model is None:

        st.error(
            "The trained model was not found at "
            "`models/failure_classifier.pkl`. "
            "Make sure the model file is committed to GitHub."
        )

        st.stop()

    model_features = get_model_feature_names(
        model
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Decision Threshold",
            "10%"
        )

    with c2:
        st.metric(
            "Model Features",
            len(model_features)
        )

    with c3:
        st.metric(
            "Feature Window",
            "15 cycles"
        )

    st.divider()

    st.subheader(
        "📄 Step 1 — Prepare Engine Data"
    )

    template = build_prediction_template()

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="engine_prediction_template.csv",
        mime="text/csv",
    )

    st.caption(
        "The CSV must contain: unit, cycle, and sensor_1 "
        "through sensor_21. Provide multiple consecutive "
        "cycles for each engine so rolling features and "
        "degradation slopes can be calculated."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload engine sensor CSV",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV to generate a prediction. "
            "For a meaningful prediction, include recent "
            "consecutive sensor readings for each engine."
        )

    else:

        try:

            input_df = pd.read_csv(
                uploaded_file
            )

        except Exception as exc:

            st.error(
                f"Could not read the CSV: {exc}"
            )

            st.stop()

        valid, problems = validate_input(
            input_df
        )

        if not valid:

            st.error(
                "CSV validation failed."
            )

            for problem in problems:
                st.write(
                    f"• {problem}"
                )

            st.stop()

        st.success(
            f"CSV loaded successfully: "
            f"{len(input_df):,} sensor records."
        )

        st.subheader(
            "📋 Uploaded Data Preview"
        )

        st.dataframe(
            input_df.head(10),
            width="stretch",
            hide_index=True,
        )

        if st.button(
            "🚀 Run Failure Prediction",
            type="primary",
            width="stretch",
        ):

            with st.spinner(
                "Engineering time-series features "
                "and running LightGBM..."
            ):

                try:

                    all_predictions, latest_predictions = (
                        predict_uploaded_data(
                            input_df,
                            model,
                        )
                    )

                except Exception as exc:

                    st.error(
                        "Prediction could not be completed."
                    )

                    st.exception(
                        exc
                    )

                    st.stop()

            st.success(
                "Prediction completed successfully! 🚀"
            )

            st.divider()

            st.subheader(
                "🎯 Latest Prediction for Each Engine"
            )

            result_display = latest_predictions[
                [
                    "unit",
                    "cycle",
                    "probability_percent",
                    "risk",
                    "maintenance_decision",
                    "estimated_preventive_cost",
                    "estimated_exposure_if_missed",
                ]
            ].copy()

            result_display.columns = [
                "Engine",
                "Latest Cycle",
                "Failure Probability (%)",
                "Risk",
                "Maintenance Decision",
                "Preventive Cost ($)",
                "Expected AOG Exposure ($)",
            ]

            st.dataframe(
                result_display,
                width="stretch",
                hide_index=True,
            )

            probabilities = latest_predictions[
                "failure_probability"
            ].values

            high_risk_count = int(
                (
                    probabilities
                    >= DECISION_THRESHOLD
                ).sum()
            )

            average_probability = float(
                np.mean(probabilities)
            )

            highest_probability = float(
                np.max(probabilities)
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Engines Analyzed",
                    len(latest_predictions)
                )

            with c2:

                st.metric(
                    "High-Risk Engines",
                    high_risk_count
                )

            with c3:

                st.metric(
                    "Highest Failure Probability",
                    f"{highest_probability * 100:.2f}%"
                )

            st.metric(
                "Average Failure Probability",
                f"{average_probability * 100:.2f}%"
            )

            st.divider()

            st.subheader(
                "💰 Decision-Layer Interpretation"
            )

            total_preventive = int(
                latest_predictions[
                    "estimated_preventive_cost"
                ].sum()
            )

            expected_exposure = float(
                latest_predictions[
                    "estimated_exposure_if_missed"
                ].sum()
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Preventive Maintenance Cost",
                    f"${total_preventive:,.0f}"
                )

            with c2:

                st.metric(
                    "Probability-Weighted AOG Exposure",
                    f"${expected_exposure:,.0f}"
                )

            st.info(
                "The $25,000 figure represents the assumed "
                "cost of a preventive maintenance action. "
                "The $150,000 figure is the assumed AOG/"
                "missed-failure cost used by the project's "
                "business-value simulation."
            )

            st.subheader(
                "📈 Engine Risk Distribution"
            )

            risk_plot = latest_predictions[
                [
                    "unit",
                    "probability_percent"
                ]
            ].set_index(
                "unit"
            )

            st.bar_chart(
                risk_plot
            )

            st.subheader(
                "🔍 Cycle-by-Cycle Predictions"
            )

            cycle_display = all_predictions.copy()

            cycle_display[
                "failure_probability"
            ] *= 100

            cycle_display.columns = [
                "Engine",
                "Cycle",
                "Failure Probability (%)",
                "Risk",
                "Maintenance Decision",
            ]

            st.dataframe(
                cycle_display,
                width="stretch",
                hide_index=True,
            )

            csv_output = latest_predictions.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=csv_output,
                file_name="engine_prediction_results.csv",
                mime="text/csv",
            )

            with st.expander(
                "🧠 What happened behind the scenes?"
            ):

                st.markdown(
                    """
                    **1. Sensor history was uploaded**

                    The system requires `unit`, `cycle`, and
                    21 sensor measurements.

                    **2. Time-series features were created**

                    For every sensor, the system calculates:

                    - rolling mean
                    - rolling standard deviation
                    - cycle-to-cycle difference
                    - degradation slope

                    using the same 15-cycle feature window
                    used during model development.

                    **3. LightGBM generated a probability**

                    The saved classifier produces the probability
                    that the engine belongs to the critical
                    failure-risk class.

                    **4. Decision threshold was applied**

                    A probability of **10% or higher** triggers
                    the project's safety-focused preventive-
                    maintenance decision.

                    **5. Business interpretation**

                    The decision is translated into a preventive
                    maintenance cost assumption of **$25,000**
                    and, separately, a missed-failure/AOG
                    assumption of **$150,000**.
                    """
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "✈️ FLEETORA | AI-Powered Fleet Predictive Maintenance "
    "& Inventory Intelligence"
)

st.caption(
    "Predict. Prevent. Optimize. | "
    "AI + Predictive Analytics + Decision Optimization"
)