import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    FAILURE_THRESHOLD,
    WINDOW_SIZE
)

from src.data_loader import (
    load_cmapss_file,
    calculate_rul
)

from src.features import build_features

from src.model import FailureClassifier

from src.inventory import InventoryOptimizer

from src.kpi import calculate_inventory_kpis

from src.inventory_simulation import simulate_inventory_fulfillment

from src.value_layer import calculate_business_value


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("Loading CMAPSS data...")

df = load_cmapss_file(TRAIN_FILE)
df = calculate_rul(df)

print(f"Dataset shape: {df.shape}")

# ============================================================
# LOAD UNSEEN TEST DATA
# ============================================================

print("\nLoading unseen test data...")

test_df = load_cmapss_file(TEST_FILE)

# Load the true RUL for each test engine
test_rul = pd.read_csv(
    RUL_FILE,
    sep=r"\s+",
    header=None
)

test_rul = test_rul.iloc[:, 0].values

print(f"Test dataset shape: {test_df.shape}")
print(f"Number of test engines: {len(test_rul)}")

# ============================================================
# BUILD FEATURES FOR TEST DATA
# ============================================================

print("\nBuilding test features...")

test_df = build_features(
    test_df,
    window=WINDOW_SIZE
)

print(f"Test feature dataset shape: {test_df.shape}")
# ============================================================
# RECONSTRUCT TRUE RUL FOR TEST DATA
# ============================================================

test_rul_df = pd.DataFrame({
    "unit": range(1, len(test_rul) + 1),
    "final_RUL": test_rul
})

test_df = test_df.merge(
    test_rul_df,
    on="unit",
    how="left"
)

max_cycles = test_df.groupby("unit")["cycle"].transform("max")

test_df["RUL"] = (
    test_df["final_RUL"]
    + (max_cycles - test_df["cycle"])
)

test_df["failure_risk"] = (
    test_df["RUL"] <= FAILURE_THRESHOLD
).astype(int)

print("\nTest RUL reconstructed successfully.")
print("Test failure label distribution:")
print(test_df["failure_risk"].value_counts())

# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================

print("\nBuilding features...")

df = build_features(
    df,
    window=WINDOW_SIZE
)

print(f"Feature dataset shape: {df.shape}")


# ============================================================
# STEP 3 — CREATE FAILURE LABEL
# ============================================================

df["failure_risk"] = (
    df["RUL"] <= FAILURE_THRESHOLD
).astype(int)

print("\nFailure label distribution:")
print(df["failure_risk"].value_counts())


# ============================================================
# STEP 4 — SELECT FEATURES
# ============================================================

exclude_columns = [
    "unit",
    "cycle",
    "RUL",
    "failure_risk"
]

feature_columns = [
    column
    for column in df.columns
    if column not in exclude_columns
]

X = df[feature_columns]
y = df["failure_risk"]


# ============================================================
# STEP 5 — ENGINE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

print("\nSplitting engines...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

os.makedirs("models", exist_ok=True)

train_idx, validation_idx = next(
    splitter.split(
        X,
        y,
        groups=df["unit"]
    )
)

X_train = X.iloc[train_idx]
X_validation = X.iloc[validation_idx]

y_train = y.iloc[train_idx]
y_validation = y.iloc[validation_idx]

train_df = df.iloc[train_idx].copy()
validation_df = df.iloc[validation_idx].copy()

print(
    f"Training engines: "
    f"{train_df['unit'].nunique()}"
)

print(
    f"Validation engines: "
    f"{validation_df['unit'].nunique()}"
)


# ============================================================
# STEP 6 — TRAIN FAILURE CLASSIFIER
# ============================================================

print("\nTraining failure classifier...")

classifier = FailureClassifier()

classifier.train(
    X_train,
    y_train
)
classifier.save("models/failure_classifier.pkl")

print("Failure classifier trained successfully.")

# ============================================================
# STEP 6B — MODEL EVALUATION
# ============================================================

print("\n")
print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

# Predict probabilities and classes
validation_probabilities = (
    classifier.predict_probability(X_validation)
)

validation_predictions = (
    validation_probabilities >= 0.10
).astype(int)

# Confusion matrix
cm = confusion_matrix(
    y_validation,
    validation_predictions
)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions
    )
)

# Additional metrics
precision = precision_score(
    y_validation,
    validation_predictions
)

recall = recall_score(
    y_validation,
    validation_predictions
)

f1 = f1_score(
    y_validation,
    validation_predictions
)

roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)

print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")
print(f"ROC-AUC   : {roc_auc:.3f}")



# ============================================================
# FINAL UNSEEN TEST-SET EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("UNSEEN TEST-SET EVALUATION")
print("=" * 80)

X_test = test_df[feature_columns]
y_test = test_df["failure_risk"]

# Predict failure probabilities
test_probabilities = classifier.predict_probability(X_test)

# Use our selected safety-focused threshold
test_predictions = (
    test_probabilities >= 0.10
).astype(int)

# Metrics
test_cm = confusion_matrix(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

print("\nConfusion Matrix:")
print(test_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print(f"Precision : {test_precision:.3f}")
print(f"Recall    : {test_recall:.3f}")
print(f"F1 Score  : {test_f1:.3f}")
print(f"ROC-AUC   : {test_auc:.3f}")

# ============================================================
# BUSINESS VALUE - UNSEEN TEST SET
# ============================================================

print("\n" + "=" * 80)
print("BUSINESS VALUE - UNSEEN TEST SET")
print("=" * 80)

business_value = calculate_business_value(
    actual=y_test,
    predicted=test_probabilities,
    threshold=0.10
)

print(
    f"Reactive maintenance cost : "
    f"${business_value['reactive_cost']:,.2f}"
)

print(
    f"AI-assisted cost          : "
    f"${business_value['ai_cost']:,.2f}"
)

print(
    f"Simulated cost reduction  : "
    f"${business_value['cost_difference']:,.2f}"
)

print(
    f"Simulated savings         : "
    f"{business_value['savings_percentage']:.2f}%"
)

print(
    f"Preventive actions        : "
    f"{business_value['preventive_actions']}"
)

print(
    f"True critical cases       : "
    f"{business_value['true_failures_detected']}"
)

print(
    f"False alarms              : "
    f"{business_value['false_alarms']}"
)

print(
    f"Missed critical cases    : "
    f"{business_value['missed_failures']}"
)

print(
    f"Preventive maintenance   : "
    f"${business_value['preventive_cost']:,.2f}"
)

print(
    f"Simulated AOG cost       : "
    f"${business_value['aog_cost']:,.2f}"
)


# ============================================================
# STEP 7 — CREATE INVENTORY OPTIMIZER
# ============================================================

optimizer = InventoryOptimizer(
    lead_time=5,
    service_level=0.95,
    initial_stock=10
)


# ============================================================
# STEP 8 — SIMULATE FLEET AT MULTIPLE CYCLE POINTS
# ============================================================

print("\nStarting fleet simulation...")

simulation_cycles = [
    50,
    75,
    100,
    125,
    150,
    175,
    200
]

simulation_results = []


for cycle in simulation_cycles:

    print(
        f"\nSimulating fleet at cycle {cycle}..."
    )

    # Select engine records close to this lifecycle point
    # Select the latest information available
    # at or before the simulation cycle
    cycle_data = (
       validation_df[
           validation_df["cycle"] <= cycle
       ]
       .sort_values(
          ["unit", "cycle"]
       )
       .groupby("unit")
       .tail(1)
       .copy()
    )

    if len(cycle_data) == 0:
       continue

    # --------------------------------------------------------
    # Predict failure probabilities
    # --------------------------------------------------------

    cycle_features = (
        cycle_data[feature_columns]
    )

    probabilities = (
        classifier.predict_probability(
            cycle_features
        )
    )

    cycle_data["failure_probability"] = (
        probabilities
    )

    # --------------------------------------------------------
    # Inventory calculation
    # --------------------------------------------------------

    inventory = (
        optimizer.calculate_target_stock(
            probabilities
        )
    )

    # --------------------------------------------------------
    # Fleet statistics
    # --------------------------------------------------------

    high_risk_engines = (
        cycle_data[
            cycle_data["failure_probability"] >= 0.10
        ]
    )

    average_risk = (
        cycle_data["failure_probability"].mean()
    )

    average_rul = (
        cycle_data["RUL"].mean()
    )

    simulation_results.append({
        "cycle": cycle,
        "engines_observed": len(cycle_data),
        "average_rul": average_rul,
        "average_failure_probability": average_risk,
        "high_risk_engines": len(high_risk_engines),
        "expected_demand": inventory["expected_demand"],
        "safety_stock": inventory["safety_stock"],
        "target_stock": inventory["target_stock"]
    })


# ============================================================
# STEP 9 — DISPLAY SIMULATION RESULTS
# ============================================================

results_df = pd.DataFrame(
    simulation_results
)

print("\n")
print("=" * 80)
print("DYNAMIC FLEET INVENTORY SIMULATION")
print("=" * 80)

if len(results_df) > 0:

    print(
        results_df.to_string(
            index=False,
            formatters={
                "average_rul":
                    "{:.2f}".format,
                "average_failure_probability":
                    "{:.3f}".format,
                "expected_demand":
                    "{:.2f}".format,
                "safety_stock":
                    "{:.2f}".format
            }
        )
    )

else:

    print(
        "No simulation results generated."
    )


# ============================================================
# STEP 10 — INVENTORY TREND
# ============================================================

if len(results_df) > 0:

    highest_stock_row = (
        results_df.loc[
            results_df["target_stock"].idxmax()
        ]
    )

    lowest_stock_row = (
        results_df.loc[
            results_df["target_stock"].idxmin()
        ]
    )

    print("\n")
    print("=" * 80)
    print("INVENTORY TREND")
    print("=" * 80)

    print(
        f"Lowest recommended stock : "
        f"{lowest_stock_row['target_stock']} "
        f"units at cycle "
        f"{int(lowest_stock_row['cycle'])}"
    )

    print(
        f"Highest recommended stock: "
        f"{highest_stock_row['target_stock']} "
        f"units at cycle "
        f"{int(highest_stock_row['cycle'])}"
    )


# ============================================================
# STEP 11 — SAVE SIMULATION RESULTS
# ============================================================

results_df.to_csv(
    "fleet_simulation_results.csv",
    index=False
)

# ============================================================
# STEP 12 — CALCULATE BUSINESS KPIs
# ============================================================

print("\n")
print("=" * 80)
print("BUSINESS & INVENTORY KPIs")
print("=" * 80)

kpis = calculate_inventory_kpis(
    results_df
)

print(
    f"Total inventory carrying cost : "
    f"${kpis['total_inventory_cost']:,.2f}"
)

print(
    f"Average inventory              : "
    f"{kpis['average_inventory']:.2f} units"
)

print(
    f"Peak inventory                 : "
    f"{kpis['peak_inventory']} units"
)

print(
    f"Minimum inventory              : "
    f"{kpis['minimum_inventory']} units"
)

print(
    f"Inventory turnover             : "
    f"{kpis['inventory_turnover']:.2f}"
)

print(
    f"Service-level proxy            : "
    f"{kpis['service_level'] * 100:.2f}%"
)

print(
    f"Average fleet failure risk     : "
    f"{kpis['average_fleet_risk'] * 100:.2f}%"
)

print(
    f"Peak fleet failure risk        : "
    f"{kpis['peak_fleet_risk'] * 100:.2f}%"
)

print(
    f"Total expected demand          : "
    f"{kpis['total_expected_demand']:.2f}"
)

print(
    "\nSimulation results saved to "
    "fleet_simulation_results.csv"
)

print(
    "\nFleet simulation completed successfully. 🚀"
)


# ============================================================
# STEP 13 — INVENTORY FULFILLMENT SIMULATION
# ============================================================

print("\n")
print("=" * 80)
print("INVENTORY FULFILLMENT SIMULATION")
print("=" * 80)

fulfillment_df, fulfillment_kpis = (
    simulate_inventory_fulfillment(
        results_df,
        initial_stock=10
    )
)


print(
    f"Total simulated demand       : "
    f"{fulfillment_kpis['total_demand']}"
)

print(
    f"Fulfilled demand             : "
    f"{fulfillment_kpis['fulfilled_demand']}"
)

print(
    f"Total stockouts               : "
    f"{fulfillment_kpis['total_stockouts']}"
)

print(
    f"Fill rate                     : "
    f"{fulfillment_kpis['fill_rate'] * 100:.2f}%"
)

print(
    f"Stockout rate                 : "
    f"{fulfillment_kpis['stockout_rate'] * 100:.2f}%"
)

print(
    f"Inventory used                : "
    f"{fulfillment_kpis['inventory_used']}"
)

print(
    f"Average inventory             : "
    f"{fulfillment_kpis['average_inventory']:.2f}"
)

print(
    f"Inventory turnover            : "
    f"{fulfillment_kpis['inventory_turnover']:.2f}"
)

print(
    f"Total carrying cost           : "
    f"${fulfillment_kpis['total_carrying_cost']:,.2f}"
)


# Save fulfillment simulation

fulfillment_df.to_csv(
    "inventory_fulfillment_results.csv",
    index=False
)

print(
    "\nFulfillment results saved to "
    "inventory_fulfillment_results.csv"
)

print(
    "\nInventory fulfillment simulation completed. 🚀"
)