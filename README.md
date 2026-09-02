# ✈️ Fleet Predictive Maintenance & Inventory Optimization

An AI-driven predictive maintenance system that identifies high-risk aircraft engines before failure and dynamically optimizes spare-component inventory based on fleet degradation risk.

## 🎯 Project Overview

Unplanned aircraft engine failures can result in expensive Aircraft-on-Ground (AOG) events, emergency maintenance, operational delays, and inefficient spare-parts inventory.

This project combines:

- Machine Learning-based engine failure prediction
- Engine degradation and lifecycle analysis
- Cost-sensitive maintenance decision making
- Dynamic spare inventory optimization
- Fleet-level risk monitoring
- Interactive Streamlit dashboard

The system uses NASA's C-MAPSS turbofan engine degradation dataset as a benchmark for predictive maintenance modeling.

---

## 🏗️ System Architecture

```text
NASA C-MAPSS Dataset
        ↓
Data Loading & RUL Calculation
        ↓
Feature Engineering
 ├── Rolling Mean
 ├── Rolling Standard Deviation
 ├── Sensor Differences
 └── Degradation Slopes
        ↓
LightGBM Failure Classifier
        ↓
Failure Probability
        ↓
Maintenance Decision Layer
 ├── Preventive Maintenance
 └── AOG Risk
        ↓
Fleet-Level Risk Aggregation
        ↓
Dynamic Inventory Optimizer
 ├── Expected Spare Demand
 ├── Demand Uncertainty
 ├── Safety Stock
 └── Target Inventory
        ↓
Streamlit Dashboard
        ↓
Operational & Business KPIs