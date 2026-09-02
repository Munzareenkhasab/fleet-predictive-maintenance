# ✈️ Fleet Predictive Maintenance & Inventory Optimization

> An AI-driven predictive maintenance and dynamic spare inventory optimization platform for aircraft fleets.

## 🚀 Live Demo

👉 **[Open the Live Dashboard](https://fleet-predictive-maintenance.streamlit.app/)**

---

## 🎯 Project Overview

Unplanned aircraft engine failures can result in expensive Aircraft-on-Ground (AOG) events, emergency maintenance, operational delays, and inefficient spare-parts inventory.

This project develops an end-to-end AI-driven system that predicts high-risk aircraft engines before failure and converts those predictions into maintenance and inventory decisions.

The system combines:

- Machine Learning-based engine failure prediction
- Engine degradation and lifecycle analysis
- Cost-sensitive maintenance decision making
- Dynamic spare inventory optimization
- Fleet-level risk monitoring
- Interactive Streamlit dashboard

The system uses NASA's C-MAPSS turbofan engine degradation dataset as a predictive-maintenance benchmark.

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