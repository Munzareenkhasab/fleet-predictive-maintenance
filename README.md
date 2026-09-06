# ✈️ Fleetora

> **AI-Powered Fleet Predictive Maintenance & Inventory Optimization Platform**

Fleetora is an end-to-end AI-driven decision-support platform designed to predict aircraft engine failure risk, support proactive maintenance decisions, estimate business impact, and dynamically optimize spare-parts inventory.

---

## 🚀 Live Demo

👉 **[Open Fleetora Live Dashboard](https://fleet-predictive-maintenance.streamlit.app/)**

---

## 🎯 Problem Statement

Unplanned aircraft engine failures can lead to:

- Expensive Aircraft-on-Ground (AOG) events
- Emergency maintenance
- Flight delays and operational disruption
- High reactive maintenance costs
- Poor spare-parts planning
- Excess inventory or unexpected stockouts

Traditional maintenance approaches often react to failures after they occur or rely on fixed maintenance schedules.

Fleetora tackles this problem by combining **predictive analytics, business decision-making, and inventory optimization** into one platform.

---

## 💡 What Fleetora Does

Fleetora follows an end-to-end pipeline:

```text
Engine Sensor Data
        ↓
Time-Series Feature Engineering
        ↓
LightGBM Failure Prediction
        ↓
Failure Probability
        ↓
Decision Threshold
        ↓
Preventive Maintenance Decision
        ↓
Business Cost Analysis
        ↓
Fleet Risk Aggregation
        ↓
Probabilistic Spare Demand Estimation
        ↓
Safety Stock & Target Inventory
        ↓
Interactive Fleet Dashboard