import math
import numpy as np


class InventoryOptimizer:

    def __init__(
        self,
        lead_time=5,
        service_level=0.95,
        initial_stock=10
    ):
        self.lead_time = lead_time
        self.service_level = service_level
        self.stock = initial_stock

        self.z_values = {
            0.90: 1.28,
            0.95: 1.65,
            0.975: 1.96,
            0.99: 2.33
        }

        self.z = self.z_values.get(
            service_level,
            1.65
        )

    def calculate_target_stock(
        self,
        failure_probabilities
    ):

        probabilities = np.asarray(
            failure_probabilities
        )

        # Expected spare demand
        expected_demand = probabilities.sum()

        # Demand variance adjusted for lead time
        demand_variance = (
            np.sum(
                probabilities
                * (1 - probabilities)
            )
            * self.lead_time
        )

        demand_std = math.sqrt(
            max(demand_variance, 0)
        )

        # Safety stock
        safety_stock = (
            self.z
            * demand_std
        )

        # Final inventory target
        target_stock = math.ceil(
            expected_demand
            + safety_stock
        )

        return {
            "expected_demand": expected_demand,
            "demand_std": demand_std,
            "safety_stock": safety_stock,
            "target_stock": target_stock
        }