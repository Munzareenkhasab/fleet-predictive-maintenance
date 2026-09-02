import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def add_rolling_features(df, window=15):

    df = df.copy()
    df = df.sort_values(["unit", "cycle"])

    grouped = df.groupby("unit", group_keys=False)

    for sensor in SENSOR_COLUMNS:

        # Recent average
        df[f"{sensor}_rolling_mean"] = (
            grouped[sensor]
            .transform(
                lambda x: x.rolling(
                    window,
                    min_periods=1
                ).mean()
            )
        )

        # Recent variability
        df[f"{sensor}_rolling_std"] = (
            grouped[sensor]
            .transform(
                lambda x: x.rolling(
                    window,
                    min_periods=2
                ).std()
            )
            .fillna(0)
        )

        # Change from previous cycle
        df[f"{sensor}_diff"] = (
            grouped[sensor]
            .diff()
            .fillna(0)
        )

    return df


def add_degradation_slopes(df, window=15):

    df = df.copy()

    for sensor in SENSOR_COLUMNS:

        slope_column = f"{sensor}_slope"
        slopes = []

        for _, group in df.groupby("unit"):

            values = group[sensor].values
            cycles = group["cycle"].values

            local_slopes = np.zeros(len(group))

            for i in range(len(group)):

                start = max(
                    0,
                    i - window + 1
                )

                x = cycles[start:i + 1]
                y = values[start:i + 1]

                if len(x) >= 2:
                    slope = np.polyfit(
                        x,
                        y,
                        1
                    )[0]
                else:
                    slope = 0.0

                local_slopes[i] = slope

            slopes.extend(local_slopes)

        df[slope_column] = slopes

    return df


def build_features(df, window=15):

    df = add_rolling_features(
        df,
        window
    )

    df = add_degradation_slopes(
        df,
        window
    )

    return df