import pandas as pd

from src.config import COLUMNS


def load_cmapss_file(path):
    """
    Load NASA CMAPSS whitespace-separated data.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    # Keep only the 26 actual columns
    df = df.iloc[:, :26]

    df.columns = COLUMNS

    return df


def calculate_rul(df):
    """
    Calculate Remaining Useful Life (RUL)
    for every engine cycle.
    """

    df = df.copy()

    max_cycles = (
        df.groupby("unit")["cycle"]
        .max()
        .rename("max_cycle")
    )

    df = df.merge(
        max_cycles,
        left_on="unit",
        right_index=True
    )

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop(columns=["max_cycle"], inplace=True)

    return df