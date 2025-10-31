# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .condition_fun import check_y, x_variable  # ensure these exist


def iv(dt, y, x=None, positive='bad|1', order=True):
    """
    Calculate Information Value (IV) for multiple features.
    """
    dt = dt.copy(deep=True)

    # Normalize input types
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str):
        x = [x]

    # Keep only relevant columns
    if x is not None:
        dt = dt[y + x]

    # Standardize target variable
    dt = check_y(dt, y, positive)

    # Identify predictor variables
    xs = x_variable(dt, y, x)

    # Calculate IV for each variable
    iv_list = [
        {"variable": col, "info_value": iv_xy(dt[col], dt[y[0]])}
        for col in xs
    ]

    iv_df = pd.DataFrame(iv_list)

    # Sort descending by IV
    if order:
        iv_df = iv_df.sort_values("info_value", ascending=False)

    return iv_df


def iv_xy(x, y):
    """Calculate IV for a single variable vs binary target."""
    df = pd.DataFrame({"x": x, "y": y}).fillna("missing")
    df["x"] = df["x"].astype("string")

    # Group and count
    counts = df.groupby(["x", "y"], observed=True).size().unstack(fill_value=0)
    if counts.shape[1] == 1:  # if only one class exists
        counts = counts.assign(**{1 - counts.columns[0]: 0})
    counts.columns = ["good", "bad"] if 0 in counts.columns else ["bad", "good"]

    # Avoid division by zero
    counts = counts.replace(0, 1e-6)

    # Compute distributions
    total_good = counts["good"].sum()
    total_bad = counts["bad"].sum()
    counts["DistrGood"] = counts["good"] / total_good
    counts["DistrBad"] = counts["bad"] / total_bad

    # IV calculation
    counts["iv"] = (counts["DistrBad"] - counts["DistrGood"]) * np.log(
        counts["DistrBad"] / counts["DistrGood"]
    )

    return counts["iv"].sum()


def iv_01(good, bad):
    """Calculate total IV from good/bad counts."""
    df = pd.DataFrame({"good": good, "bad": bad}).replace(0, 1e-6)
    total_good = df["good"].sum()
    total_bad = df["bad"].sum()
    df["DistrGood"] = df["good"] / total_good
    df["DistrBad"] = df["bad"] / total_bad
    df["iv"] = (df["DistrBad"] - df["DistrGood"]) * np.log(df["DistrBad"] / df["DistrGood"])
    return df["iv"].sum()


def miv_01(good, bad):
    """Return IV per bin."""
    df = pd.DataFrame({"good": good, "bad": bad}).replace(0, 1e-6)
    total_good = df["good"].sum()
    total_bad = df["bad"].sum()
    df["DistrGood"] = df["good"] / total_good
    df["DistrBad"] = df["bad"] / total_bad
    df["iv"] = (df["DistrBad"] - df["DistrGood"]) * np.log(df["DistrBad"] / df["DistrGood"])
    return df["iv"]


def woe_01(good, bad):
    """Calculate WOE (Weight of Evidence) for each bin."""
    df = pd.DataFrame({"good": good, "bad": bad}).replace(0, 1e-6)
    total_good = df["good"].sum()
    total_bad = df["bad"].sum()
    df["DistrGood"] = df["good"] / total_good
    df["DistrBad"] = df["bad"] / total_bad
    df["woe"] = np.log(df["DistrGood"] / df["DistrBad"])
    return df["woe"]
