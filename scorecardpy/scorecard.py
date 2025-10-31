# -*- coding: utf-8 -*-
"""
scorecard.py
Modernized version for pandas>=2.2, numpy>=2.0, sklearn>=1.5
"""

import pandas as pd
import numpy as np
import re
import warnings
from .condition_fun import *
from .woebin import woepoints_ply1


# ======================================================
# Base A/B coefficient calculation
# ======================================================
def ab(points0=540, odds0=1 / 9, pdo=40):
    """Calculate scorecard scaling coefficients a and b."""
    b = pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    return {'a': a, 'b': b}


# ======================================================
# Scorecard creation
# ======================================================
def scorecard(
    bins,
    model,
    xcolumns,
    points0=540,
    odds0=1 / 9,
    pdo=40,
    start_zero=True,
    digits=0
):
    """
    Create a scorecard from a fitted LogisticRegression model and binning data.
    """
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a, b = aabb['a'], aabb['b']

    # combine bins if dict
    if isinstance(bins, dict):
        bins = pd.concat(bins.values(), ignore_index=True)

    # clean column names
    xs = [re.sub('_woe$', '', c) for c in xcolumns]

    # model coefficients
    coef_df = (
        pd.Series(model.coef_.ravel(), index=np.array(xs))
        .loc[lambda s: s != 0]
    )

    basepoints = a - b * model.intercept_.ravel()[0]

    card = {}
    base_df = pd.DataFrame(
        {'variable': ["basepoints"], 'bin': [np.nan], 'points': [round(basepoints, digits)]}
    )
    card['basepoints'] = base_df

    # per-variable score components
    for var in coef_df.index:
        sub = bins.loc[bins['variable'] == var, ['variable', 'bin', 'woe']].copy()
        sub['points'] = np.round(-b * sub['woe'] * coef_df[var], digits)
        if start_zero:
            min_points = sub['points'].min()
            sub['points'] -= min_points
            card['basepoints'].loc[:, 'points'] += min_points
        card[var] = sub[['variable', 'bin', 'points']]

    return card


# ======================================================
# Apply scorecard to dataset
# ======================================================
def scorecard_ply(
    dt,
    card,
    only_total_score=True,
    print_step=0,
    replace_blank_na=True,
    var_kp=None
):
    """
    Apply a scorecard to a dataset to compute total and/or variable-level scores.
    """
    dt = dt.copy()

    if replace_blank_na:
        dt = rep_blank_na(dt)

    print_step = check_print_step(print_step)

    # Combine dict of cards if needed
    if isinstance(card, dict):
        card_df = pd.concat(card.values(), ignore_index=True)
    elif isinstance(card, pd.DataFrame):
        card_df = card.copy()
    else:
        raise ValueError("`card` must be a dict or DataFrame")

    # variables used for scoring
    xs = card_df.loc[card_df['variable'] != 'basepoints', 'variable'].unique().tolist()
    dat = dt.loc[:, list(set(dt.columns) - set(xs))].copy()

    # Apply scoring variable by variable
    for i, x_i in enumerate(xs, start=1):
        if print_step and i % print_step == 0:
            print(f"{i}/{len(xs)}: processing {x_i}")
        cardx = card_df.loc[card_df['variable'] == x_i].copy()
        dtx_points = woepoints_ply1(dt[[x_i]], cardx, x_i, woe_points="points")
        dat = pd.concat([dat, dtx_points], axis=1)

    # Basepoints
    card_basepoints = card_df.loc[card_df['variable'] == 'basepoints', 'points']
    base_val = card_basepoints.iloc[0] if not card_basepoints.empty else 0

    # Collect score columns safely
    score_cols = [f"{x}_points" for x in xs if f"{x}_points" in dat.columns]
    dat_score = dat[score_cols].copy()
    dat_score['score'] = base_val + dat_score.sum(axis=1, numeric_only=True)

    # Keep only total score if requested
    if only_total_score:
        dat_score = dat_score[['score']]

    # Keep key variables if required
    if var_kp is not None:
        var_kp = [var_kp] if isinstance(var_kp, str) else list(var_kp)
        var_kp_existing = [v for v in var_kp if v in dt.columns]
        if missing := list(set(var_kp) - set(var_kp_existing)):
            warnings.warn(
                f"The following var_kp variables are missing from input data and will be ignored: {missing}"
            )
        if var_kp_existing:
            dat_score = pd.concat([dt[var_kp_existing], dat_score], axis=1)

    return dat_score
