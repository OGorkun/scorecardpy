# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .germancredit import check_const_cols, check_datetime_cols, rep_blank_na, x_variable

# -------------------------
# Information Entropy (ID3)
# -------------------------

def ie(dt, y='creditability', x=None, order=True):
    """Calculate information entropy for all x variables."""
    dt = check_datetime_cols(check_const_cols(dt))
    dt = rep_blank_na(dt)
    x = x_variable(dt, y, x)
    
    results = pd.DataFrame({
        'variable': x,
        'info_ent': [ie_xy(dt[i], dt[y]) for i in x]
    })
    if order:
        results = results.sort_values(by='info_ent', ascending=False, ignore_index=True)
    return results


def ie_xy(x, y):
    """Entropy of variable x with respect to y."""
    df_xy = (
        pd.DataFrame({'x': x, 'y': y})
        .groupby(['x', 'y'], dropna=False)
        .size()
        .reset_index(name='xy_N')
    )
    df_xy['x_N'] = df_xy.groupby('x')['xy_N'].transform('sum')
    df_xy['p'] = df_xy['xy_N'] / df_xy['x_N']
    df_xy['enti'] = -df_xy['p'] * np.log2(df_xy['p'].clip(lower=1e-12))
    
    df_enti = (
        df_xy.groupby('x', dropna=False)
        .agg({'xy_N': 'sum', 'enti': 'sum'})
        .rename(columns={'xy_N': 'x_N', 'enti': 'ent'})
        .replace(np.nan, 0)
    )
    df_enti['xN_distr'] = df_enti['x_N'] / df_enti['x_N'].sum()
    return (df_enti['ent'] * df_enti['xN_distr']).sum()


def ie_01(good, bad):
    """Entropy based on good/bad counts."""
    df = pd.DataFrame({'good': good, 'bad': bad})
    df['p0'] = df['good'] / (df['good'] + df['bad']).replace(0, np.nan)
    df['p1'] = df['bad'] / (df['good'] + df['bad']).replace(0, np.nan)
    df['enti'] = -(df['p0'] * np.log2(df['p0'].clip(lower=1e-12)) + 
                   df['p1'] * np.log2(df['p1'].clip(lower=1e-12)))
    df['xN_distr'] = df['good'] + df['bad']
    df['xN_distr'] /= df['xN_distr'].sum()
    return (df['enti'] * df['xN_distr']).sum()


# -------------------------
# Gini Impurity (CART)
# -------------------------

def ig(dt, y, x=None, order=True):
    """Calculate Gini impurity for all x variables."""
    dt = check_datetime_cols(check_const_cols(dt))
    dt = rep_blank_na(dt)
    x = x_variable(dt, y, x)
    
    results = pd.DataFrame({
        'variable': x,
        'gini_impurity': [ig_xy(dt[i], dt[y]) for i in x]
    })
    if order:
        results = results.sort_values(by='gini_impurity', ascending=False, ignore_index=True)
    return results


def ig_xy(x, y):
    """Gini impurity of variable x with respect to y."""
    df_xy = (
        pd.DataFrame({'x': x, 'y': y})
        .groupby(['x', 'y'], dropna=False)
        .size()
        .reset_index(name='xy_N')
    )
    df_xy['x_N'] = df_xy.groupby('x')['xy_N'].transform('sum')
    df_xy['p'] = df_xy['xy_N'] / df_xy['x_N']
    
    df_gini = (
        df_xy.groupby('x', dropna=False)
        .agg({'xy_N': 'sum', 'p': lambda x: 1 - np.sum(x ** 2)})
        .rename(columns={'xy_N': 'x_N', 'p': 'ent'})
        .replace(np.nan, 0)
    )
    df_gini['xN_distr'] = df_gini['x_N'] / df_gini['x_N'].sum()
    return (df_gini['ent'] * df_gini['xN_distr']).sum()


def ig_01(good, bad):
    """Gini impurity based on good/bad counts."""
    df = pd.DataFrame({'good': good, 'bad': bad})
    df['p0'] = df['good'] / (df['good'] + df['bad']).replace(0, np.nan)
    df['p1'] = df['bad'] / (df['good'] + df['bad']).replace(0, np.nan)
    df['bin_ig'] = 1 - (df['p0'] ** 2 + df['p1'] ** 2)
    df['xN_distr'] = (df['good'] + df['bad']) / (df['good'] + df['bad']).sum()
    return (df['bin_ig'] * df['xN_distr']).sum()
