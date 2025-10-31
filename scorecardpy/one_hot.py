import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from .condition_fun import *


def one_hot(
    dt,
    cols_skip=None,
    cols_encode=None,
    nacol_rm=False,
    replace_na=-1,
    category_to_integer=False
):
    """
    One Hot Encoding
    ------
    Performs one-hot encoding on categorical variables. Useful when models
    require numeric input instead of WOE-transformed variables.

    Parameters
    ----------
    dt : pd.DataFrame
        Input dataframe.
    cols_skip : list or str, optional
        Columns to skip from encoding.
    cols_encode : list or str, optional
        Columns to encode; if None, automatically detect all categorical.
    nacol_rm : bool, default False
        If True, remove dummy columns indicating NA values.
    replace_na : {int, float, str, None}, default -1
        Value or method ('mean', 'median') to replace missing values.
    category_to_integer : bool, default False
        If True, convert category dtypes to integer labels before encoding.

    Returns
    -------
    pd.DataFrame
        One-hot encoded dataframe.
    """
    # Convert skip and encode columns to list
    cols_skip = str_to_list(cols_skip)
    cols_encode = str_to_list(cols_encode)

    dt = dt.copy(deep=True)

    # Convert categorical columns to integers if requested
    if category_to_integer:
        cols_cate = dt.select_dtypes(include=["category"]).columns.tolist()
        if cols_skip:
            cols_cate = [c for c in cols_cate if c not in cols_skip]
        for c in cols_cate:
            dt[c] = pd.factorize(dt[c], sort=True)[0]

    # Determine columns to encode
    if cols_encode is None:
        cols_encode = [
            c for c in dt.columns
            if not is_numeric_dtype(dt[c]) and not is_datetime64_any_dtype(dt[c])
        ]
    else:
        cols_encode = x_variable(dt, y=cols_skip, x=cols_encode)

    # Remove skipped columns
    if cols_skip:
        cols_encode = [c for c in cols_encode if c not in cols_skip]

    # Perform one-hot encoding
    if not cols_encode:
        dt_new = dt
    else:
        temp_dt = pd.get_dummies(dt[cols_encode], dummy_na=not nacol_rm, dtype=float)
        rm_cols_nan1 = [
            c for c in temp_dt.columns
            if temp_dt[c].nunique(dropna=False) == 1 and "_nan" in c
        ]
        temp_dt = temp_dt.drop(columns=rm_cols_nan1, errors="ignore")
        dt_new = pd.concat([dt.drop(columns=cols_encode), temp_dt], axis=1)

    # Helper function for missing value replacement
    def rep_na(x):
        if not x.isna().any():
            return x
        if isinstance(replace_na, (int, float)):
            fill_val = replace_na
        elif replace_na == "mean" and is_numeric_dtype(x):
            fill_val = x.mean()
        elif replace_na == "median" and is_numeric_dtype(x):
            fill_val = x.median()
        else:
            fill_val = -1
        return x.fillna(fill_val)

    # Replace missing values globally if needed
    if replace_na is not None:
        fill_cols = [c for c in dt_new.columns if not (cols_skip and c in cols_skip)]
        dt_new[fill_cols] = dt_new[fill_cols].apply(rep_na)

    return dt_new
