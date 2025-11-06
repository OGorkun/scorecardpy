# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
import re
import ast
from pandas.api.types import is_numeric_dtype


def str_to_list(x):
    return [x] if isinstance(x, str) else x


def check_const_cols(dat):
    unique1_cols = [col for col in dat.columns if dat[col].nunique(dropna=False) == 1]
    if unique1_cols:
        warnings.warn(
            f"There are {len(unique1_cols)} columns with only one unique value, removed: {', '.join(unique1_cols)}",
            stacklevel=2
        )
        dat = dat.drop(columns=unique1_cols)
    return dat


def check_datetime_cols(dat):
    datetime_cols = dat.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    if datetime_cols:
        warnings.warn(
            f"There are {len(datetime_cols)} datetime columns removed: {', '.join(datetime_cols)}",
            stacklevel=2
        )
        dat = dat.drop(columns=datetime_cols)
    return dat


def check_cateCols_uniqueValues(dat, var_skip=None, confirm=True):
    char_cols = [col for col in dat.columns if not is_numeric_dtype(dat[col])]
    if var_skip is not None:
        char_cols = list(set(char_cols) - set(str_to_list(var_skip)))

    too_many_uniques = [col for col in char_cols if dat[col].nunique(dropna=False) >= 50]
    if too_many_uniques:
        msg = (
            f">>> {len(too_many_uniques)} variables have too many unique values:\n"
            f"{', '.join(too_many_uniques)}"
        )
        warnings.warn(msg, stacklevel=2)
        if confirm:
            response = input("Continue binning? (y/n): ").strip().lower()
            if response != 'y':
                raise SystemExit("Stopped by user due to too many unique categorical values.")
    return None


def rep_blank_na(dat):
    if dat.index.duplicated().any():
        dat = dat.reset_index(drop=True)
        warnings.warn("Duplicated index found and reset.", stacklevel=2)

    # Replace blank strings with NaN
    dat = dat.apply(lambda col: col.replace(r'^\s*$', np.nan, regex=True) if col.dtype == 'object' else col)

    # Replace inf/-inf
    num_cols = dat.select_dtypes(include='number').columns
    for col in num_cols:
        if np.isinf(dat[col]).any():
            warnings.warn(f"Infinite values found in '{col}', replaced with -999.", stacklevel=2)
            dat[col] = dat[col].replace([np.inf, -np.inf], -999)
    return dat


def check_y(dat, y, positive):
    if not isinstance(dat, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if dat.shape[1] <= 1:
        raise ValueError("DataFrame must have at least two columns.")

    y = str_to_list(y)
    if len(y) != 1:
        raise ValueError("`y` must contain exactly one column name.")
    y = y[0]

    if y not in dat.columns:
        raise KeyError(f"No '{y}' column found in DataFrame.")

    if dat[y].isna().any():
        warnings.warn(f"NaNs found in '{y}', corresponding rows removed.", stacklevel=2)
        dat = dat.dropna(subset=[y])

    # Convert numeric y safely
    if is_numeric_dtype(dat[y]):
        dat[y] = dat[y].astype('Int64')

    unique_y = dat[y].dropna().unique()
    if len(unique_y) != 2:
        raise ValueError(f"Target '{y}' must have exactly two unique values.")

    # If the two values are already 0/1 (or their string forms), keep them and return
    vals_str = set(str(v) for v in unique_y)
    if vals_str.issubset({'0', '1'}):
        # ensure numeric 0/1
        dat[y] = dat[y].astype(int)
        return dat

    if any(re.search(positive, str(v)) for v in unique_y):
        dat[y] = dat[y].astype(str).apply(lambda x: 1 if re.search(positive, x) else 0)
        warnings.warn(f"Positive value in '{y}' replaced by 1, negative by 0.", stacklevel=2)
    else:
        raise ValueError(f"Positive class not found in '{y}'.")
    return dat


def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step < 0:
        warnings.warn("`print_step` must be a non-negative number. Reset to 1.", stacklevel=2)
        return 1
    return int(print_step)


def x_variable(dat, y, x, var_skip=None):
    y = str_to_list(y)
    if var_skip:
        y += str_to_list(var_skip)
    x_all = list(set(dat.columns) - set(y))

    if x is None:
        return x_all

    x = str_to_list(x)
    x_not_found = set(x) - set(x_all)
    if x_not_found:
        warnings.warn(f"{len(x_not_found)} variables not found and removed: {', '.join(x_not_found)}", stacklevel=2)
    x = list(set(x) & set(x_all))
    return x


def check_breaks_list(breaks_list, xs):
    if breaks_list is None:
        return None
    if isinstance(breaks_list, str):
        # Normalize representation
        breaks_list = breaks_list.replace("[inf]", "[np.inf]")
        # Safe evaluation allowing only numpy
        breaks_list = eval(breaks_list, {"np": np, "__builtins__": {}})
    if not isinstance(breaks_list, dict):
        raise TypeError("`breaks_list` must be a dictionary.")
    # Filter to only xs variables
    breaks_list = {k: v for k, v in breaks_list.items() if k in xs}
    return breaks_list


def check_special_values(dt, special_values, xs):
    if special_values is None:
        return None
    if isinstance(special_values, list):
        sv_dict = {i: [v for v in special_values if v in dt[i].unique()] for i in xs}
        return {k: v for k, v in sv_dict.items() if v}
    if not isinstance(special_values, dict):
        raise TypeError("`special_values` must be a list or dictionary.")
    return special_values


def check_max_bin_num(dt, xs, min_perc_fine_bin, bin_decimals):
    var_one_bin = []
    for i in xs:
        dt_i = dt[i]
        if is_numeric_dtype(dt_i):
            dt_i = dt_i.round(bin_decimals)
        if dt_i.value_counts(normalize=True).max() > (1 - min_perc_fine_bin):
            var_one_bin.append(i)

    if var_one_bin:
        warnings.warn(
            f"{len(var_one_bin)} variables cannot be split into bins and will be removed: {', '.join(var_one_bin)}",
            stacklevel=2
        )
    return var_one_bin
