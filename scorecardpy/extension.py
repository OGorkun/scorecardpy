# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


def var_types(df, var_skip=None):
    """Return lists of categorical and numerical variable names."""
    var_skip = var_skip or []
    var_cat, var_num = [], []

    for var, dtype in df.dtypes.items():
        if var in var_skip:
            continue
        if dtype.name in ['category', 'object'] or df[var].nunique(dropna=False) <= 10:
            var_cat.append(var)
        else:
            var_num.append(var)

    return var_cat, var_num


def miss_heatmap(df, var_skip=None, save_to=None, fig_width=10, fig_height=6):
    """Generate a heatmap showing missing values across columns."""
    var_cat, var_num = var_types(df, var_skip)
    cols = var_cat + var_num
    percent_missing = df[cols].isna().mean() * 100

    percent_missing = percent_missing.sort_values(ascending=False)
    df_missing = df[percent_missing.index]

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(df_missing.isna().T, cmap="YlGnBu", cbar_kws={'label': 'Missing Data'})
    plt.title("Missing Data Heatmap", fontsize=14)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def hhi(series):
    """Compute Herfindahl-Hirschman Index for a categorical variable."""
    _, counts = np.unique(series.dropna().astype(str), return_counts=True)
    shares = counts / counts.sum()
    return np.sum(shares ** 2)


def expl_analysis(df, var_skip=None, special_values=None,
                  hhi_low=0.05, hhi_high=0.95,
                  min_share=0.05, max_share=0.9,
                  save_to=None):
    """Perform exploratory analysis on categorical and numeric variables."""
    special_values = special_values or []
    var_cat, var_num = var_types(df, var_skip)

    # --- Categorical variables summary ---
    cat_summaries = []
    for var in var_cat:
        ser = df[var]
        miss = ser.isna().sum() + (ser.astype(str).str.strip() == '').sum()
        miss_share = miss / len(df)

        hhi_val = round(hhi(ser), 4)
        shares = ser.value_counts(normalize=True, dropna=True)
        min_share_val = shares.min() if not shares.empty else 1
        max_share_val = shares.max() if not shares.empty else 1

        cat_summaries.append({
            "Variable": var,
            "HHI": hhi_val,
            "Min share": round(min_share_val, 4),
            "Max share": round(max_share_val, 4),
            "Missings share": round(miss_share, 4),
            "HHI warning": "HHI < 0.05" if hhi_val < hhi_low else ("HHI > 0.95" if hhi_val > hhi_high else ""),
            "Min share warning": f"Min share {min_share_val:.2%}" if min_share_val < min_share else "",
            "Max share warning": f"Max share {max_share_val:.2%}" if max_share_val > max_share else "",
            "Missings warning": f"{miss_share:.2%} missing" if miss_share > 0 else ""
        })

    var_cat_summary = pd.DataFrame(cat_summaries)

    # --- Numeric variables summary ---
    num_summaries = []
    for var in var_num:
        ser = df[var].copy()
        ser = ser[~ser.isin(special_values)]

        q1, q3 = ser.quantile([0.25, 0.75])
        iqr = q3 - q1
        lw = ser[ser >= q1 - 3 * iqr].min()
        uw = ser[ser <= q3 + 3 * iqr].max()
        outliers = ((ser < q1 - 3 * iqr) | (ser > q3 + 3 * iqr)).sum()
        out_share = outliers / len(ser)

        miss_share = ser.isna().mean()
        max_share_val = ser.value_counts(normalize=True, dropna=True).max() or 1

        num_summaries.append({
            "Variable": var,
            "Q1": round(q1, 4),
            "Median": round(ser.median(), 4),
            "Q3": round(q3, 4),
            "Lower whisker": round(lw, 4),
            "Upper whisker": round(uw, 4),
            "Share of outliers": round(out_share, 4),
            "Max share": round(max_share_val, 4),
            "Missings share": round(miss_share, 4),
            "Outliers warning": f"{out_share:.2%} outliers" if out_share > 0 else "",
            "Max share warning": f"Max share {max_share_val:.2%}" if max_share_val > max_share else "",
            "Missings warning": f"{miss_share:.2%} missing" if miss_share > 0 else ""
        })

    var_num_summary = pd.DataFrame(num_summaries)

    # --- Export ---
    if save_to:
        with pd.ExcelWriter(save_to, engine='xlsxwriter') as writer:
            var_cat_summary.to_excel(writer, sheet_name='Categorical Summary', index=False)
            var_num_summary.to_excel(writer, sheet_name='Numeric Summary', index=False)

    return var_cat_summary, var_num_summary, var_cat + var_num


def nan_treatment(df, x=None, var_skip=None, special_values=None):
    """Fill NaNs — median for numeric, 'Missing' for categorical."""
    df2 = df.copy()
    var_skip = var_skip or []
    special_values = special_values or []
    if x is None:
        x = list(set(df2.columns) - set(var_skip))

    for var, dtype in df2[x].dtypes.items():
        if var in var_skip or df2[var].isna().sum() == 0:
            continue

        if dtype.name in ['category', 'object']:
            df2[var] = df2[var].astype('string').fillna('Missing')
        else:
            median_val = df2[var].median()
            df2[var] = df2[var].fillna(median_val)
    return df2


def var_distr(df, var_list=None, groupby='target', special_values=None):
    """Plot categorical and numeric variable distributions."""
    special_values = special_values or []
    var_skip = list(set(df.columns) - set(var_list))
    var_cat, var_num = var_types(df, var_skip)

    # --- Categorical vars ---
    for col in var_cat:
        dfi = df[~df[col].astype(str).isin(special_values)]
        if groupby not in dfi.columns:
            warnings.warn(f"Groupby column '{groupby}' not found. Skipping {col}.")
            continue

        cross_tab = pd.crosstab(dfi[col], dfi[groupby])
        cross_tab = cross_tab.sort_values(cross_tab.sum(axis=1), ascending=False)

        norm_tab = cross_tab.div(cross_tab.sum(axis=1), axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        cross_tab.plot(kind='bar', stacked=True, ax=axes[0], colormap='Accent', title=f"{col} counts")
        norm_tab.plot(kind='bar', stacked=True, ax=axes[1], colormap='Accent', title=f"{col} % distribution")

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()

    # --- Numeric vars ---
    for col in var_num:
        dfi = df[~df[col].astype(str).isin(special_values)]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        sns.boxplot(data=dfi, x=col, ax=axes[0])
        axes[0].set_title(f"{col} Boxplot")

        if groupby in dfi.columns:
            sns.kdeplot(data=dfi, x=col, hue=groupby, common_norm=False, fill=True, ax=axes[1])
        else:
            sns.kdeplot(data=dfi, x=col, fill=True, ax=axes[1])

        axes[1].set_title(f"{col} Density by {groupby}")
        plt.tight_layout()
        plt.show()
