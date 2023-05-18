import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def var_types(df, var_skip=None):
    var_cat = []
    var_num = []
    for var, dt in df.dtypes.items():
        if var not in var_skip:
            if dt.name in ['category', 'object'] or len(df[var].unique()) <= 10:
                var_cat.append(var)
            else:
                var_num.append(var)
    return var_cat, var_num


# heatmap for the missing values
def miss_heatmap(df, var_skip=None, save_to='1_1_missings_heatmap.png', fig_width=10, fig_height=6):
    var_cat, var_num = var_types(df, var_skip)

    percent_missing = df.loc[:, var_cat + var_num].isna().sum() * 100 / len(df)
    percent_missing = pd.DataFrame({'column': percent_missing.index, 'percent_missing': percent_missing.values})
    percent_missing.sort_values('percent_missing', ascending=False, inplace=True)
    percent_missing.reset_index(drop=True)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(df[percent_missing.column].isna().transpose(),
                cmap="YlGnBu",
                cbar_kws={'label': 'Missing Data'})
    plt.savefig(save_to, dpi=100, bbox_inches="tight")


def hhi(series):
    _, cnt = np.unique(series.astype(str), return_counts=True)
    return np.square(cnt / cnt.sum()).sum()


# Exploratory analysis of variables
def expl_analysis(df, var_skip=None, special_values=[],
                  hhi_low=0.05, hhi_high=0.95,
                  min_share=0.05, max_share=0.9,
                  save_to='1_2_exploratory_analysis.xlsx'):
    var_cat, var_num = var_types(df, var_skip)
    # categorical variables
    var_hhi = []
    var_min_share = []
    var_max_share = []
    var_miss_share = []
    for var in var_cat:
        var_hhi.append(round(hhi(df[var]), 4))
        if df[var].dtype.name in ['category', 'object']:
            miss = len(df.index) - df[var].count() + df.loc[df[var].str.strip() == '', var].count()
        else:
            miss = df.loc[:, var].isna().sum()
        var_miss_share.append(round(miss / len(df.index), 8))
        if miss < len(df.index):
            var_min_share.append(round(min(df[var].value_counts(normalize=True)), 4))
            var_max_share.append(round(max(df[var].value_counts(normalize=True)), 4))
        else:
            var_min_share.append(1)
            var_max_share.append(1)
    var_cat_summary = pd.DataFrame({'Variable': var_cat, \
                                    'HHI': var_hhi, \
                                    'Min share': var_min_share, \
                                    'Max share': var_max_share, \
                                    'Missings share': var_miss_share})
    var_cat_summary['HHI warning'] = var_cat_summary['HHI'] \
        .apply(lambda x: 'HHI < 0.05' if x < hhi_low else ('HHI > 0.95' if x > hhi_high else ''))
    var_cat_summary['Min share warning'] = var_cat_summary['Min share'] \
        .apply(lambda x: 'Min share is ' + str(round(x * 100, 2)) + '%' if x < min_share else '')
    var_cat_summary['Max share warning'] = var_cat_summary['Max share'] \
        .apply(lambda x: 'Max share is ' + str(round(x * 100, 2)) + '%' if x > max_share else '')
    var_cat_summary['Missings warning'] = var_cat_summary['Missings share'] \
        .apply(lambda x: str(round(x * 100, 2)) + '% missing values' if x > 0 else '')

    # numeric variables
    q1 = []
    med = []
    q3 = []
    lw = []
    uw = []
    out_share = []
    var_miss_share = []
    var_max_share = []
    for var in var_num:
        var_ser = df[var]
        var_ser = var_ser[~var_ser.isin(special_values)] # TODO - consider special values to be a dict
        q1_num = var_ser.quantile(0.25)
        q3_num = var_ser.quantile(0.75)
        iqr = q3_num - q1_num
        q1.append(round(q1_num, 4))
        q3.append(round(q3_num, 4))
        med.append(round(var_ser.quantile(0.5), 4))
        lw.append(round(var_ser[var_ser >= q1_num - 3 * iqr].min(), 4))
        uw.append(round(var_ser[var_ser <= q3_num + 3 * iqr].max(), 4))
        out_share_num = var_ser[var_ser < q1_num - 3 * iqr].count() + var_ser[var_ser > q3_num + 3 * iqr].count()
        out_share.append(round(out_share_num / len(var_ser.index), 4))
        miss = df.loc[:, var].isna().sum()
        var_miss_share.append(round(miss / len(df.index), 8))
        if miss < len(df.index):
            var_max_share.append(round(max(df[var].value_counts(normalize=True)), 4))
        else:
            var_max_share.append(1)
    var_num_summary = pd.DataFrame({'Variable': var_num, \
                                    'Q1': q1, \
                                    'Median': med, \
                                    'Q3': q3, \
                                    'Lower whisker': lw, \
                                    'Upper whisker': uw, \
                                    'Share of outliers': out_share, \
                                    'Max share': var_max_share, \
                                    'Missings share': var_miss_share})
    var_num_summary['Outliers warning'] = var_num_summary['Share of outliers'] \
        .apply(lambda x: str(round(x * 100, 2)) + '% outliers' if x > 0 else '')
    var_num_summary['Max share warning'] = var_num_summary['Max share'] \
        .apply(lambda x: 'Max share is ' + str(round(x * 100, 2)) + '%' if x > max_share else '')
    var_num_summary['Missings warning'] = var_num_summary['Missings share'] \
        .apply(lambda x: str(round(x * 100, 2)) + '% missing values' if x > 0 else '')

    # export to excel
    writer = pd.ExcelWriter(save_to, engine='xlsxwriter')
    var_cat_summary.to_excel(writer, sheet_name='var_cat_summary')
    var_num_summary.to_excel(writer, sheet_name='var_num_summary')
    writer.close()

    return var_cat_summary, var_num_summary, var_cat+var_num

# treatment of nan - median for numeric and 'Missing' for string
def nan_treatment(df, x=None, var_skip=None, special_values=[]):
    df2 = df
    if x is None:
        x = list(set(df2.columns) - set(var_skip))
    for var, dt in df2[x].dtypes.items():
        if var not in var_skip and df2[var].isna().sum() > 0:
            print(var, df2[var].isna().sum())
            if dt.name == 'category':
                df2[var] = df2[var].cat.add_categories('Missing').fillna('Missing')
                print('Missing')
            if dt.name == 'object':
                df2[var] = df2[var].fillna('Missing')
                print('Missing')
            else:
                print(df[var].median())
                df2[var] = df2[var].fillna(df2[var].median())
    return df2

# Distribution of categorical variable (bar plots saved as pdf)
def var_distr(df, var_skip=None, groupby='target', special_values=[]):
    # pp = PdfPages(pdf_name)
    var_cat, var_num = var_types(df, var_skip)

    #categorical vars
    for i in var_cat:
        dfi = df[~df[i].isin(special_values)]
        cross_tab = pd.crosstab(index=dfi[i],  # .fillna('missing'),
                                columns=dfi[groupby])
        cross_tab['Total'] = cross_tab.sum(axis=1)
        cross_tab = cross_tab.sort_values(by=['Total'], ascending=False)
        cross_tab = cross_tab.drop(columns=['Total'])

        cross_tab_norm = pd.crosstab(index=dfi[i],
                                     columns=dfi[groupby],
                                     normalize="index")
        cross_tab_norm = cross_tab_norm.reindex(index=cross_tab.index)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax1 = cross_tab.plot(kind='bar',
                             ax=axes[0],
                             stacked=True,
                             colormap='Accent')
        plt.sca(axes[0])
        plt.xticks(
            rotation=45,
            horizontalalignment='right',
            fontweight='light',
            fontsize=8)

        ax2 = cross_tab_norm.plot(kind='bar',
                                  ax=axes[1],
                                  stacked=True,
                                  colormap='Accent')
        plt.sca(axes[1])
        plt.xticks(
            rotation=45,
            horizontalalignment='right',
            fontweight='light',
            fontsize=8)
        # pp.savefig(fig, bbox_inches = 'tight')
    # pp.close()

    # numerical vars
    for i in var_num:
        dfi = df[~df[i].isin(special_values)]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax1 = sns.boxplot(data=dfi,
                          x=i,
                          ax=axes[0])
        ax2 = sns.kdeplot(data=dfi,
                          x=i,
                          hue=groupby,
                          common_norm=False,
                          ax=axes[1])
        # pp.savefig(fig, bbox_inches = 'tight')
    # pp.close()
