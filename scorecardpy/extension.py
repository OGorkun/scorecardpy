# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def var_types(smp, var_skip):
    var_cat = []
    var_num = []
    for var, dt in smp.dtypes.items():
        if var in var_skip:
            break
        else:
            if dt.name in ['category', 'object']: #or len(smp[var].unique()) <= 10:
                var_cat.append(var)
            else: var_num.append(var)
    return var_cat, var_num

def hhi(series):
    _, cnt = np.unique(series.astype(str), return_counts=True)
    return np.square(cnt/cnt.sum()).sum()  
 
# Preliminary analysis of variables
def var_pre_analysis(smp, var_cat=[], var_num=[], spl_val=[], hhi_low=0.05, hhi_high=0.95, min_share=0.05):
    # categorical variables
    var_hhi = []
    var_min_share = []
    var_miss_share = []
    for var in var_cat:
        var_hhi.append(round(hhi(smp[var]),4))
        var_min_share.append(round(min(smp[var].value_counts(normalize=True)),4))
        miss = len(smp.index) - smp[var].count() + smp.loc[smp[var].str.strip() == '',var].count()
        var_miss_share.append(round(miss/len(smp.index),4))
    var_cat_summary = pd.DataFrame({'Variable': var_cat, \
                                    'HHI': var_hhi, \
                                    'Min share': var_min_share, \
                                    'Missings share': var_miss_share})
    var_cat_summary['HHI warning'] = var_cat_summary['HHI']\
                                        .apply(lambda x: 'HHI < 0.05' if x < hhi_low else ('HHI > 0.95' if x > hhi_high else ''))
    var_cat_summary['Min share warning'] = var_cat_summary['Min share']\
                                            .apply(lambda x: 'Min share is ' + str(round(x*100,2)) + '%' if x < min_share else '')
    var_cat_summary['Missings warning'] = var_cat_summary['Missings share']\
                                            .apply(lambda x: str(round(x*100,2)) + '% missing values' if x > 0 else '')
    
    # numeric variables
    q1 = []
    med = []
    q3 = []
    lw = []
    uw = []
    out_share = []
    var_miss_share = []
    for var in var_num:
        var_ser = smp[var]
        var_ser = var_ser[~var_ser.isin(spl_val)]
        q1_num = var_ser.quantile(0.25)
        q3_num = var_ser.quantile(0.75)
        iqr = q3_num-q1_num
        q1.append(round(q1_num,4))
        q3.append(round(q3_num,4))
        med.append(round(var_ser.quantile(0.5),4))
        lw.append(round(var_ser[var_ser >= q1_num - 3*iqr].min(),4))
        uw.append(round(var_ser[var_ser <= q3_num + 3*iqr].max(),4))
        out_share_num = var_ser[var_ser < q1_num - 3*iqr].count() + var_ser[var_ser > q3_num + 3*iqr].count()
        out_share.append(round(out_share_num/len(var_ser.index),4))
        miss = smp.loc[:, var].isna().sum()
        var_miss_share.append(round(miss/len(smp.index),4))
    var_num_summary = pd.DataFrame({'Variable': var_num, \
                                    'Q1': q1, \
                                    'Median': med, \
                                    'Q3': q3, \
                                    'Lower whisker': lw, \
                                    'Upper whisker': uw, \
                                    'Share of outliers': out_share, \
                                    'Missings share': var_miss_share})
    var_num_summary['Outliers warning'] = var_num_summary['Share of outliers']\
                                            .apply(lambda x: str(round(x*100,2)) + '% outliers' if x > 0 else '')
    var_num_summary['Missings warning'] = var_num_summary['Missings share']\
                                            .apply(lambda x: str(round(x*100,2)) + '% missing values' if x > 0 else '')
    
    return var_cat_summary, var_num_summary

# Distribution of categorical variable (bar plots saved as pdf)
def var_cat_distr(smp, var_list, pdf_name, groupby='will_default'):
    pp = PdfPages(pdf_name)
    for i in var_list:
        cross_tab = pd.crosstab(index=train[i], 
                                columns=train[groupby])
        cross_tab['Total'] = cross_tab.sum(axis=1)
        cross_tab = cross_tab.sort_values(by=['Total'], ascending = False)
        cross_tab = cross_tab.drop(columns=['Total'])

        cross_tab_norm = pd.crosstab(index=train[i],
                                     columns=train[groupby],
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
        pp.savefig(fig, bbox_inches = 'tight')
    pp.close()

def var_num_distr(smp, var_list, pdf_name, groupby='will_default'):
    pp = PdfPages(pdf_name)
    for i in var_list:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax1 = sns.boxplot(data=smp, 
                          x=i, 
                          ax=axes[0])
        ax2 = sns.kdeplot(data=smp, 
                          x=i, 
                          hue=groupby, 
                          common_norm=False, 
                          ax=axes[1])
        pp.savefig(fig, bbox_inches = 'tight')
    pp.close()

# Creating summary table with binning results
def vars_iv(var_list, bins_var):
    iv = []
    for var in var_list:
        iv.append(bins_var[var].iloc[0]['total_iv'].round(4))
    vars_iv = pd.DataFrame({
      'variable': var_list,
      'iv': iv
    })
    return vars_iv.sort_values('iv', ascending=False).reset_index(drop=True)

