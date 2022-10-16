# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import random as rd


def gini_vars(sample, target, vars_list, result_name):
    gini_vars = []
    for var in vars_list:
        gini_var = -(roc_auc_score(sample[target], sample[var])*2 - 1)
        gini_vars.append(gini_var)
    gini_vars_df = pd.DataFrame({'Variable': vars_list, result_name: gini_vars})
    return gini_vars_df

def gini_over_time(sample, target, vars_list, date) :
    sorted_date = sorted(sample[date].unique())
    # del sorted_date[-12:]
    gini_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        gini_date = gini_vars(sample_date, target, vars_list, i)
        gini_date = gini_date.rename(columns={"Variable": "Period"})
        gini_date = gini_date.set_index('Period').T
        gini_ot = pd.concat([gini_ot, gini_date])
    return gini_ot

def score_ranges(sample, score, nintervals=10) :
    intervals = pd.cut(sample[score], nintervals)
    intervals_unique = intervals.unique()
    output = pd.DataFrame({'range': intervals_unique})
    output['score_range'] = output['range']
    output = output.set_index(['range'])
    return output

def score_distr(sample, target, score='score', score_range='score_range') :
    sample_ranges = sample[['score_range',target]].groupby(['score_range']).agg(['count', 'sum' ])
    sample_ranges.columns = sample_ranges.columns.droplevel(0)
    sample_ranges = sample_ranges.rename(columns={"count": "Total", "sum": "Bads"})
    sample_ranges['Goods'] = (sample_ranges['Total'] - sample_ranges['Bads'])
    sample_ranges['Total Share'] = sample_ranges['Total'] / sample_ranges['Total'].sum()
    sample_ranges['Bads Share'] = sample_ranges['Bads'] / sample_ranges['Bads'].sum()
    sample_ranges['Goods Share'] = sample_ranges['Goods'] / sample_ranges['Goods'].sum()
    sample_ranges['score_range'] = sample_ranges.index
    ranges_mid = sample_ranges.apply(lambda x: x['score_range'].mid, axis=1)
    ranges_mid_df = pd.DataFrame({'score': ranges_mid}) 
    sample_ranges = pd.merge(sample_ranges, ranges_mid_df, left_index=True, right_index=True) 
    sample_ranges.reset_index(drop=True, inplace=True)
    cols = sample_ranges.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    sample_ranges = sample_ranges[cols]
    return sample_ranges

def psi(expected_share, actual_share):
    '''Calculate the PSI for a single variable
    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into
    Returns:
       psi_value: calculated PSI value
    '''

    def sub_psi(e_perc, a_perc):
        '''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)
    
    iterable = (sub_psi(expected_share[x], actual_share[x]) for x in range(0, len(expected_share)))
    psi_value = np.sum(np.fromiter(iterable, float))

    return(psi_value)

def psi_vars(ref_sample, sample, vars_list, result_name):
    psi_vars = []
    for var in vars_list:
        ref_sample_groups = ref_sample.groupby([var]).size()
        ref_sample_groups_df = pd.DataFrame({'Total': ref_sample_groups})
        ref_sample_groups_df['Total_Share'] = ref_sample_groups_df['Total'] / ref_sample_groups_df['Total'].sum()

        sample_groups = sample.groupby([var]).size()
        sample_groups_df = pd.DataFrame({'Total1': sample_groups})
        sample_groups_df = pd.merge(ref_sample_groups_df, sample_groups_df, left_index=True, right_index=True, how="outer")
        sample_groups_df['Total_Share'] = sample_groups_df['Total1'] / sample_groups_df['Total1'].sum()
        
        ref_sample_groups_df.reset_index(drop=True, inplace=True) 
        sample_groups_df.reset_index(drop=True, inplace=True)
        
        psi_var = psi(ref_sample_groups_df['Total_Share'], sample_groups_df['Total_Share'])
        psi_vars.append(psi_var)
    psi_vars_df = pd.DataFrame({'Variable': vars_list, result_name: psi_vars})
    return psi_vars_df

def psi_over_time(ref_sample, sample, vars_list, date) :
    sorted_date = sorted(sample[date].unique())
    psi_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        psi_date = psi_vars(ref_sample, sample_date, vars_list, i)
        psi_date = psi_date.rename(columns={"Variable": date})
        psi_date = psi_date.set_index(date).T
        psi_ot = pd.concat([psi_ot, psi_date])
    return psi_ot

def hhi(series):
    _, cnt = np.unique(series.astype(str), return_counts=True)
    return np.square(cnt/cnt.sum()).sum()  
    
# Creating summary table with binning results
def vars_iv(var_list, bins_var):
    iv = []
    for var in var_list:
        iv.append(bins_var[var].iloc[0]['total_iv'])
    vars_iv = pd.DataFrame({
      'variable': var_list,
      'iv': iv
    })
    return vars_iv.sort_values('iv', ascending=False).reset_index(drop=True)
