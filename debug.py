import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandasql import sqldf

# data prepare ------
# load germancredit data
smp_full = germancredit()
smp_full['target'] = smp_full['creditability'].apply(lambda x: 1 if x == 'bad' else 0)
smp_full = smp_full.drop(columns=['creditability'])
smp_full.loc[0:95, 'credit_amount'] = np.nan
smp_full.loc[0:99, 'purpose'] = np.nan
smp_full.loc[100:109, 'target'] = np.nan
smp_full.loc[80:100, 'age_in_years'] = -9999999
smp_full.loc[40:80, 'age_in_years'] = np.nan
smp_full['credit_test'] = 1/smp_full['credit_amount']

for i in range(5):
    smp_full = pd.concat([smp_full, smp_full])
smp_full['RepDate_End'] = np.random.randint(1, 73, smp_full.shape[0])
smp_full = smp_full.reset_index(drop=True)

# good/bad label
target = 'target'

# date column (e.g. snapshot date or application date)
date = 'RepDate_End'

# other columns that are not variables
var_skip = []

# all columns that are not variables
var_skip_all = var_skip + [target, date]

# special values for numeric variables - TBD
special_values = [-9999999]

# heatmap for the missing values
# miss_heatmap(smp_full, var_skip, fig_width=10, fig_height=6)


# variables checks summary
var_cat_summary, var_num_summary, var_list = expl_analysis(smp_full, var_skip_all, special_values)

display(var_cat_summary)
display(var_num_summary)

var_no_bin = var_cat_summary[var_cat_summary['Max share'] > 0.95]['Variable'].tolist() \
             + var_num_summary[var_num_summary['Max share'] > 0.95]['Variable'].tolist()


# treatment of nan - median for numeric and 'Missing' for string
# smp_full2 = sc.nan_treatment(smp_full, x = None, var_skip = var_skip_all)


# variables distribution
# var_distr(smp_full, var_list, groupby = target, special_values = special_values)
# var_distr(smp_full, var_list=['purpose'], groupby = target, special_values = special_values)


# analysis of shares of missings and bads in target over time
def nan_rate(target): return sum(np.isnan(target)) / len(target)


def bad_rate(target): return sum(target == 1) / (sum(target == 0) + sum(target == 1))


target_ot = smp_full.groupby(date)[target].agg([nan_rate, bad_rate])

# dates with blank target
# pd.DataFrame(target_ot[target_ot['nan_rate'] > 0]['nan_rate'])


# bad rate over time
# target_ot['bad_rate'].plot.bar()


# selection of the development window
smp_dev = smp_full[smp_full[date].between(1, 60)]

# selection of variables that will be used for the development
smp_dev = smp_dev[var_list + [target]]

# check target
print(smp_dev.groupby(target, dropna=False).size())

# delete records with blank target
smp_dev = smp_dev[smp_dev[target].notna()]

# train/test split as 80/20
train, test = split_df(smp_dev, ratio=0.8, seed=123).values()
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# train/test sample size
query = """ select 'train' as sample, 
                sum(target) as bads, 
                count(*) as obs, 
                sum(target)*1.00/count(target) as BR
            from train 
                union 
            select 'test' as sample, 
                sum(target) as bads, 
                count(*) as obs, 
                sum(target)*1.00/count(target) as BR
            from test 
        """
# Query execution
sqldf(query)
# pd.DataFrame({'train':pd.Series(train.groupby(target, dropna=False).size()),
#               'test':pd.Series(test.groupby(target, dropna=False).size())})


# binning
fine_class, coarse_class = woebin(train, y=target,
                                  #x = ["credit_test"],
                                  var_skip=var_skip_all,
                                  special_values=special_values,
                                  min_perc_fine_bin=0.05,  # min bin size for fine classing
                                  count_distr_limit=0.1,  # min bin size for coarse classing
                                  bin_num_limit=10,  # max number of coarse classes
                                  print_step = 1,
                                  ignore_datetime_cols = False,
                                  bin_decimals = 4)

# var_no_bin = []
# for i in smp_full.columns:
#     ser_i = smp_full[~smp_full[i].isin(special_values)][i]
#     if max(ser_i.value_counts()) > len(ser_i) - 0.05*len(smp_full.index):
#         var_no_bin = var_no_bin + [i]


# extracting binning results to excel
pd.concat(fine_class.values()).reset_index(drop=True).to_excel('3_1_fine_classing.xlsx')
pd.concat(coarse_class.values()).reset_index(drop=True).to_excel('3_2_coarse_classing_auto.xlsx')


# iv for variables after automated binning
coarse_class_iv = vars_iv(coarse_class)
coarse_class_iv.to_excel('3_3_coarse_classing_auto_iv.xlsx')
coarse_class_iv


# binning visualization
woebin_plot(coarse_class)