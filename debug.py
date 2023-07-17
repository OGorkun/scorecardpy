import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandasql import sqldf
from scorecardpy.woebin import *

# data prepare ------
# load germancredit data
smp_full = germancredit()
smp_full['target'] = smp_full['creditability'].apply(lambda x: 1 if x == 'bad' else 0)
smp_full = smp_full.drop(columns = ['creditability'])
smp_full.loc[0:99, 'credit_amount'] = np.nan
smp_full.loc[0:99, 'purpose'] = np.nan
smp_full.loc[100:109, 'target'] = np.nan
smp_full.loc[0:40, 'age_in_years'] = np.nan
smp_full.loc[40:60, 'age_in_years'] = -9999

for i in range(5):
    smp_full = pd.concat([smp_full, smp_full])
smp_full['RepDate_End'] = np.random.randint(1, 73, smp_full.shape[0])
smp_full = smp_full.reset_index(drop=True)


# 1. Exploratory analysis of variables (missings, outliers, concentration/distribution) - based on smp_full
# good/bad label
target = 'target'

# date column (e.g. snapshot date or application date)
date = 'RepDate_End'

# other columns that are not variables
var_skip = []

# all columns that are not variables
var_skip_all = var_skip + [target, date]

# special values for numeric variables
special_values = [-9999999]


# heatmap for the missing values
# miss_heatmap(smp_full, var_skip, fig_width=10, fig_height=6)


# variables checks summary
var_cat_summary, var_num_summary, var_list = expl_analysis(smp_full, var_skip_all, special_values)

display(var_cat_summary)
display(var_num_summary)

var_max_share = var_cat_summary[var_cat_summary['Max share'] > 0.95]['Variable'].tolist() \
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


# 2. Development sample creation
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



# 3. Automated binning
# min bin size for fine classing
min_perc_fine_bin = 0.05

# min bin size for coarse classing
count_distr_limit = 0.1

# max number of coarse classes
bin_num_limit = int(1/count_distr_limit)

# number of decimals for bin intervals
bin_decimals = 4


# binning
fine_class, coarse_class = woebin(train, y = target,
                                  #x = ["age_in_years", "status_of_existing_checking_account", "foreign_worker"],
                                  var_skip = var_skip_all,
                                  special_values = special_values,
                                  min_perc_fine_bin = min_perc_fine_bin,
                                  count_distr_limit = count_distr_limit,
                                  bin_num_limit = bin_num_limit,
                                  print_step = 5,
                                  ignore_datetime_cols = False,
                                  bin_decimals = bin_decimals)


# extracting binning results to excel
pd.concat(fine_class.values()).reset_index(drop=True).to_excel('3_1_fine_classing.xlsx')
pd.concat(coarse_class.values()).reset_index(drop=True).to_excel('3_2_coarse_classing_auto.xlsx')

# iv for variables after automated binning
coarse_class_iv = vars_iv(coarse_class)
coarse_class_iv.to_excel('3_3_coarse_classing_auto_iv.xlsx')
coarse_class_iv


# automated filtering of variables using iv and correlation from the fine classing
var_list, var_rej_fine = vars_filter(train, fine_class, corr_threshold = 0.7, iv_threshold = 0.02)
var_rej_fine


# removing excluded variables from coarse_class dictionary
coarse_class_filt = {k: v for k, v in coarse_class.items() if k in var_list}


# binning visualization
# woebin_plot(coarse_class_filt)


