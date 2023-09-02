import pandas as pd
import numpy as np

def germancredit_breaks_list():
    germancredit_breaks_list = {'age_in_years': [30.0,40.0],
        'credit_amount': [983.0,2788.0,4042.0,6579.0],
        'credit_history': ['no credits taken/ all credits paid back duly%,%all credits at this bank paid back duly%,%existing credits paid back duly till now%,%delay in paying off in the past','critical account/ other credits existing (not at this bank)'],
        'duration_in_month': [12.0,18.0,21.0,36.0],
        'housing': ['rent','own','for free'],
        'other_installment_plans': ['bank%,%stores','none'],
        'present_employment_since': ['unemployed%,%... < 1 year','1 <= ... < 4 years','4 <= ... < 7 years','... >= 7 years'],
        'property': ['real estate','building society savings agreement/ life insurance','car or other, not in attribute Savings account/bonds','unknown / no property'],
        'purpose': ['retraining%,%car (used)','radio/television','furniture/equipment%,%business%,%domestic appliances%,%car (new)%,%repairs%,%others%,%education'],
        'savings_account_and_bonds': ['... < 100 DM','100 <= ... < 500 DM','500 <= ... < 1000 DM%,%... >= 1000 DM%,%unknown/ no savings account'],
        'status_of_existing_checking_account': ['... < 0 DM','0 <= ... < 200 DM','... >= 200 DM / salary assignments for at least 1 year%,%no checking account']}

    return germancredit_breaks_list

def germancredit_scorecard_points():
    germancredit_scorecard_points = {'basepoints': pd.DataFrame({'variable': ['basepoints'], 'bin': [np.nan], 'points': [232.0]}),
        'credit_amount': pd.DataFrame({'variable': ['credit_amount', 'credit_amount', 'credit_amount', 'credit_amount', 'credit_amount', 'credit_amount'], 'bin': ['missing', '[-inf,983.0)', '[983.0,2788.0)', '[2788.0,4042.0)', '[4042.0,6579.0)', '[6579.0,inf)'], 'points': [43.0, 27.0, 38.0, 52.0, 21.0, 0.0]}),
        'property': pd.DataFrame({'variable': ['property', 'property', 'property', 'property'], 'bin': ['real estate', 'building society savings agreement/ life insurance', 'car or other, not in attribute Savings account/bonds', 'unknown / no property'], 'points': [26.0, 15.0, 14.0, 0.0]}),
        'duration_in_month': pd.DataFrame({'variable': ['duration_in_month', 'duration_in_month', 'duration_in_month', 'duration_in_month', 'duration_in_month'], 'bin': ['[-inf,12.0)', '[12.0,18.0)', '[18.0,21.0)', '[21.0,36.0)', '[36.0,inf)'], 'points': [85.0, 53.0, 30.0, 38.0, 0.0]}),
        'other_installment_plans': pd.DataFrame({'variable': ['other_installment_plans', 'other_installment_plans'], 'bin': ['bank%,%stores', 'none'], 'points': [0.0, 30.0]}),
        'purpose': pd.DataFrame({'variable': ['purpose', 'purpose', 'purpose', 'purpose'], 'bin': ['missing', 'retraining%,%car (used)', 'radio/television', 'furniture/equipment%,%business%,%domestic appliances%,%car (new)%,%repairs%,%others%,%education'], 'points': [32.0, 64.0, 43.0, 0.0]}),
        'housing': pd.DataFrame({'variable': ['housing', 'housing', 'housing'], 'bin': ['rent', 'own', 'for free'], 'points': [2.0, 21.0, 0.0]}),
        'age_in_years': pd.DataFrame({'variable': ['age_in_years', 'age_in_years', 'age_in_years', 'age_in_years', 'age_in_years'], 'bin': ['missing', '-9999.0', '[-inf,30.0)', '[30.0,40.0)', '[40.0,inf)'], 'points': [22.0, 47.0, 0.0, 24.0, 26.0]}),
        'savings_account_and_bonds': pd.DataFrame({'variable': ['savings_account_and_bonds', 'savings_account_and_bonds', 'savings_account_and_bonds'], 'bin': ['... < 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM%,%... >= 1000 DM%,%unknown/ no savings account'], 'points': [0.0, 5.0, 42.0]}),
        'status_of_existing_checking_account': pd.DataFrame({'variable': ['status_of_existing_checking_account', 'status_of_existing_checking_account', 'status_of_existing_checking_account'], 'bin': ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year%,%no checking account'], 'points': [0.0, 21.0, 93.0]}),
        'present_employment_since': pd.DataFrame({'variable': ['present_employment_since', 'present_employment_since', 'present_employment_since', 'present_employment_since'], 'bin': ['unemployed%,%... < 1 year', '1 <= ... < 4 years', '4 <= ... < 7 years', '... >= 7 years'], 'points': [0.0, 17.0, 34.0, 28.0]})}

    return germancredit_scorecard_points