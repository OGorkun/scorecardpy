import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def pd_from_score(score, points0=540, odds0=1 / 9, pdo=40):
    b = pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    pd = 1 / (1 + np.exp(score / b - a / b))
    return pd


def calibration(smp, score='score', target='target', points0=540, odds0=1/9, pdo=40):
    b = pdo/np.log(2) 
    a = points0 + b*np.log(odds0)
    log_odds = a/b - smp[score]/b
    x = log_odds.to_numpy().reshape(-1, 1)
    y = smp[target].to_numpy()
    
    lr_calib = LogisticRegression(penalty="l1", C=0.9, solver="saga", n_jobs=-1)
    lr_calib.fit(x, y)

    pd_calib = lr_calib.predict_proba(x)[:,1]
    log_odds_calib = np.log((1-pd_calib)/(pd_calib))

    intercept_calib = points0 + (np.log(odds0) - lr_calib.intercept_) * pdo / np.log(2)
    slope_calib = lr_calib.coef_ * pdo / np.log(2)
    intercept_calib = intercept_calib[0]
    slope_calib = slope_calib[0,0]

    intercept = intercept_calib - slope_calib*a/b
    slope = slope_calib/b
    return intercept, slope