# -*- coding: utf-8 -*-
'''
perf.py
Modernized version for pandas>=2.2, numpy>=2, matplotlib>=3.9
Performance evaluation, KS, ROC, Lift, PSI, IV, and Gini utilities.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from .condition_fun import check_y
from .info_value import iv_01
from .woebin import n0, n1
from .calibration import pd_from_score
from .extension import hhi
from sklearn.metrics import roc_auc_score
from pathlib import Path


# ==========================================================
# Utility Functions
# ==========================================================

def _safe_sum(series, val):
    r'''Avoid issues with boolean dtype summation.'''
    return int((series == val).sum())


# ==========================================================
# KS, ROC, LIFT computation
# ==========================================================

def eva_dfkslift(df, groupnum=None):
    r'''Generate dataframe with cumulative stats for KS and Lift charts.'''
    if groupnum is None:
        groupnum = len(df)

    df = (
        df.sort_values('pred', ascending=False)
        .reset_index(drop=True)
        .assign(
            group=lambda x: np.ceil((x.index + 1) / (len(x) / groupnum))
        )
        .groupby('group', observed=True)['label']
        .agg(good=lambda x: (x == 0).sum(), bad=lambda x: (x == 1).sum())
        .reset_index()
        .assign(
            group=lambda x: (x.index + 1) / len(x),
            good_distri=lambda x: x.good / x.good.sum(),
            bad_distri=lambda x: x.bad / x.bad.sum(),
            badrate=lambda x: x.bad / (x.good + x.bad),
            cumbadrate=lambda x: np.cumsum(x.bad) / np.cumsum(x.good + x.bad),
            lift=lambda x: (
                (np.cumsum(x.bad) / np.cumsum(x.good + x.bad))
                / (x.bad.sum() / (x.good.sum() + x.bad.sum()))
            ),
            cumgood=lambda x: np.cumsum(x.good) / x.good.sum(),
            cumbad=lambda x: np.cumsum(x.bad) / x.bad.sum()
        )
        .assign(ks=lambda x: np.abs(x.cumbad - x.cumgood))
    )

    # Add 0th row for baseline
    zero_row = pd.DataFrame({
        'group': [0],
        'good': [0],
        'bad': [0],
        'good_distri': [0],
        'bad_distri': [0],
        'badrate': [0],
        'cumbadrate': [np.nan],
        'cumgood': [0],
        'cumbad': [0],
        'ks': [0],
        'lift': [np.nan]
    })

    return pd.concat([zero_row, df], ignore_index=True)


# ==========================================================
# PLOTTING FUNCTIONS
# ==========================================================

def eva_pks(dfkslift, title="", ax=None):
    r'''Plot KS curve. Draws into provided ax if given, otherwise creates a new figure.'''
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    dfks = dfkslift.loc[dfkslift['ks'].idxmax()]

    ax.plot(dfkslift.group, dfkslift.ks, 'b-', label='KS')
    ax.plot(dfkslift.group, dfkslift.cumgood, 'k-', label='Cumulative Good')
    ax.plot(dfkslift.group, dfkslift.cumbad, 'k--', label='Cumulative Bad')
    ax.axvline(dfks['group'], color='r', linestyle='--')

    ax.set_title(f"{title} K-S")
    ax.set_xlabel("% of population")
    ax.set_ylabel("% of total Good/Bad")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(dfks['group'], dfks['ks'], f"KS={dfks['ks']:.4f}", color='b', ha='center')
    ax.legend(loc="lower right")

    return ax.figure if created_fig else ax.figure


def eva_plift(dfkslift, title="", ax=None):
    r'''Plot Lift curve. Draws into provided ax if given, otherwise creates a new figure.'''
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    badrate_avg = dfkslift['bad'].sum() / (dfkslift['good'].sum() + dfkslift['bad'].sum())

    ax.plot(dfkslift.group, dfkslift.cumbadrate, 'k-', label='Cumulative Bad Rate')
    ax.axhline(badrate_avg, color='r', linestyle='--', label='Average Bad Rate')

    ax.set_title(f"{title} Lift")
    ax.set_xlabel("% of population")
    ax.set_ylabel("% of Bad")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    return ax.figure if created_fig else ax.figure


def eva_dfrocpr(df):
    r'''Generate detailed ROC/PR DataFrame.'''
    df = (
        df.sort_values('pred')
        .groupby('pred', observed=True)['label']
        .agg(countN=lambda x: (x == 0).sum(),
             countP=lambda x: (x == 1).sum())
        .reset_index()
        .assign(
            FN=lambda x: np.cumsum(x.countP),
            TN=lambda x: np.cumsum(x.countN)
        )
        .assign(
            TP=lambda x: x.countP.sum() - x.FN,
            FP=lambda x: x.countN.sum() - x.TN
        )
        .assign(
            TPR=lambda x: x.TP / (x.TP + x.FN),
            FPR=lambda x: x.FP / (x.TN + x.FP),
            precision=lambda x: np.divide(
                x.TP, x.TP + x.FP, out=np.zeros_like(x.TP, dtype=float), where=(x.TP + x.FP) != 0
            ),
            recall=lambda x: np.divide(
                x.TP, x.TP + x.FN, out=np.zeros_like(x.TP, dtype=float), where=(x.TP + x.FN) != 0
            )
        )
        .assign(F1=lambda x: np.divide(
            2 * x.precision * x.recall,
            x.precision + x.recall,
            out=np.zeros_like(x.precision, dtype=float),
            where=(x.precision + x.recall) != 0
        ))
    )
    return df


def eva_proc(dfrocpr, title="", ax=None):
    r'''Plot ROC curve. Draws into provided ax if given, otherwise creates a new figure.'''
    dfrocpr = pd.concat(
        [dfrocpr[['FPR', 'TPR']], pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})],
        ignore_index=True
    ).sort_values(['FPR', 'TPR'])

    auc_val = (
        (dfrocpr['TPR'] + dfrocpr['TPR'].shift(1))
        * (dfrocpr['FPR'] - dfrocpr['FPR'].shift(1)) / 2
    ).sum()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    ax.plot(dfrocpr.FPR, dfrocpr.TPR, 'k-', label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.fill_between(dfrocpr.FPR, 0, dfrocpr.TPR, color='blue', alpha=0.1)

    ax.set_title(f"{title} ROC (Gini={(2 * auc_val - 1):.4f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    return (ax.figure, auc_val) if created_fig else (ax.figure, auc_val)


def eva_ppr(dfrocpr, title="", ax=None):
    r'''Plot Precision-Recall (P-R) curve. Draws into provided ax if given, otherwise creates a new figure.'''
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Plot precision vs recall
    ax.plot(dfrocpr.recall, dfrocpr.precision, 'k-', label='P-R Curve')
    # Diagonal reference (previous implementation used x==y line)
    ax.plot([0, 1], [0, 1], 'r--', label='Reference')

    ax.set_title(f"{title} P-R")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    try:
        ax.set_aspect('equal', adjustable='box')
    except Exception:
        # set_aspect may fail in some backends; ignore silently
        pass
    ax.legend(loc="lower left")

    return ax.figure if created_fig else ax.figure


# ==========================================================
# MAIN PERFORMANCE EVALUATION
# ==========================================================

def perf_eva(label, pred, title=None, groupnum=None,
             plot_type=("ks", "roc"), show_plot=True,
             positive="bad|1", seed=186):
    r'''
    KS, ROC, Lift, PR
    ------
    perf_eva provides performance evaluations, such as
    kolmogorov-smirnow(ks), ROC, lift and precision-recall curves,
    based on provided label and predicted probability values.

    Params
    ------
    label: Label values, such as 0s and 1s, 0 represent for good
      and 1 for bad.
    pred: Predicted probability or score.
    title: Title of plot, default is "performance".
    groupnum: The group number when calculating KS.  Default NULL,
      which means the number of sample size.
    plot_type: Types of performance plot, such as "ks", "lift", "roc", "pr".
      Default c("ks", "roc").
    show_plot: Logical value, default is TRUE. It means whether to show plot.
    positive: Value of positive class, default is "bad|1".
    seed: Integer, default is 186. The specify seed is used for random sorting data.

    Returns
    ------
    dict
        ks, auc, gini values, and figure objects

    Details
    ------
    Accuracy =
        true positive and true negative/total cases
    Error rate =
        false positive and false negative/total cases
    TPR, True Positive Rate(Recall or Sensitivity) =
        true positive/total actual positive
    PPV, Positive Predicted Value(Precision) =
        true positive/total predicted positive
    TNR, True Negative Rate(Specificity) =
        true negative/total actual negative
    NPV, Negative Predicted Value =
        true negative/total predicted negative

    Examples
    ------
    import scorecardpy

    # load data
    dat = sc.germancredit()

    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")

    # woe binning ------
    bins = sc.woebin(dt_sel, "creditability")
    dt_woe = sc.woebin_ply(dt_sel, bins)

    y = dt_woe.loc[:,'creditability']
    X = dt_woe.loc[:,dt_woe.columns != 'creditability']

    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X, y)

    # predicted proability
    dt_pred = lr.predict_proba(X)[:,1]

    # performace ------
    # Example I # only ks & auc values
    sc.perf_eva(y, dt_pred, show_plot=False)

    # Example II # ks & roc plot
    sc.perf_eva(y, dt_pred)

    # Example III # ks, lift, roc & pr plot
    sc.perf_eva(y, dt_pred, plot_type = ["ks","lift","roc","pr"])
    '''
    # Validation
    if len(label) != len(pred):
        raise ValueError("`label` and `pred` must have the same length.")

    # Normalize label to 0/1 to avoid check_y emitting the "Positive value ..." warning.
    # - booleans -> ints
    # - two distinct numeric values that are not {0,1} -> map smaller->0 larger->1
    lab_ser = pd.Series(label)
    lab_non_na = lab_ser.dropna()
    if lab_non_na.dtype == bool:
        label = lab_ser.astype(int).values
    else:
        unique_vals = lab_non_na.unique()
        if len(unique_vals) == 2 and not set(unique_vals).issubset({0, 1}):
            vals = sorted(unique_vals)
            label = lab_ser.map({vals[0]: 0, vals[1]: 1}).values

    # Pred range check
    if not (0 <= np.mean(pred) <= 1):
        warnings.warn("Predicted values not in [0,1]; treating as scores, flipping sign.")
        pred = -pred

    df = pd.DataFrame({'label': label, 'pred': pred}).dropna().sample(frac=1, random_state=seed)
    df = check_y(df, 'label', positive)

    title = '' if title is None else str(title) + ': '
    rt = {}

    # KS / Lift
    if any(p in plot_type for p in ['ks', 'lift']):
        dfkslift = eva_dfkslift(df, groupnum)
        #rt['KS'] = round(dfkslift['ks'].max(),4)

    # ROC / PR
    if any(p in plot_type for p in ['roc', 'pr']):
        dfrocpr = eva_dfrocpr(df)
        auc_val = roc_auc_score(df['label'], df['pred'])
        #rt['AUC'] = round(auc_val,4)
        rt['Gini'] = round(2 * auc_val - 1,4)

    # Plot section
    if show_plot:
        fig, axs = plt.subplots(
            nrows=int(np.ceil(len(plot_type) / 2)),
            ncols=int(np.ceil(len(plot_type) / np.ceil(len(plot_type) / 2))),
            figsize=(10, 5)
        )
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for i, ptype in enumerate(plot_type):
            current_ax = axs[i]
            if ptype == "ks":
                eva_pks(dfkslift, title, ax=current_ax)
            elif ptype == "lift":
                eva_plift(dfkslift, title, ax=current_ax)
            elif ptype == "roc":
                eva_proc(dfrocpr, title, ax=current_ax)
            elif ptype == "pr":
                eva_ppr(dfrocpr, title, ax=current_ax)
        plt.tight_layout()
        plt.show()

    return rt


def perf_psi(score, label=None, title=None, x_limits=None, x_tick_break=50, show_plot=True, seed=186,
             return_distr_dat=False):
    r'''
    PSI
    ------
    perf_psi calculates population stability index (PSI) and provides
    credit score distribution based on credit score datasets.

    Params
    ------
    score: A list of credit score for actual and expected data samples.
      For example, score = list(actual = score_A, expect = score_E), both
      score_A and score_E are dataframes with the same column names.
    label: A list of label value for actual and expected data samples.
      The default is NULL. For example, label = list(actual = label_A,
      expect = label_E), both label_A and label_E are vectors or
      dataframes. The label values should be 0s and 1s, 0 represent for
      good and 1 for bad.
    title: Title of plot, default is NULL.
    x_limits: x-axis limits, default is None.
    x_tick_break: x-axis ticker break, default is 50.
    show_plot: Logical, default is TRUE. It means whether to show plot.
    return_distr_dat: Logical, default is FALSE.
    seed: Integer, default is 186. The specify seed is used for random
      sorting data.

    Returns
    ------
    dict
        psi values and figure objects

    Details
    ------
    The population stability index (PSI) formula is displayed below:
    \deqn{PSI = \sum((Actual\% - Expected\%)*(\ln(\frac{Actual\%}{Expected\%}))).}
    The rule of thumb for the PSI is as follows: Less than 0.1 inference
    insignificant change, no action required; 0.1 - 0.25 inference some
    minor change, check other scorecard monitoring metrics; Greater than
    0.25 inference major shift in population, need to delve deeper.

    Examples
    ------
    import scorecardpy as sc

    # load data
    dat = sc.germancredit()

    # filter variable via missing rate, iv, identical value rate
    dt_sel = sc.var_filter(dat, "creditability")

    # breaking dt into train and test ------
    train, test = sc.split_df(dt_sel, 'creditability').values()

    # woe binning ------
    bins = sc.woebin(train, "creditability")

    # converting train and test into woe values
    train_woe = sc.woebin_ply(train, bins)
    test_woe = sc.woebin_ply(test, bins)

    y_train = train_woe.loc[:,'creditability']
    X_train = train_woe.loc[:,train_woe.columns != 'creditability']
    y_test = test_woe.loc[:,'creditability']
    X_test = test_woe.loc[:,train_woe.columns != 'creditability']

    # logistic regression ------
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
    lr.fit(X_train, y_train)

    # predicted proability
    pred_train = lr.predict_proba(X_train)[:,1]
    pred_test = lr.predict_proba(X_test)[:,1]

    # performance ks & roc ------
    perf_train = sc.perf_eva(y_train, pred_train, title = "train")
    perf_test = sc.perf_eva(y_test, pred_test, title = "test")

    # score ------
    # scorecard
    card = sc.scorecard(bins, lr, X_train.columns)
    # credit score
    train_score = sc.scorecard_ply(train, card)
    test_score = sc.scorecard_ply(test, card)

    # Example I # psi
    psi1 = sc.perf_psi(
      score = {'train':train_score, 'test':test_score},
      label = {'train':y_train, 'test':y_test},
      x_limits = [250, 750],
      x_tick_break = 50
    )

    # Example II # credit score, only_total_score = FALSE
    train_score2 = sc.scorecard_ply(train, card, only_total_score=False)
    test_score2 = sc.scorecard_ply(test, card, only_total_score=False)
    # psi
    psi2 = sc.perf_psi(
      score = {'train':train_score2, 'test':test_score2},
      label = {'train':y_train, 'test':y_test},
      x_limits = [250, 750],
      x_tick_break = 50
    )
    '''

    # inputs checking
    ## score
    if not isinstance(score, dict) and len(score) != 2:
        raise Exception("Incorrect inputs; score should be a dictionary with two elements.")
    else:
        if any([not isinstance(i, pd.DataFrame) for i in score.values()]):
            raise Exception("Incorrect inputs; score is a dictionary of two dataframes.")
        score_columns = [list(i.columns) for i in score.values()]
        if set(score_columns[0]) != set(score_columns[1]):
            raise Exception("Incorrect inputs; the column names of two dataframes in score should be the same.")
    ## label
    if label is not None:
        if not isinstance(label, dict) and len(label) != 2:
            raise Exception("Incorrect inputs; label should be a dictionary with two elements.")
        else:
            if set(score.keys()) != set(label.keys()):
                raise Exception("Incorrect inputs; the keys of score and label should be the same. ")
            for i in label.keys():
                if isinstance(label[i], pd.DataFrame):
                    if len(label[i].columns) == 1:
                        label[i] = label[i].iloc[:, 0]
                    else:
                        raise Exception("Incorrect inputs; the number of columns in label should be 1.")
    # score dataframe column names
    score_names = score[list(score.keys())[0]].columns
    # merge label with score
    for i in score.keys():
        score[i] = score[i].copy(deep=True)
        if label is not None:
            score[i].loc[:, 'y'] = label[i]
        else:
            score[i].copy(deep=True).loc[:, 'y'] = np.nan
    # dateset of score and label
    dt_sl = pd.concat(score, names=['ae', 'rowid']).reset_index() \
        .sample(frac=1, random_state=seed)

    # ae refers to 'Actual & Expected'

    # PSI function
    def psi(dat):
        dt_bae = dat.groupby(['ae', 'bin'], observed=True).size().reset_index(name='N') \
            .pivot_table(values='N', index='bin', columns='ae').fillna(0.9) \
            .agg(lambda x: x / sum(x))
        dt_bae.columns = ['A', 'E']
        psi_dt = dt_bae.assign(
            AE=lambda x: x.A - x.E,
            logAE=lambda x: np.log(x.A / x.E)
        ).assign(
            bin_PSI=lambda x: x.AE * x.logAE
        )['bin_PSI'].sum()
        return psi_dt

    # return psi and pic
    rt_psi = {}
    rt_pic = {}
    rt_dat = {}
    rt = {}
    for sn in score_names:
        # dataframe with columns of ae y sn
        dat = dt_sl[['ae', 'y', sn]]
        if len(dt_sl[sn].unique()) > 10:
            # breakpoints
            if x_limits is None:
                x_limits = dat[sn].quantile([0.02, 0.98])
                x_limits = round(x_limits / x_tick_break) * x_tick_break
                x_limits = list(x_limits)

            brkp = np.unique([np.floor(min(dt_sl[sn]) / x_tick_break) * x_tick_break] + \
                             list(np.arange(x_limits[0], x_limits[1], x_tick_break)) + \
                             [np.ceil(max(dt_sl[sn]) / x_tick_break) * x_tick_break])
            # cut
            labels = ['[{},{})'.format(int(brkp[i]), int(brkp[i + 1])) for i in range(len(brkp) - 1)]
            dat.loc[:, 'bin'] = pd.cut(dat[sn], brkp, right=False, labels=labels)
        else:
            dat.loc[:, 'bin'] = dat[sn]
        # psi ------
        rt_psi[sn] = pd.DataFrame({'PSI': psi(dat)}, index=np.arange(1))

        # distribution of scorecard probability
        def good(x):
            return sum(x == 0)

        def bad(x):
            return sum(x == 1)

        distr_prob = dat.groupby(['ae', 'bin'], observed=True) \
            ['y'].agg([good, bad]) \
            .assign(N=lambda x: x.good + x.bad,
                    badprob=lambda x: x.bad / (x.good + x.bad)
                    ).reset_index()
        distr_prob.loc[:, 'distr'] = distr_prob.groupby('ae', observed=True)['N'].transform(lambda x: x / sum(x))
        # pivot table
        distr_prob = distr_prob.pivot_table(values=['N', 'badprob', 'distr'], index='bin', columns='ae')

        # plot ------
        if show_plot:
            ###### param ######
            ind = np.arange(len(distr_prob.index))  # the x locations for the groups
            width = 0.35  # the width of the bars: can also be len(x) sequence
            ###### plot ######
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            title_string = sn + '_PSI: ' + str(round(psi(dat), 4))
            title_string = title_string if title is None else str(title) + ' ' + title_string
            # ax1
            p1 = ax1.bar(ind, distr_prob.distr.iloc[:, 0], width, color=(24 / 254, 192 / 254, 196 / 254), alpha=0.6)
            p2 = ax1.bar(ind + width, distr_prob.distr.iloc[:, 1], width, color=(246 / 254, 115 / 254, 109 / 254),
                         alpha=0.6)
            # ax2
            p3 = ax2.plot(ind + width / 2, distr_prob.badprob.iloc[:, 0], color=(24 / 254, 192 / 254, 196 / 254))
            ax2.scatter(ind + width / 2, distr_prob.badprob.iloc[:, 0], facecolors='w',
                        edgecolors=(24 / 254, 192 / 254, 196 / 254))
            p4 = ax2.plot(ind + width / 2, distr_prob.badprob.iloc[:, 1], color=(246 / 254, 115 / 254, 109 / 254))
            ax2.scatter(ind + width / 2, distr_prob.badprob.iloc[:, 1], facecolors='w',
                        edgecolors=(246 / 254, 115 / 254, 109 / 254))
            # settings
            ax1.set_ylabel('Score distribution')
            ax2.set_ylabel('Bad probability')  # , color='blue')
            # ax2.tick_params(axis='y', colors='blue')
            # ax1.set_yticks(np.arange(0, np.nanmax(distr_prob['distr'].values), 0.2))
            # ax2.set_yticks(np.arange(0, 1+0.2, 0.2))
            ax1.set_ylim([0, np.ceil(np.nanmax(distr_prob['distr'].values) * 10) / 10])
            ax2.set_ylim([0, 1])
            plt.xticks(ind + width / 2, distr_prob.index)
            plt.title(title_string, loc='left')
            ax1.legend((p1[0], p2[0]), list(distr_prob.columns.levels[1]), loc='upper left')
            ax2.legend((p3[0], p4[0]), list(distr_prob.columns.levels[1]), loc='upper right')
            # show plot
            plt.show()

            # return of pic
            rt_pic[sn] = fig

        # return distr_dat ------
        if return_distr_dat:
            rt_dat[sn] = distr_prob[['N', 'badprob']].reset_index()
    # return rt
    rt['psi'] = pd.concat(rt_psi).reset_index().rename(columns={'level_0': 'variable'})[['variable', 'PSI']]
    rt['pic'] = rt_pic
    if return_distr_dat: rt['dat'] = rt_dat
    return rt


# Calculation of IV for difference subsamples
def iv_group(smp, var_list, groupby, y):
    groups = sorted(smp[groupby].unique())
    iv_groups = pd.DataFrame({'variable': var_list})
    for i in groups:
        df_i = smp.loc[smp[groupby] == i]
        iv_i = []
        for var in var_list:
            gb_var = df_i.groupby(var, observed=True)[y].agg([n0, n1]).reset_index().rename(columns={'n0': 'good', 'n1': 'bad'})
            iv_var = iv_01(gb_var['good'], gb_var['bad'])
            iv_i.append(iv_var)
        iv_df = pd.DataFrame({
            'variable': var_list,
            i: iv_i
        })
        iv_groups = pd.merge(iv_groups, iv_df, how='left', on='variable')
    return iv_groups


def gini_vars(sample, target, vars_list, result_name):
    gini_vars = []
    for var in vars_list:
        gini_var = -(roc_auc_score(sample[target], sample[var]) * 2 - 1)
        gini_vars.append(gini_var)
    gini_vars_df = pd.DataFrame({'Variable': vars_list, result_name: gini_vars})
    return gini_vars_df


def gini_over_time(sample, target, vars_list, date):
    sorted_date = sorted(sample[date].unique())
    # del sorted_date[-12:]
    gini_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        gini_date = gini_vars(sample_date, target, vars_list, i)
        gini_date = gini_date.rename(columns={"Variable": date})
        gini_date = gini_date.set_index(date).T
        gini_ot = pd.concat([gini_ot, gini_date])
    return gini_ot


def score_ranges(sample, score, nintervals=10):
    intervals = pd.cut(sample[score], nintervals)
    intervals_unique = intervals.unique()
    output = pd.DataFrame({'range': intervals_unique})
    output['score_range'] = output['range']
    output = output.set_index(['range'])
    return output


def score_distr(sample, target, score='score', score_range='score_range'):
    sample_ranges = sample[['score_range', target]].groupby(['score_range'], observed=True).agg(['count', 'sum'])
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
    r'''Calculate the PSI for a single variable
    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into
    Returns:
       psi_value: calculated PSI value
    '''

    def sub_psi(e_perc, a_perc):
        r'''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return (value)

    iterable = (sub_psi(expected_share[x], actual_share[x]) for x in range(0, len(expected_share)))
    psi_value = np.sum(np.fromiter(iterable, float))

    return (psi_value)


def psi_vars(ref_sample, sample, vars_list, result_name):
    psi_vars = []
    for var in vars_list:
        ref_sample_groups = ref_sample.groupby([var], observed=True).size()
        ref_sample_groups_df = pd.DataFrame({'Total': ref_sample_groups})
        ref_sample_groups_df['Total_Share'] = ref_sample_groups_df['Total'] / ref_sample_groups_df['Total'].sum()

        sample_groups = sample.groupby([var], observed=True).size()
        sample_groups_df = pd.DataFrame({'Total1': sample_groups})
        sample_groups_df = pd.merge(ref_sample_groups_df, sample_groups_df, left_index=True, right_index=True,
                                    how="outer")
        sample_groups_df['Total_Share'] = sample_groups_df['Total1'] / sample_groups_df['Total1'].sum()

        ref_sample_groups_df.reset_index(drop=True, inplace=True)
        sample_groups_df.reset_index(drop=True, inplace=True)

        psi_var = psi(ref_sample_groups_df['Total_Share'], sample_groups_df['Total_Share'])
        psi_vars.append(psi_var)
    psi_vars_df = pd.DataFrame({'Variable': vars_list, result_name: psi_vars})
    return psi_vars_df


def psi_over_time(ref_sample, sample, vars_list, date):
    sorted_date = sorted(sample[date].unique())
    psi_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        psi_date = psi_vars(ref_sample, sample_date, vars_list, i)
        psi_date = psi_date.rename(columns={"Variable": date})
        psi_date = psi_date.set_index(date).T
        psi_ot = pd.concat([psi_ot, psi_date])
    return psi_ot


def get_value_counts_by_date(df, vars_woe, date_col):
    """
    Compute counts by variable and date.

    Returns:
        DataFrame with columns:
        var_name | date | var_value | cnt
    """
    results = []
    for v in vars_woe:
        temp = (
            df.groupby([date_col, v])
              .size()
              .reset_index(name='cnt')
              .rename(columns={v: 'var_value', date_col: 'date'})
        )
        temp["var_name"] = v
        results.append(temp)
    return pd.concat(results, ignore_index=True)[["var_name", "date", "var_value", "cnt"]]


def performance_testing(
    smp_testing_outcome,
    smp_testing,
    train_score,
    train_woe,
    vars_woe,
    target,
    date_col,
    test_score=None,
    test_woe=None,
    score_col='score',
    output_path="7_1_testing_results.xlsx",
    groupby_col=None
):
    """
    Compute and export model testing performance summary to Excel.

    Parameters
    ----------
    smp_testing_outcome : pd.DataFrame
        Full testing sample with actuals and scores.
    smp_testing : pd.DataFrame
        Testing sample (full) with same score/rating columns.
    train_score : pd.DataFrame
        Train sample with scores.
    train_woe : pd.DataFrame
        Train dataset with WOE-transformed variables.
    vars_woe : list
        List of WOE variable names.
    target : str
        Target variable name.
    date_col : str
        Date column name.
    test_score : pd.DataFrame, optional
        Test sample with scores. Default is None.
    test_woe : pd.DataFrame, optional
        Test dataset with WOE-transformed variables. Default is None.
    score_col : str, default 'score'
        Score column name.
    output_path : str, default '7_1_testing_results.xlsx'
        Path to Excel output file.
    groupby_col : str, optional
        Column name in smp_testing_outcome and smp_testing to split analysis by.
    """

    # --- Helper to process one subset ---
    def _run_for_subset(sub_outcome, sub_testing, group_value=None):
        sub_outcome = sub_outcome.copy()
        sub_testing = sub_testing.copy()

        # 1. PDs and Gini over time
        sub_outcome["pd"] = pd_from_score(sub_outcome[score_col])
        gini_ot = (
            sub_outcome
            .groupby(date_col)
            .agg(
                Total=(target, "count"),
                Bads=(target, "sum"),
                Avg_PD=("pd", "mean")
            )
            .reset_index()
        )
        gini_ot["Bad Rate"] = gini_ot["Bads"] / gini_ot["Total"]
        gini_df = gini_over_time(sub_outcome, target, [score_col], date_col).reset_index()
        gini_ot["Gini"] = gini_df[score_col]

        if group_value is not None:
            gini_ot[groupby_col] = group_value

        # 2. Variable Ginies
        if test_woe is not None and not test_woe.empty:
            gini_vars_train = gini_vars(train_woe, target, vars_woe, "Train")
            gini_vars_test = gini_vars(test_woe, target, vars_woe, "Test")
            gini_vars_train_test = pd.merge(gini_vars_train, gini_vars_test, on="Variable")
        else:
            gini_vars_train_test = None

        gini_vars_ot = gini_over_time(sub_outcome, target, vars_woe, date_col)
        if group_value is not None:
            gini_vars_ot[groupby_col] = group_value

        # 3. Score distributions
        _, brk = pd.cut(train_score[score_col], bins=10, retbins=True, duplicates="drop")
        brk = list(
            filter(
                lambda x: x > np.nanmin(train_score[score_col])
                and x < np.nanmax(train_score[score_col]),
                brk.round(2),
            )
        )
        brk = [np.nanmin(sub_testing[score_col])] + sorted(brk) + [np.nanmax(sub_testing[score_col])]

        train_score_local = train_score.copy()
        train_score_local["score_range"] = pd.cut(train_score_local[score_col], bins=brk, include_lowest=False)
        sub_testing["score_range"] = pd.cut(sub_testing[score_col], bins=brk, include_lowest=False)

        if test_score is not None and not test_score.empty:
            test_score_local = test_score.copy()
            test_score_local["score_range"] = pd.cut(test_score_local[score_col], bins=brk, include_lowest=False)
            test_distr = score_distr(test_score_local, target, score_col, "score_range")
        else:
            test_distr = None

        train_distr = score_distr(train_score_local, target, score_col, "score_range")
        if group_value is not None:
            train_distr[groupby_col] = group_value
            if test_distr is not None:
                test_distr[groupby_col] = group_value

        # 4. PSI, HHI, DR
        psi_ot = psi_over_time(train_score_local, sub_testing, ["score_range"], date_col)
        psi_vars_ot = psi_over_time(train_woe, sub_testing, vars_woe, date_col)
        distr_vars_ot = get_value_counts_by_date(sub_testing, vars_woe, date_col)
        hhi_ot = sub_testing.groupby(date_col).agg({"score_range": hhi}).rename(columns={"score_range": "HHI"})

        if group_value is not None:
            psi_ot[groupby_col] = group_value
            psi_vars_ot[groupby_col] = group_value
            distr_vars_ot[groupby_col] = group_value
            hhi_ot[groupby_col] = group_value

        if test_score is not None and not test_score.empty:
            train_hhi = hhi(train_score_local["score_range"].astype(str))
            test_hhi = hhi(test_score_local["score_range"].astype(str))
            hhi_train_test = pd.DataFrame({"train": [train_hhi], "test": [test_hhi]}, index=["hhi"])
        else:
            hhi_train_test = None

        # 5. Ratings
        bins = [0, 499, 539, 579, 619, 659, 699, 739, 779, 999]
        labels = ["4.5", "4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0", "0.5"]

        sub_outcome["rating"] = pd.cut(sub_outcome[score_col], bins=bins, labels=labels, include_lowest=True)
        sub_testing["rating"] = pd.cut(sub_testing[score_col], bins=bins, labels=labels, include_lowest=True)

        DR_rating = (
            sub_outcome.groupby("rating", observed=True)[target]
            .agg(["count", "sum"])
            .rename(columns={"count": "Total", "sum": "Bads"})
            .assign(DR=lambda d: d["Bads"] / d["Total"])
            .reset_index()
        )
        DR_rating_ot = (
            sub_testing.groupby([date_col, "rating"], observed=True)[target]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "Total", "sum": "Bads"})
        )
        DR_rating_ot["DR"] = DR_rating_ot["Bads"] / DR_rating_ot["Total"]

        if group_value is not None:
            DR_rating[groupby_col] = group_value
            DR_rating_ot[groupby_col] = group_value

        return {
            "gini_ot": gini_ot,
            "gini_vars_ot": gini_vars_ot,
            "gini_vars_train_test": gini_vars_train_test,
            "train_distr": train_distr,
            "test_distr": test_distr,
            "psi_ot": psi_ot,
            "psi_vars_ot": psi_vars_ot,
            "distr_vars_ot": distr_vars_ot,
            "hhi_ot": hhi_ot,
            "hhi_train_test": hhi_train_test,
            "DR_rating": DR_rating,
            "DR_rating_ot": DR_rating_ot,
        }

    # --- Run analysis (with or without grouping) ---
    if groupby_col and groupby_col in smp_testing_outcome.columns:
        results_list = []
        for val in smp_testing_outcome[groupby_col].dropna().unique():
            subset_out = smp_testing_outcome[smp_testing_outcome[groupby_col] == val]
            subset_test = smp_testing[smp_testing[groupby_col] == val]
            res = _run_for_subset(subset_out, subset_test, group_value=val)
            results_list.append(res)

        # concatenate across groups
        combined = {}
        for key in results_list[0].keys():
            dfs = [r[key] for r in results_list if r[key] is not None]
            if dfs:
                combined[key] = pd.concat(dfs, ignore_index=True)
            else:
                combined[key] = None
    else:
        combined = _run_for_subset(smp_testing_outcome, smp_testing)

    # --- Export results to Excel ---
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet, df in combined.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"✅ Performance results exported to: {output_path.resolve()}")
