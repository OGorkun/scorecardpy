# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from .condition_fun import *
from .info_value import iv_01
from .woebin import n0, n1
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc


def eva_dfkslift(df, groupnum=None):
    if groupnum is None: groupnum = len(df.index)

    # good bad func
    def n0(x): return sum(x == 0)

    def n1(x): return sum(x == 1)

    df_kslift = df.sort_values('pred', ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / groupnum))) \
        .groupby('group')['label'].agg([n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .assign(
        group=lambda x: (x.index + 1) / len(x.index),
        good_distri=lambda x: x.good / sum(x.good),
        bad_distri=lambda x: x.bad / sum(x.bad),
        badrate=lambda x: x.bad / (x.good + x.bad),
        cumbadrate=lambda x: np.cumsum(x.bad) / np.cumsum(x.good + x.bad),
        lift=lambda x: (np.cumsum(x.bad) / np.cumsum(x.good + x.bad)) / (sum(x.bad) / sum(x.good + x.bad)),
        cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
        cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
    ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))
    # bind 0
    df_kslift = pd.concat([
        pd.DataFrame(
            {'group': 0, 'good': 0, 'bad': 0, 'good_distri': 0, 'bad_distri': 0, 'badrate': 0, 'cumbadrate': np.nan,
             'cumgood': 0, 'cumbad': 0, 'ks': 0, 'lift': np.nan}, index=np.arange(1)),
        df_kslift
    ], ignore_index=True)
    # return
    return df_kslift


# plot ks
def eva_pks(dfkslift, title):
    dfks = dfkslift.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]
    ###### plot ######
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.ks, 'b-',
             dfkslift.group, dfkslift.cumgood, 'k-',
             dfkslift.group, dfkslift.cumbad, 'k-')
    # ks vline
    plt.plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    # set xylabel
    plt.gca().set(title=title + 'K-S',
                  xlabel='% of population', ylabel='% of total Good/Bad',
                  xlim=[0, 1], ylim=[0, 1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'K-S', fontsize=15,horizontalalignment='center')
    plt.text(0.2, 0.8, 'Bad', horizontalalignment='center')
    plt.text(0.8, 0.55, 'Good', horizontalalignment='center')
    plt.text(dfks['group'], dfks['ks'], 'KS:' + str(round(dfks['ks'], 4)), horizontalalignment='center', color='b')
    # plt.grid()
    # plt.show()
    # return fig


# plot lift
def eva_plift(dfkslift, title):
    badrate_avg = sum(dfkslift.bad) / sum(dfkslift.good + dfkslift.bad)
    ###### plot ######
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.cumbadrate, 'k-')
    # ks vline
    plt.plot([0, 1], [badrate_avg, badrate_avg], 'r--')
    # set xylabel
    plt.gca().set(title=title + 'Lift',
                  xlabel='% of population', ylabel='% of Bad',
                  xlim=[0, 1], ylim=[0, 1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'Lift', fontsize=15,horizontalalignment='center')
    plt.text(0.7, np.mean(dfkslift.cumbadrate), 'cumulate badrate', horizontalalignment='center')
    plt.text(0.7, badrate_avg, 'average badrate', horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig


def eva_dfrocpr(df):
    def n0(x): return sum(x == 0)

    def n1(x): return sum(x == 1)

    dfrocpr = df.sort_values('pred') \
        .groupby('pred')['label'].agg([n0, n1, len]) \
        .reset_index().rename(columns={'n0': 'countN', 'n1': 'countP', 'len': 'countpred'}) \
        .assign(
        FN=lambda x: np.cumsum(x.countP),
        TN=lambda x: np.cumsum(x.countN)
    ).assign(
        TP=lambda x: sum(x.countP) - x.FN,
        FP=lambda x: sum(x.countN) - x.TN
    ).assign(
        TPR=lambda x: x.TP / (x.TP + x.FN),
        FPR=lambda x: x.FP / (x.TN + x.FP),
        precision=lambda x: x.TP / (x.TP + x.FP),
        recall=lambda x: x.TP / (x.TP + x.FN)
    ).assign(
        F1=lambda x: 2 * x.precision * x.recall / (x.precision + x.recall)
    )
    return dfrocpr


# plot roc
def eva_proc(dfrocpr, title):
    dfrocpr = pd.concat(
        [dfrocpr[['FPR', 'TPR']], pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})],
        ignore_index=True).sort_values(['FPR', 'TPR'])
    auc = dfrocpr.sort_values(['FPR', 'TPR']) \
        .assign(
        TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
    ).assign(
        auc=lambda x: (x.TPR + x.TPR_lag) * (x.FPR - x.FPR_lag) / 2
    )['auc'].sum()
    ###### plot ######
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.FPR, dfrocpr.TPR, 'k-')
    # ks vline
    x = np.array(np.arange(0, 1.1, 0.1))
    plt.plot(x, x, 'r--')
    # fill
    plt.fill_between(dfrocpr.FPR, 0, dfrocpr.TPR, color='blue', alpha=0.1)
    # set xylabel
    plt.gca().set(title=title + 'ROC',
                  xlabel='FPR', ylabel='TPR',
                  xlim=[0, 1], ylim=[0, 1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'ROC', fontsize=15, horizontalalignment='center')
    plt.text(0.55, 0.45, 'AUC:' + str(round(auc, 4)), horizontalalignment='center', color='b')
    # plt.grid()
    # plt.show()
    # return fig


# plot ppr
def eva_ppr(dfrocpr, title):
    ###### plot ######
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr.recall, dfrocpr.precision, 'k-')
    # ks vline
    x = np.array(np.arange(0, 1.1, 0.1))
    plt.plot(x, x, 'r--')
    # set xylabel
    plt.gca().set(title=title + 'P-R',
                  xlabel='Recall', ylabel='Precision',
                  xlim=[0, 1], ylim=[0, 1], aspect='equal')
    # text
    # plt.text(0.5,0.96, 'P-R', fontsize=15, horizontalalignment='center')
    # plt.grid()
    # plt.show()
    # return fig


# plot f1
def eva_pf1(dfrocpr, title):
    dfrocpr = dfrocpr.assign(pop=lambda x: np.cumsum(x.countpred) / sum(x.countpred))
    ###### plot ######
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfrocpr['pop'], dfrocpr['F1'], 'k-')
    # ks vline
    F1max_pop = dfrocpr.loc[dfrocpr['F1'].idxmax(), 'pop']
    F1max_F1 = dfrocpr.loc[dfrocpr['F1'].idxmax(), 'F1']
    plt.plot([F1max_pop, F1max_pop], [0, F1max_F1], 'r--')
    # set xylabel
    plt.gca().set(title=title + 'F1',
                  xlabel='% of population', ylabel='F1',
                  xlim=[0, 1], ylim=[0, 1], aspect='equal')
    # pred text
    pred_0 = dfrocpr.loc[dfrocpr['pred'].idxmin(), 'pred']
    pred_F1max = dfrocpr.loc[dfrocpr['F1'].idxmax(), 'pred']
    pred_1 = dfrocpr.loc[dfrocpr['pred'].idxmax(), 'pred']
    if np.mean(dfrocpr.pred) < 0 or np.mean(dfrocpr.pred) > 1:
        pred_0 = -pred_0
        pred_F1max = -pred_F1max
        pred_1 = -pred_1
    plt.text(0, 0, 'pred \n' + str(round(pred_0, 4)), horizontalalignment='left', color='b')
    plt.text(F1max_pop, 0, 'pred \n' + str(round(pred_F1max, 4)), horizontalalignment='center', color='b')
    plt.text(1, 0, 'pred \n' + str(round(pred_1, 4)), horizontalalignment='right', color='b')
    # title F1
    plt.text(F1max_pop, F1max_F1, 'F1 max: \n' + str(round(F1max_F1, 4)), horizontalalignment='center', color='b')
    # plt.grid()
    # plt.show()
    # return fig


def perf_eva(label, pred, title=None, groupnum=None, plot_type=["ks", "roc"], show_plot=True, positive="bad|1",
             seed=186):
    '''
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

    # inputs checking
    if len(label) != len(pred):
        warnings.warn('Incorrect inputs; label and pred should be list with the same length.')
    # if pred is score
    if np.mean(pred) < 0 or np.mean(pred) > 1:
        warnings.warn(
            'Since the average of pred is not in [0,1], it is treated as predicted score but not probability.')
        pred = -pred
    # random sort datatable
    df = pd.DataFrame({'label': label, 'pred': pred}).sample(frac=1, random_state=seed)
    # remove NAs
    if any(np.unique(df.isna())):
        warnings.warn('The NANs in \'label\' or \'pred\' were removed.')
        df = df.dropna()
    # check label
    df = check_y(df, 'label', positive)
    # title
    title = '' if title is None else str(title) + ': '

    ### data ###
    # dfkslift ------
    if any([i in plot_type for i in ['ks', 'lift']]):
        dfkslift = eva_dfkslift(df, groupnum)
        if 'ks' in plot_type: df_ks = dfkslift
        if 'lift' in plot_type: df_lift = dfkslift
    # dfrocpr ------
    if any([i in plot_type for i in ["roc", "pr", 'f1']]):
        dfrocpr = eva_dfrocpr(df)
        if 'roc' in plot_type: df_roc = dfrocpr
        if 'pr' in plot_type: df_pr = dfrocpr
        if 'f1' in plot_type: df_f1 = dfrocpr
    ### return list ###
    rt = {}
    # plot, KS ------
    if 'ks' in plot_type:
        rt['KS'] = round(dfkslift.loc[lambda x: x.ks == max(x.ks), 'ks'].iloc[0], 4)
    # plot, ROC ------
    if 'roc' in plot_type:
        auc = pd.concat(
            [dfrocpr[['FPR', 'TPR']], pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})],
            ignore_index=True).sort_values(['FPR', 'TPR']) \
            .assign(
            TPR_lag=lambda x: x['TPR'].shift(1), FPR_lag=lambda x: x['FPR'].shift(1)
        ).assign(
            auc=lambda x: (x.TPR + x.TPR_lag) * (x.FPR - x.FPR_lag) / 2
        )['auc'].sum()
        ###
        rt['AUC'] = round(auc, 4)
        rt['Gini'] = round(2 * auc - 1, 4)

    ### export plot ###
    if show_plot:
        plist = ["eva_p" + i + '(df_' + i + ',title)' for i in plot_type]
        subplot_nrows = np.ceil(len(plist) / 2).astype(int)
        subplot_ncols = np.ceil(len(plist) / subplot_nrows).astype(int)

        fig = plt.figure()
        for i in np.arange(len(plist)):
            plt.subplot(subplot_nrows, subplot_ncols, i + 1)
            eval(plist[i])
        plt.show()
        rt['pic'] = fig
    # return
    return rt


def perf_psi(score, label=None, title=None, x_limits=None, x_tick_break=50, show_plot=True, seed=186,
             return_distr_dat=False):
    '''
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
        dt_bae = dat.groupby(['ae', 'bin']).size().reset_index(name='N') \
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

        distr_prob = dat.groupby(['ae', 'bin']) \
            ['y'].agg([good, bad]) \
            .assign(N=lambda x: x.good + x.bad,
                    badprob=lambda x: x.bad / (x.good + x.bad)
                    ).reset_index()
        distr_prob.loc[:, 'distr'] = distr_prob.groupby('ae')['N'].transform(lambda x: x / sum(x))
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
            gb_var = df_i.groupby(var)[y].agg([n0, n1]).reset_index().rename(columns={'n0': 'good', 'n1': 'bad'})
            iv_var = iv_01(gb_var['good'], gb_var['bad'])
            iv_i.append(iv_var)
        iv_df = pd.DataFrame({
            'variable': var_list,
            i: iv_i
        })
        iv_groups = pd.merge(iv_groups, iv_df, how='left', on='variable')
    return iv_groups


def gini_vars(sample, target, vars_list, result_name, weight=None):
    gini_vars = []
    for var in vars_list:
        if weight is not None:
            sample_target =  pd.Series([1]*len(sample) + [0]*len(sample))
            sample_var = pd.concat([sample[var], sample[var]], axis=0)
            sample_weight = pd.concat([sample[target]*sample[weight], (1 - sample[target])*sample[weight]], axis=0)
        else:
            sample_target = sample[target]
            sample_var = sample[var]
            sample_weight = None
        gini_var = -(roc_auc_score(sample_target, sample_var, sample_weight=sample_weight) * 2 - 1)
        gini_vars.append(gini_var)
    gini_vars_df = pd.DataFrame({'Variable': vars_list, result_name: gini_vars})
    return gini_vars_df


def gini_over_time(sample, target, vars_list, date, weight=None):
    sorted_date = sorted(sample[date].unique())
    # del sorted_date[-12:]
    gini_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        gini_date = gini_vars(sample_date, target, vars_list, result_name = i, weight=weight)
        gini_date = gini_date.rename(columns={"Variable": "Period"})
        gini_date = gini_date.set_index('Period').T
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
    sample_ranges = sample[['score_range', target]].groupby(['score_range']).agg(['count', 'sum'])
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
        return (value)

    iterable = (sub_psi(expected_share[x], actual_share[x]) for x in range(0, len(expected_share)))
    psi_value = np.sum(np.fromiter(iterable, float))

    return (psi_value)


def psi_vars(ref_sample, sample, vars_list, result_name, n_obs=None):
    psi_vars = []
    for var in vars_list:
        if n_obs is None:
            ref_sample_groups = ref_sample.groupby([var]).size()
        else:
            ref_sample_groups = ref_sample[[n_obs, var]].groupby([var]).sum()[n_obs]

        ref_sample_groups_df = pd.DataFrame({'Total': ref_sample_groups})
        ref_sample_groups_df['Total_Share'] = ref_sample_groups_df['Total'] / ref_sample_groups_df['Total'].sum()

        if n_obs is None:
            sample_groups = sample.groupby([var]).size()
        else:
            sample_groups = sample[[n_obs, var]].groupby([var]).sum()[n_obs]
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


def psi_over_time(ref_sample, sample, vars_list, date, n_obs=None):
    sorted_date = sorted(sample[date].unique())
    psi_ot = pd.DataFrame([])
    for i in sorted_date:
        sample_date = sample.loc[sample[date] == i]
        psi_date = psi_vars(ref_sample, sample_date, vars_list, i, n_obs)
        psi_date = psi_date.rename(columns={"Variable": date})
        psi_date = psi_date.set_index(date).T
        psi_ot = pd.concat([psi_ot, psi_date])
    return psi_ot


def psi_prev_period_ot(sample, vars_list, date, n_obs=None):
    sorted_date = sorted(sample[date].unique())
    psi_ot = pd.DataFrame([])
    for i in range(len(sorted_date)-1):
        sample_date = sample.loc[sample[date] == sorted_date[i+1]]
        ref_sample = sample.loc[sample[date] == sorted_date[i]]
        psi_date = psi_vars(ref_sample, sample_date, vars_list, sorted_date[i+1], n_obs)
        psi_date = psi_date.rename(columns={"Variable": date})
        psi_date = psi_date.set_index(date).T
        psi_ot = pd.concat([psi_ot, psi_date])
    return psi_ot