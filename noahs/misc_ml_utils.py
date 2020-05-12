from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix as cm_sklearn
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold

import inspect
import json
import os
import re
import warnings
import pickle
import copy
from collections import defaultdict






###############################################
####### ML utils

def xgb_optim(x_train, y_train, class_weight=None):
    '''
    Descrip: pass training data and get returned a xgb model fitted on training 
    data with optimal hyper parameters. Hyper parameter search starts from fixed 
    reasonable starting values.
    '''

    ## (1) Num iters optimize step
    def num_iters_optim(alg, xgtrain):

        eval_hist = xgb.cv(
            alg.get_xgb_params(),
            xgtrain,
            num_boost_round= 1000,
            nfold=5,
            metrics='auc',
            early_stopping_rounds=50,
            verbose_eval=False
            )
        best_num_iters = eval_hist.shape[0]
        alg.set_params(n_estimators = best_num_iters)
        
        display(pd.Series(alg.get_params()))
        display(eval_hist.loc[best_num_iters-1, eval_hist.columns.isin(['test-auc-mean'])])

        return alg

    ## (2) Optimize other params step
    def param_optim(alg, params):
        
        #cw = compute_sample_weight(class_weight='balanced', y=y_train)
        cv = GridSearchCV(estimator=alg, param_grid = params, scoring='roc_auc', # 'accuracy',#
        n_jobs=4, iid=False, cv=5, refit=False) # True) #
        cv.fit(x_train, y_train)#, sample_weight=cw)
        #alg = cv.best_estimator_
        # set refit=False and do below to not retrain on whole set, to save time
        alg.set_params(**cv.best_params_)

        display(pd.DataFrame(cv.cv_results_)[['params','mean_test_score', 'std_test_score','rank_test_score']].sort_values('rank_test_score'))
        display(pd.Series(alg.get_params()))

        return alg


    if class_weight == 'ratio_weight':
        cw = np.divide.accumulate(np.unique(y_train, return_counts=True)[1])[1]
    else:
        cw = 1

    # initialize model hyper params
    alg = XGBClassifier(
        learning_rate =0.1,
        #n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        #nthread=4,
        scale_pos_weight= cw,# 1, #
        #seed=27
        )

    # xgb data format
    xgtrain = xgb.DMatrix(x_train, label= y_train)

    # get optimal num boosting iterations given fixed high learning rate 
    # (for faster convergence during hyper param search fits) and reasonable 
    # starting tree level parameters
    alg = num_iters_optim(alg, xgtrain)

    # max_depth & min_child_weight
    params= {'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}
    alg = param_optim(alg, params)

    md = alg.get_params()['max_depth']; mcw = alg.get_params()['min_child_weight']
    params= {'max_depth':[md-1,md,md+1], 'min_child_weight':[mcw-1,mcw,mcw+1]}
    alg = param_optim(alg, params)

    # gamma
    params= {'gamma':[i/10.0 for i in range(0,5)]}
    alg = param_optim(alg, params)

    # reset num iters
    alg = num_iters_optim(alg, xgtrain)

    # subsample and colsample_bytree 
    params = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
    alg = param_optim(alg, params)

    ss= int(alg.get_params()['subsample']*100); csbt= int(alg.get_params()['colsample_bytree']*100)
    params = {'subsample':[i/100.0 for i in range(ss-5,ss+10,5)], 'colsample_bytree':[i/100.0 for i in range(csbt-5,csbt+10,5)]}
    alg = param_optim(alg, params)

    # reg_alpha
    params = {'reg_alpha':[0, 1e-5, 0.001, 0.005, 1e-2, 0.1, 1, 100]}
    alg = param_optim(alg, params)

    # final small learning rate
    alg.set_params(learning_rate = 0.01)
    alg = num_iters_optim(alg, xgtrain)

    # fit optimal hyper parameters on full train data
    alg.fit(x_train, y_train)

    return alg




