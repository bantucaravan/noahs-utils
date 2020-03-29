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





##################################################
################### regression plot utils


def plot_resid_predicted(yhat, resid):
    plt.plot(yhat, resid, 'o')
    plt.axhline(color='orange')
    plt.title('Residuals vs Predicted Values')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Values')
    plt.axis('equal') # so that we see errors on same scale as range of yhat


    return plt.gca()


def plot_pred_obsv(data, results_fittedvalues, *kwargs):
    fig, ax = plt.subplots()
    ax.plot(results_fittedvalues, data.cost, 'o', *kwargs)
    sm.graphics.abline_plot(0,.5 , ax = ax, color='orange')
    ax.set(xlabel='predicted y values', ylabel='observed y values')
    ax.margins(.1) # why is this necesary

    return plt.gca()