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




############################################
############ Model Evaluation



def metrics_report(ytrue, preds, classnames=None):
    out = pd.DataFrame(classification_report(ytrue, preds, output_dict=True))
    if classnames is not None:
        cols = list(out.columns)
        cols[:-3] = classnames
        out.columns = cols
    return out


# Almost Direct Copy from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def confusion_matrix(y_true, y_pred, class_names=None, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    ####confusion_matrix: numpy.ndarray
        ##The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        ###Similarly constructed ndarrays can also be used.
    class_names: list-like
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure


    Issue: sort out the return figure issue (return figure will plot imge twice)

    Issue: change color map?
    """

    #df_cm = pd.crosstab(y_true, y_pred) # don't use bc it does not include col 
    # of zeros if a class is not predicted
    df_cm = pd.DataFrame(cm_sklearn(y_true, y_pred))
    if class_names is not None:
        df_cm.columns = class_names
        df_cm.index = class_names
    
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    #return fig
    return df_cm


def pretty_cm(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred)
    cm.columns.name = 'Predictions'
    cm.index.name = 'Truth'
    return cm


def plot_roc_auc(ytrue, prob_pred):
    '''
    mostly from copied form sklearn docs

    Issue: binary class only so far, refer to sklearn docs for multilabel example to mostly copy

    '''

    fpr, tpr, thresh = roc_curve(ytrue, prob_pred)
    auc_score = roc_auc_score(ytrue, prob_pred)

    #plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic\nArea Under Curve')
    plt.legend(loc="lower right")
    plt.gca().set_aspect('equal', 'box')
    #plt.show()

    return plt.gca()