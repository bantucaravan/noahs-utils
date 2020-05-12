from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix as cm_sklearn
import seaborn as sns
import statsmodels.api as sm


# fix this
from noahs.plt_utils import offset_xlabels








##################################################
################### regression plot utils


def plot_resid_predicted(yhat, resid, scale_equal=True):
    plt.plot(yhat, resid, 'o')
    plt.axhline(color='orange')
    plt.title('Residuals vs Predicted Values')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Values')
    if scale_equal:
        plt.axis('equal') # so that we see errors on same scale as range of yhat


    return plt.gca()


def plot_pred_obsv(data, results_fittedvalues, *kwargs):
    fig, ax = plt.subplots()
    ax.plot(results_fittedvalues, data.cost, 'o', *kwargs)
    sm.graphics.abline_plot(0,.5 , ax = ax, color='orange')
    ax.set(xlabel='predicted y values', ylabel='observed y values')
    ax.margins(.1) # why is this necesary

    return plt.gca()


def plot_resid_categorical(resid, feature):
    plt.plot(feature,resid, 'o')
    plt.axhline(color='orange')
    plt.title('Residuals vs Predictor')
    plt.ylabel('Residuals')
    plt.xlabel('Predictor')

    ax = offset_xlabels(plt.gca())

    return ax

def plot_roc_auc(ytrue, prob_pred):
    '''
    mostly from copied form sklearn docs, get link...

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