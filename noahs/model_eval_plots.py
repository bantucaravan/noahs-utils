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


# NB: its hard to write unit tests for plots (I think)





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





def calibration_plot(true_preds, proba_preds, n_bins = 5, return_fig=False):
    '''
    plot calibration of predicted "probabilities" from ML models, 
    e.g. does a .25 prediction from a random forest model really have a 25% 
    chance of being an accurate prediction.

    From version 1.0 sklearn has a function for this `CalibrationDisplay.from_predictions()` 
    ( https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html#sklearn.calibration.CalibrationDisplay.from_predictions )
    however my version includes historgram of probabilities in the plot which is not included in the skelarn function.
    '''

    counts, edges = np.histogram(proba_preds, n_bins)
    edges[-1] = edges[-1] + .0001 # account for the right most bin in np.digitize having an open (rather than closed) boundary and np.histogram returning bin edges with the right most bin having a closed boundary
    bucket_ids = np.digitize(proba_preds, edges)
    observed_probs = pd.Series(true_preds).groupby(bucket_ids).mean()
    stated_probs = pd.Series(proba_preds).groupby(bucket_ids).mean()

    fig, ax1 = plt.subplots()
    ax1.plot(stated_probs, observed_probs, 'o-')
    ax1.plot(np.linspace(0,1,10), np.linspace(0,1,10), '--')
    ax1.set(ylabel='Observed Probabilility (calculated within bin)', xlabel='Stated Probability', title='Probability Calibration Plot \n (Positive Class)')
    ax2 = ax1.twinx()
    ax2.bar(edges[:-1], counts, width=edges[1:]-edges[:-1], align='edge', alpha = .3)
    ax2.set(ylabel='Count of observations in bin')
    plt.show()

    if return_fig:
        return fig


