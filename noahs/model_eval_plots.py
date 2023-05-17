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




################### regression model eval plots ###############################


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
    '''
    plot predicted values (x axis) against actual values (y axis)
    with a reference diagnonal x=y line
    '''
    fig, ax = plt.subplots()
    ax.plot(results_fittedvalues, data.cost, 'o', *kwargs)
    sm.graphics.abline_plot(0,.5 , ax = ax, color='orange')
    ax.set(xlabel='predicted y values', ylabel='observed y values')
    ax.margins(.1) # why is this necesary

    return plt.gca()




def residPlots(results):
    '''
    Params:
        results: (statsmodels results obj)
    '''
    # NB: taken from someone at BP


    # Get different variables for diagnostic
    residuals = results.resid
    fitted_value = results.fittedvalues
    stand_resid = results.resid_pearson
    sqrt_abs_stand_resids = np.absolute(stand_resid)**0.5
    influence = results.get_influence()
    leverage = influence.hat_matrix_diag
    smoothed = lowess(residuals,fitted_value)



    # Plot settings
    plt.rcParams["figure.figsize"] = (15,10)

    fig, ax = plt.subplots(nrows=2, ncols=2)



    # Residual vs Fitted Plot
    sns.scatterplot(x=fitted_value, y=residuals, ax=ax[0, 0])
    ax[0, 0].axhline(y=0, linestyle='dashed')
    ax[0, 0].set_xlabel('Fitted Values')
    ax[0, 0].set_ylabel('Residuals')
    ax[0, 0].set_title('Residuals vs Fitted Fitted')
    smoothed = lowess(residuals,fitted_value)
    ax[0, 0].plot(smoothed[:,0],smoothed[:,1],color = 'r')



    # Normal Q-Q plot
    sm.qqplot(residuals, fit=True, line='45',ax=ax[0, 1])
    ax[0, 1].set_title('Normal Q-Q')



    # Scale-Location Plot
    sns.scatterplot(x=fitted_value, y=sqrt_abs_stand_resids, ax=ax[1, 0])
    ax[1, 0].set_xlabel('Fitted values')
    ax[1, 0].set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
    ax[1, 0].set_title('Scale-Location Plot')
    smoothed = lowess(sqrt_abs_stand_resids,fitted_value)
    ax[1, 0].plot(smoothed[:,0],smoothed[:,1],color = 'r')



    # Residual vs Leverage Plot
    sns.scatterplot(x=leverage, y=stand_resid, ax=ax[1, 1])
    ax[1, 1].axhline(y=0, linestyle='dashed')
    ax[1, 1].set_xlabel('Leverage')
    ax[1, 1].set_ylabel('Standardized residuals')
    ax[1, 1].set_title('Residuals vs Leverage Plot')
    smoothed = lowess(stand_resid,leverage)
    ax[1, 1].plot(smoothed[:,0],smoothed[:,1],color = 'r')



    xpos = max(leverage)*1.02
    cooksx = np.linspace(min(leverage),xpos, 50)
    p = len(results.params)
    poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
    poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
    negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
    negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)



    ax[1, 1].plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
    ax[1, 1].plot(cooksx,poscooks05y, ls = ':', color = 'r')
    ax[1, 1].plot(cooksx,negcooks1y, ls = ':', color = 'r')
    ax[1, 1].plot(cooksx,negcooks05y, ls = ':', color = 'r')



    ax[1, 1].annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
    ax[1, 1].annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
    ax[1, 1].annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
    ax[1, 1].annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
    ax[1, 1].legend()
    ax[1,1].set_xlim([min(leverage)*0.98, max(leverage)*1.02])
    ax[1,1].set_ylim([np.floor(min(stand_resid)),np.ceil(max(stand_resid))])



    fig.suptitle('Diagnostic Plots', fontsize=18)
    plt.tight_layout()
    plt.show()





################ Plots for classification tasks ############################

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


def plot_roc_auc(ytrue, prob_pred):
    '''
    mostly from copied form sklearn docs, get link...

    Issue: binary class only so far, refer to sklearn docs for multilabel example to mostly copy

    NB: there is a sklearn version of this in newer versions
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


def plot_resid_categorical(resid, feature):
    '''
    Plot residuals (y) against a categorical factor (x)
    # how are residuaks calcualted? proba pred residuals?
    '''
    plt.plot(feature,resid, 'o')
    plt.axhline(color='orange')
    plt.title('Residuals vs Predictor')
    plt.ylabel('Residuals')
    plt.xlabel('Predictor')

    ax = offset_xlabels(plt.gca())

    return ax