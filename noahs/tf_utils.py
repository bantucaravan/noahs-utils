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




#####################
####### tf  model eval


def test_train_curve(history, metric, ax=None, save=False):
    '''
    # was going to run this in plot_tf_training did not so as to ease 
    # plotting in subplot, I could easily change to have this func plot
    #  in subplot
    
    save -- pass full file path for img to be saved file 

    Issue: add pretty formatting and naming for plot labels

    Issue: - add optional model name (data prep and model type) to train test 
    graph and epoch number
    '''

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(history.epoch, history.history[metric], label='Train '+metric)
    ax.plot(history.epoch, history.history['val_'+metric], label='Test '+metric)
    ax.set(title='Train vs Test ' + metric)
    ax.legend()

    #Add model name and epoch
    #text = 'Model: \nEpoch: %d' %(len(history.epoch))
    #boxstyle = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax.text(1.05, 0.95, text, transform=ax.transAxes, fontsize=12,
    #va='top', ha='left', bbox=boxstyle)

    if save:
        # do some path validation...?
        #savepath = '../figs/Train Test %s %s.png' %(metric, save)
        dirpath = os.path.split(save)[0]
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(path)
    

    return ax




def plot_tf_training(history, metric='accuracy', save=False):
    '''
    save -- pass full file path for img to be saved file
    '''
    fig, axes = plt.subplots(2,1, figsize=(5,8))

    test_train_curve(history, metric='loss', ax=axes[0])
    test_train_curve(history, metric=metric, ax=axes[1])

    plt.tight_layout() # change to fig?

    if save:
        # do some path validation...?
        #savepath = '../figs/Train Test %s %s.png' %(metric, save)
        dirpath = os.path.split(save)[0]
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save)
    
    #plt.show()


def top_epochs(history, metric='accuracy', top_n=-1):
    '''
    Issue: top_n not implemented because [:top_n] where top_n==-1 was 
    cutting off the last  value.
    '''
    res = dict(zip(history.epoch, history.history['val_'+metric]))
    #best_val_acc, best_val_acc_epoch = float(max(res.values())),  int(max(res, key=res.get))

    print('Best %s by epoch (1-indexed):' %(metric))
    reverse = False if metric in ['loss'] else True
    out = sort_dict(res, reverse=reverse)
    # + 1 for 1-indexing
    return {k+1:v for k,v in out.items()}
    
    # bad implementation of top_n
    #return {k+1:out[k] for k in list(out.keys())[:top_n]}

