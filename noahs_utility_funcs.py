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


###########################################
################ general utils


# Download this file (in ipython) with:
# !curl -L -o noahs_utils.py https://gist.github.com/bantucaravan/1956003e25c056c550a088542b41dc91/raw/noahs_utility_funcs.py

#Inspo: https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook

# fix 8 space default

def allcols(df, rows=False):
    with pd.option_context('display.max_columns', df.shape[-1]):
        display(df)

def allrows(df):
    with pd.option_context('display.max_rows', None):#len(df)):
        display(df)

def allcolsrows(df):
    with pd.option_context('display.max_columns', df.shape[-1], 'display.max_rows', df.shape[0]):
        display(df)

allrowscols = allcolsrows

def fullcolwidth(df):
    with pd.option_context('display.max_colwidth', -1):#len(df)): # None did not work
        display(df)
        
def show_group(grouped_df, idx=None):
    '''
    Return group from grouped df by group numeric index not label.
    Returns random group if no index is passed.
    Useful for exploring groups.
    
    :grouped_df: obvs
    
    :idx: the numeric index of the group as would be returned if iterating through the grouped df.
    '''
    if idx is None:
        idx = np.random.randint(len(grouped_df))
    
    tup = list(grouped_df)[idx]
    print(tup[0])
    
    return tup[1]


def insert(text, breaks, insertion='\n'):
    '''
    insert substr at arbitrary indexes in str.

    Issue: what about inserting at single insertion point?

    Issue: are insertion points 0-indexed or 1-indexed?

    '''
    bits = [text[breaks[i-1]:breaks[i]] for i in range(1, len(breaks))]
    if breaks[0] != 0:
        bits = [text[:breaks[0]]] + bits
    if breaks[-1] != (len(text)):
            bits = bits + [text[breaks[-1]:]]


def sort_dict(dicti, reverse=True):
    assert isinstance(dicti, dict)
    out = {k: v for k, v in sorted(dicti.items(), key=lambda item: item[1], reverse=reverse)}
    return out

def load_pickle(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
        return out
    
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)



# stolen verbatim from  https://github.ibm.com/vterpstra/CPD25_write_data_asset/blob/master/assets/jupyterlab/FileAccessTests.ipynb
def list_file_hierarchy(startpath):
    """Hierarchically print the contents of the folder tree, starting with the `startpath`.

    Usage::

        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        parent_dir_2 = os.path.abspath(os.path.join(parent_dir, os.pardir))
        list_file_hierarchy(parent_dir_2) #List tree starting at the grand-parent of the current directory


    Args:
        startpath (str): Root of the tree

    Returns:
        None

    
    Issue: return string optionally (to write to disk or search/regex through), 
    maybe add a native grep like regex functionality

    Issue: include option to only return directory structure not files.
    """
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))




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

############################################
########## Deep Learning Image processing

def plot_image(pixels, ax=None):
    """
    From https://raw.githubusercontent.com/hellodanylo/ucla-deeplearning/master/02_cnn/utils.py
    Simply plots an image from its pixels.
    Pixel values must be either integers in [0, 255], or floats in [0, 1].
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(pixels)
    ax.axis('off')
    return ax
    #plt.show()
    
##########################################
############### Proj Management utils
    
    

# write json - write json (not append) to disk
def write_json(dct, path, **kwargs):
    '''
    write json to disk

    path -- file path

    **kwargs -- passed to json.dump()

    issues:
    validate .json file extension?
    '''
    with open(path, 'wt') as f:
        json.dump(dct, f, **kwargs)


def integer_keys(dct):
    '''
    Convert any string dict keys that represent digits to int type

    # for use as object_hook= in json.load()

    # Issue: convert all single numerics (floats and negatives too) back, 
    # numerics in lists and in values position are already converted
    '''

    if any(k.isdigit() for k in dct):
        return {int(k) if k.isdigit() else k:v for k,v in dct.items()}
    return dct



# read json - read json from disk to memory
def read_json(path, **kwargs):
    '''
    read json from disk

    path -- file path

    issue: validate .json file extension?
    '''
    with open(path, 'rt') as f:
        dct = json.load(f, **kwargs)
    return dct

#BRIAN: 
# write log json: handle if file doesn’t exist yet, and is empty; 
#*with key validation handle if key is string ,try catch read_jsonlog conversion from string .. or use some native json module default()
# * handele more complex objs…(else return str(obJ)? 
# * handle jsone encode error partial writing to disk
# * also loook up existing json logging options



def read_log_json(run_num=None, logfile='../logs/model logs (master file).json', object_hook=integer_keys):
    '''
    Description: read entire log json into memory, optionally return only specific single 
    (or multiple) run logs
    
    Issues:
    * valudate .json file ext?
    * currently only single not multiple run num specification supported

    '''

    outlog = read_json(logfile, object_hook=object_hook)
    # all json keys (or all json keys and values? NO) must be str. I am 
    # assuming that keys can be converted by int()
    #outlog = {int(k): v for k,v in outlog.items()}
    if run_num is not None:
        outlog = {run_num: outlog[run_num]} # for compatibilty with read_log_df expectations
        #return outlog[run_num]

    return outlog


class NumpyEncoder(json.JSONEncoder):
    '''
    for use in converting non-json serializable obj types into json serializable 
    types in write_log_json()

    See for explanation: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
   
    Issue: overly braod - if not np array or np numeric, convert to string.. be more specific
    
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        else: # is this too broad...?
            return str(obj)
        return json.JSONEncoder.default(self, obj)



def write_log_json(json_log, logfile='../logs/model logs (master file).json', **kwargs):
    '''
    Description: take a json log for a single (or multiple) runs, and .update() 
    master json log so that entries for that run number are overwritten

    json_log -- must be a dict with single (or multiple) integer keys

    **kwargs -- passed to json.dump() 

    Issues:
    * valudate .json file ext?
    * currently only single not multiple run num specification supported
    * allows possible overwrite of all logs with incorrect information! (must save back ups of log json file regularly!)


    '''
    

    if os.path.getsize(logfile) > 0: # if file in not empty 
        #object_hook=integer_keys insures integer keys are converted into python ints
        try:
            master_log = read_json(path=logfile, object_hook=integer_keys) 
            old_log = master_log.copy()
            master_log.update(json_log)
            empty=False
        except json.JSONDecodeError as e:
            msg = 'JSON file misformatted, failed to read.'
            raise json.JSONDecodeError(msg, e.doc, e.pos)

    else:
        master_log = json_log # assuming key in run uuid
        empty=True
    

    try:
        write_json(master_log, logfile, **kwargs)
    except TypeError as e:
            # Overwrite file just in case in raising error json wrote a partial, 
            # unreadable json string
        if not empty:
            write_json(old_log, logfile, **kwargs)
        else:
            open(logfile,'wt').close() # earses file contents
        raise type(e)('Failed to write JSON because non-json serializable type was passed.')




def read_log_df(run_num=None, logfile='../logs/model logs (master file).json'):
    '''
    issues:
    * use run_nums as index?
    * currently only supports selecting single not multiple run_nums

    '''
    dct = read_log_json(run_num, logfile=logfile)
    df = json_normalize(list(dct.values()))
    df.index = dct.keys()
    df = df.dropna(axis=1, how='all')
    return df


def try_args(arg_dict, method):
    '''
    For passing a dict of a super set of acceptable args to a function, 
    removing unacceptable args until a maximal subset of acceptable args 
    is reached.
    
    '''
    dct = arg_dict.copy()
    while len(dct)>0:
        try:
            method(**dct)
            break
        except TypeError as e:
            print(e.args)
            badkey = re.findall(r"'(\w+)'", str(e))[0]
            del dct[badkey]
    return dct



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








##################################################
################## statsmodels utils


def old_kfold_statsmodels(model, metrics, n_splits=5, fit_args=None):
    '''
    fit_args: (Dict) args passed to .fit() method


    Issue: record inital model  coefs, and model coefs at each Fold, to check
     for variation in coefs over data... robustness

    Issue[closed]: why not update vars(model) instead of passing kwargs even for exog and endog, 
    A: because if init doesn't get groups= for MixedLM, it will throw an error/. Also, 
    kwargs are only things that were explicitly passed to init, 

    Issue: what if the version of init args in vars(model) is somehow transformed. Also what
     about init arg elements that depend on data size, like group for example where a group may
      not be present in the training fold.

    Issue: turn in to class, with an init where you pass the model and any init args you would
     pass to the model, and fit method which took fit args and ran kfold. Why?, to allow passing
      of init args directly and not retrivea from vars(model) without mixing init and fit args

    Issue[closed]: why not pass statsmodels model type and endog, and exog vars to function and 
    not initialized model? Reasons: (a) to get the from formula data transformations? 
    rebuf: include a from formula bool arg, (b) to capture all the init args passed 
    in the original model. But what if their processing depends on the subset of 
    endog/exog data
    '''

    # validate metrics
    metrics = metrics if isinstance(metrics, list) else [metrics]
    m_valid = {'acc': accuracy_score, 'roc':roc_auc_score, 'r2':r2_score, 'mae': mean_absolute_error}
    m = {n:f for n,f in m_valid.items() if n in metrics}
    if len(m) == 0:
        raise ValueError('Metrics must me one of {}'.format(m_valid.keys()))
    m_res = {n:[] for n,f in m.items()}

    # save input data
    x = model.exog
    y = model.endog
    #hacky solution see below
    if isinstance(model, sm.MixedLM):
        g = model.groups
        

    model_type = type(model); print('Model:', model_type)
    init_args = inspect.getfullargspec(type(model)).args
    kwargs = {k:v for k,v in vars(model).items() if k in init_args and k not in ['exog', 'endog']}

    kfold = KFold(shuffle=True, n_splits=n_splits)
    for i, (test_idx, train_idx) in enumerate(kfold.split(x)):
        ### groups= init arg... needs to be subset tooooooooooo (for mixedlm    )
        # hacky solution
        if isinstance(model, sm.MixedLM):
            kwargs['groups'] = g[train_idx]   # g is a numpy. What size?
        model = model_type(y[train_idx], x[train_idx,:], **kwargs)

        if fit_args is None:
            res = model.fit()
        else:
            res = model.fit(**fit_args)
        yhat = res.predict(x[test_idx,:])
        for n in m:
            m_res[n].append(m[n](y[test_idx], np.round(yhat)))
       #acc = accuracy_score(y[test_idx], np.round(yhat))
        #accs.append(acc)
        #roc = roc_auc_score(y[test_idx], np.round(yhat))
        #rocs.append(roc)
        print('Fold:', i)

    return m_res


### Improvement : pass exog and endog and any other init args to the function 
# itself and maybe pass model type as an arg tooo. Or even better, dict of any 
# params you want to pass to be expanded in init, its almost universal enought to 
# expand beyond statsmodels


### Problem: failed appraoch, pass all vars in vars(model) (expecpt endog and exog) 
# to new instance
# Flaw: some vars in vars() were created after/outside of __init__ and where 
# thus could not be passed to __init__
## approaches:

# Status: semi-failed
# 1) replace only exog and endog in each loop
# Drawback: there might be instance vars held over after fit, that may not be reset upun the next fit call
# Drawback I don't know what happend to a arg after its been passed. in case of groups= in sm.MixedLM replacing the attr didn't work, it needed to be passed to init

# 2) use inspect.getfullargspec to reconstruct which vars() args can be passed to __init__ 
# Is it slower?
# limitation: it will not tell you (directly) the names of args passed as **kwargs to the __init__

# 3) try loop for passing nested sets of vars() to __init__
# slower?


#### Problem: pass fit methods  not in __init__
# Approaches

#1) pass fit args explicity  (current approach)

#2) pass a fitted model initially? 
#  Drawback: (would that make recovering args more) 
# problem atic becuase the values in vars() may have been alterned by .fit() a
# nd be unsuitble for passing  to __init__

### Issue: bc kwargs have already passed through what every __init__ did to 
# it, so if we pass them to init again it may reproces them, however if we just 
# update teh attrs  some necessary secondary attrs may not be updated as a result 
# of whatever init processing!!

#possible solution: update the intance attrs with kwargs rather than pass then, 
# problems with this dicussed above (use vars(model).update(kwargs))


class KFoldStatsmodels:
    '''
    Issue: Why again did this need to be a class to re pass arbitrary many 
    model init args and fit args??? Because to do so I would have to pass the 
    model and the init args separately but to do so they would have to be in a 
    dict to separate teh arbitrary fit and init args. That might not be too 
    annoying if you use dict(k= v)
    '''
    
    def __init__(self, metrics, n_splits=5, fit_args=None, predict=None):
        '''
        predict: Custom predict function that takes results obj and newdata df
        '''
        # validate metrics
        metrics = metrics if isinstance(metrics, list) else [metrics]
        m_valid = {'acc': accuracy_score, 'roc':roc_auc_score, 'r2':r2_score, 'mae': mean_absolute_error}
        metrics = {n:f for n,f in m_valid.items() if n in metrics}
        if len(metrics) == 0:
            raise ValueError('Metrics must me one of {}'.format(m_valid.keys()))


        self.n_splits = n_splits
        self.metrics = metrics
        self.predict = predict

    def model(self, model, *data_args, from_formula=None, data=None, **kwargs):
        '''
        from_formula: (formula str) if passed do not pass endog, exog but do pass data=.

        *data_args: endog and exog, if from_formula=None

        **kwargs: other args to be passed to model init

        model: (statsmodels model type)

        Purpose: save model __init__ args

        Issue: !!!! null ypreds and it changes each time.....
        Issue: # !!!!!!!!!!important ... full model.endog shape is not same as input data.. rows.. were dropped
        Issue: passing numpy arrays to api init method may cause issue with selector syntax consistency btw np and pd
        Issue: consider a check that model is subclass of statsmodel base estimator class
        Issue: move model arg to __init__?, honestly all of these args could be in init...
        Issue: consider (almost certainly should) two separete functions one for
        from_formula, one from not.., model form_formual to __init__ and create
        two internal build_model funcs one for each inputs/api, and decide which 
        one to use from the from_formula switch
        * Better: use fitting run switch, and a from_formula/api funcs, with true 
        collect (formula, data, kwargs - for from_formula) (endog, exog, 
        kwargs - for api), run the rest of the function identically
        Issue: create check for categorical vs contiuous acc metrics, cirrently 
        only continuoius
        ''' 
        #record init vars for model (really buildmodel())
        #orig_init_args = vars()
  
        # don't pass on models self (internal namespace) and don't pass along 
        # unused args i.e. tuples and dicts of zero len
        env = vars() # so that orig_init_args is not itself in env
        orig_init_args = {}
        for k,v in env.items(): 
            nonempty = True
            if isinstance(v, (tuple, dict)):
                nonempty = len(v)>0
            if k != 'self' and nonempty:
                orig_init_args[k] = v
        #print(orig_init_args)
        
        
        # make all numpy arrays pandas (for output interpretability) # distinguish series
        #orig_init_args.update({k: pd.DataFrame(v).squeeze() for k,v in orig_init_args if isinstance(v, np.ndarray)})  
        

        # try building the model, statsmodels will throw errors if any particular required args are missing.
        self.full_model = self.build_model(**orig_init_args)
        
        self.full_x_data = orig_init_args['data'] if 'data' in orig_init_args else orig_init_args['exog']
        #print(type(self.full_x_data))
        # hard coded, solve this problem of 'np.log(cost)' using patsy expansion
        #self.full_y_data = orig_init_args['data'][self.full_model.endog_names] if 'data' in orig_init_args else orig_init_args['endog']
        self.full_y_data = np.log(orig_init_args['data']['cost'])
        # !!!!!!!!!!important ... full model.endog shape is not same as input data.. rows.. were dropped


        self.orig_init_args = orig_init_args
        self.model = model
        self.from_formula = from_formula


    def build_model(self, model, *data_args, from_formula=None, data=None, **kwargs):
        '''
        Purpose: initiate model

        for params see model() doc string

        Issue: just update groups attr after if/else for formula? or create args dict for either and add to the dict in a single if c
        '''

        try:  ## hacky , this unpacks kwargs into {'groups':'domainlower'}) to be unpacked again
            kwargs = kwargs['kwargs']
        except KeyError:
            pass

        if from_formula is not None:
            # if no kwargs unpacking an empty dict... cool??
            print('kwargs in build model:', kwargs)
            model = model.from_formula(from_formula, data=data, **kwargs) 

        else: # normal api
            model = model(endog=endog, exog=exog, **kwargs)

        return model
        

    def fit(self, **fit_args):
        '''
        alternate: pass these args as fit_args to init_model, and fit and init in one go
        '''

        #model_type = type(model); print('Model:', model_type)
        #init_args = inspect.getfullargspec(type(model)).args
        #kwargs = {k:v for k,v in vars(model).items() if k in init_args and k not in ['exog', 'endog']}
        orig_init_args = self.orig_init_args
        from_formula = self.from_formula
        metrics = self.metrics

        fold_init_args = orig_init_args.copy() # shallow copy
        
        params = []
        pvalues=[]
        exact = defaultdict(list)
        kfold = KFold(shuffle=True, n_splits=self.n_splits)
        for i, (train_idx, test_idx) in enumerate(kfold.split(self.full_x_data)):
            #print('test shape:', test_idx.shape)
            #print('train shape:', train_idx.shape)

            if from_formula is not None:
                fold_init_args['data'] = orig_init_args['data'].iloc[train_idx,:]

            else: # normal api:
                fold_init_args['endog'] = orig_init_args['endog'].iloc[train_idx]
                fold_init_args['exog'] = orig_init_args['exog'].iloc[train_idx,:]

            ### groups= init arg... needs to be subset tooooooooooo (for mixedlm    )
            # hacky solution
            if isinstance(self.model, sm.MixedLM):
                fold_init_args['groups'] = orig_init_args['groups'].iloc[train_idx] 
        
            model = self.build_model(**fold_init_args) 
            res = model.fit(**fit_args) 
            
            #print('End FIT')
            if self.predict:
                yhat = self.predict(res, self.full_x_data.iloc[test_idx,:])
            else:
                yhat = res.predict(self.full_x_data.iloc[test_idx,:])
            
            #print(self.full_y_data.shape, self.full_x_data.shape, self.full_model.exog.shape)
            #print('null preds...:' , yhat.isnull().sum())

            # record metrics
            
            #print(metrics)
            for score in metrics:
                ytrue = self.full_y_data.iloc[test_idx]
                ypred = yhat
                exact[score].append(metrics[score](ytrue , ypred)) # we 
            mean = {k: np.mean(v) for k,v in exact.items()}
            std = {k: np.std(v) for k,v in exact.items()}
            params.append(res.params)
            pvalues.append(res.pvalues)


            
            print('Fold:', i)
        
        params = pd.DataFrame(params)
        pvalues = pd.DataFrame(pvalues)
        out = {'exact': exact, 'mean': mean, 'std': std, 'params': params, 'pvalues':pvalues}
        return out

        
