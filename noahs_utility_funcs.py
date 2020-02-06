from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats
import seaborn as sns


import json
import os
import re
import warnings
import pickle
import copy




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

    df_cm = pd.crosstab(y_true, y_pred)
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


def interger_keys(dct):
# for use as object_hook= in json.load()

# Issue: convert all single numerics (floats and negatives too) back, 
# numerics in lists and in values position are already converted
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



def read_log_json(run_num=None, path='../logs/model logs (master file).json', object_hook=interger_keys):
    '''
    Description: read entire log json into memory, optionally return only specific single 
    (or multiple) run logs
    
    Issues:
    * valudate .json file ext?
    * currently only single not multiple run num specification supported

    '''

    outlog = read_json(path, object_hook=object_hook)
    # all json keys (or all json keys and values? NO) must be str. I am 
    # assuming that keys can be converted by int()
    #outlog = {int(k): v for k,v in outlog.items()}
    if run_num is not None:
        outlog = {run_num: outlog[run_num]}

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



def write_log_json(json_log, path='../logs/model logs (master file).json', **kwargs):
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
    

    if os.path.getsize(path) > 0: # if file in not empty 
        #object_hook=interger_keys insures integer keys are converted into python ints
        try:
            master_log = read_json(path=path, object_hook=interger_keys) 
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
        write_json(master_log, path, **kwargs)
    except TypeError as e:
            # Overwrite file just in case in raising error json wrote a partial, 
            # unreadable json string
        if not empty:
            write_json(old_log, path, **kwargs)
        else:
            open(path,'wt').close() # earses file contents
        raise type(e)('Failed to write JSON because non-json serializable type was passed.')




def read_log_df(run_num=None, path='../logs/model logs (master file).json'):
    '''

    issues:
    * use run_nums as index?
    * currently only supports selecting single not multiple run_nums

    '''
    dct = read_log_json(run_num)
    df = json_normalize(list(dct.values()))
    df = df.dropna(axis=1, how='all')
    return df


#####################
####### tf  model eval


def plot_tf_training_metric(history, metric, save=False):
    '''
    Issue: add pretty formatting and naming for plot labels
    '''

    plt.plot(history.epoch, history.history[metric], label='Train '+metric)
    plt.plot(history.epoch, history.history['val_'+metric], label='Test '+metric)
    plt.title('Train vs Test ' + metric)
    plt.legend()
    plt.show()
    
    if save:
        plt.savefig('../figs/' + str(num) + ' Train Test %s.png' %(metric))


def plot_tf_training(history, metric='accuracy'):
    plot_tf_training_metric(history, metric='loss')
    plot_tf_training_metric(history, metric=metric)


def top_epochs(history, metric='accuracy', top_n=-1):
    '''
    Issue: min vs max for different metrics
    '''
    res = dict(zip(history.epoch, history.history['val_'+metric]))
    #best_val_acc, best_val_acc_epoch = float(max(res.values())),  int(max(res, key=res.get))

    print('Best %s by epoch:' %(metric))
    out = sort_dict(res, reverse=True)#[:top_n]
    return {k:out[k] for k in list(out.keys())[:top_n]}