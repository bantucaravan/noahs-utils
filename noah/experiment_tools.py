
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


    
##########################################
############### experiment logging Management utils
    

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


