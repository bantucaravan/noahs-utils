from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if pd.__version__ >= '2':
    json_normalize = pd.json_normalize 
else:
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
import sys
from pathlib import Path
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

'''
ISSUE: consider checking what frame/scope layer the func is at and if not 
top level returning the option config string that can be taken by the outer 
func and applied ala allrows(fullcolwidth(df))
'''

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

def full(df, *opts):
    '''
    Params:
        *opts: (strs) specifying what kind of display options wanted. See func for options.
    '''

    args=[]
    if 'rows' in opts:
        args.extend(['display.max_rows', df.shape[0]])
    if 'cols' in opts:
        args.extend(['display.max_columns', df.shape[-1]])
    if 'width' in opts:
        args.extend(['display.max_colwidth', -1])
    if len(args) == 0:
        raise ValueError('display options must be one or more of ["rows", "cols", "width"]')

    with pd.option_context(*args):
        display(df)


def get_group(grouped_df, idx=None):
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

# https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal
def dict_compare(dct1, dct2):
    shared_items = {k: dct1[k] for k in dct1 if k in dct2 and dct1[k] == dct2[k]}
    print(len(shared_items) == len(dct1))
    return shared_items

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

    Issue: what about `tree` command in bash? doesn'ot do this for you?

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

if False:
    # JSON selection
    # come back......
    # Version: input: list of dicts - select dict key and return if any dict key matches (name: json_select(key=, value=)?)
    
    def json_select(json, key='name', value=None):
        # pd version
        pd.io.json.json_normalize(json)[key].tolist()
        # base version
        a = [i for i in a if i['name'] == collection_name]
        if len(a)==1:
            return a[0]


#Version: input list of dicts, return list where each element is the value from a given dict at selected dict key, add optional check to make sure all dicts have key
def json_select(key, json):
    '''
    json: (list of dict)
    '''
    # pd version
    pd.io.json.json_normalize(json)[key].tolist()
    # base version
    [i[key] for i in json]

    return

    json = r['collections']
    key='name'
    op = '=='
    val =  collection_name

    df = pd.io.json.json_normalize(json)
    df.loc[df['name'] == collection_name, 'collection_id']
    if len() == 1:
        pass
        #.item()



class JSONQuery(pd.DataFrame):
    #'''
    #issue: allow it to reduce to series like real df..
    #'''
    def __init__(self, json, select=None):
        # add list of dict validation
        if isinstance(json, list):
            json = pd.io.json.json_normalize(json)
            df = json.copy() # orig df
            super().__init__(df)
            self.df =df
        else:
            super().__init__(json)


        if select:
            self.select(select)


    def select(self,key):
        # return series as modified df (i.e. jsonquery obj)
        out =  type(self)(self[key])
        out.df = self.df
        if len(out) == 1:
            return out.values[0][0]
        return out
        
    def filter(self, key, op, val):
        # assumes self is a series
        mask = eval(f'self.df["{key}"] {op} "{val}"')
        out = type(self)(self.loc[mask,:])
        out.df = self.df
        if len(out) == 1:
            return out.values[0][0]
        return out


'''
#Tests

a =JSONQuery(json)
a.select('collection_id').filter('name', '==', collection_name)

'''

def join_check(right, left):
    '''
    Check the overlap and join cardinality between two keys that are to be used for a join. 

    Params:
        right: (pd.Series)
        left: (pd.Series) 

    TODO:
    * consider a looser criteria for cardinality where if there are duplicates on one side *even for
    rows that do not have a match on the other side* that is declared cardinality of "many". Currently
    there must be duplicates among rows on one side *that have a match on the other side* for a side 
    to get a cardinality of "many".
    * return value_counts() of total setdiffs left and right
    '''
    right_non_null, left_non_null = right.dropna(), left.dropna()
    
    right_matching_left_mask = right_non_null.isin(left_non_null)
    #right_setdiff = np.setdiff1d(right_non_null, left_non_null) # slow, at lease for strings
    # right_setdiff = right_non_null.loc[~right_matching_left_mask].unique() # remove .unique() to get total value count right not in left
    right_setdiff = set(right_non_null) - set(left_non_null) # Q: is this really the same as above and faster?
    # is_many_right = right_non_null.loc[right_matching_left_mask].shape[0] != right_non_null.loc[right_matching_left_mask].unique().shape[0]
    is_many_right = len(right_setdiff) != right_non_null.isin(right_setdiff).sum() # Q: is this really the same as above and faster?
    right_cardinality = 'Many' if is_many_right else 'One'
    
    print('Right row count:', right.shape[0]) 
    print('Right non-null count:', right.notnull().sum()) 
    print('Right unique non-null count:', right_non_null.unique().shape[0]) 
    print('Right non-null rows matched to left:', right.isin(left).sum()) 
    print('count of unique non-null right rows not in left:', len(right_setdiff))
    print('count of total non-null right rows not in left:', right.isin(right_setdiff).sum(),'\n')
    
    print('Intersection size:', np.intersect1d(right_non_null, left_non_null).shape[0],'\n')
    
    left_matching_right_mask = left_non_null.isin(right_non_null)
    #left_setdiff = np.setdiff1d(left_non_null, right_non_null) # slow, at lease for strings
    # left_setdiff = left_non_null.loc[~left_matching_right_mask].unique() # # remove .unique() to get total value count left not in right
    # is_many_left = left_non_null.loc[left_matching_right_mask].shape[0] != left_non_null.loc[left_matching_right_mask].unique().shape[0]
    # Q: is below really the same as above and faster?
    left_setdiff = set(left_non_null) - set(right_non_null) 
    is_many_left = len(left_setdiff) != left_non_null.isin(left_setdiff).sum()
    
    left_cardinality = 'Many' if is_many_left else 'One'
    
    print('Left row count:', left.shape[0])
    print('Left non-null rowcount:', left.notnull().sum())
    print('Left unique non-null count:', left_non_null.unique().shape[0])
    print('Left non-null rows matched to right:', left.isin(right).sum()) 
    print('count of unique non-null left rows not in right:', len(left_setdiff))
    print('count of total non-null left rows not in right:', left.isin(left_setdiff).sum())
    
    print('\nJoin Cardinality:', f'{right_cardinality} to {left_cardinality}')


def timefmtsecs(duration):
    '''
    convert duration in seconds (numeric type) to hr,min,sec str format.
    Useful for timing code with time.time()
    '''
    hrs = int(duration // (60*60))
    mins = int((duration % (60*60)) // 60)
    secs = round(duration % 60,1)
    _str = ''
    if hrs>0: _str += f'{hrs} hrs '
    if mins>0: _str += f'{mins} mins '
    return _str + f'{secs} secs'


def find_proj_root(sentinel='.git'):
    '''
    Find top level of project. Useful to set common relative 
    paths or entrypoint across project or on different machines 
    e.g. setting pythonpath for local libs.
    '''
    cwd = os.getcwd()
    # cwd = _dh[0]
    while not [i for i in Path(cwd).iterdir() if i.name == sentinel]:       
        cwd = Path(cwd).parent
    return cwd