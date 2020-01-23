from IPython.display import display
import pandas as pd
import numpy as np

import pickle


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
    out = {k: v for k, v in sorted(dicti.items(), key=lambda item: item[1])}
    return out

def load_pickle(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
        return out
    
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)

def pretty_cm(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred)
    cm.columns.name = 'Predictions'
    cm.index.name = 'Truth'
    return cm
