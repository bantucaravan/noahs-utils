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
