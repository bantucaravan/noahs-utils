
from IPython.display import display
import pandas as pd
from pandas import json_normalize
import numpy as np

import json
import os
import sys
import re
import numbers
import itertools
from collections import defaultdict
import datetime as dt

from noahs.general_utils import save_pickle, load_pickle





##########################################
############### experiment logging Management utils 

#NB: alternative for all of this is the existing mlflow logging module






# write json - write json (not append) to disk
def save_json(dct, path, **kwargs):
    '''
    write json to disk

    path -- file path

    **kwargs -- passed to json.dump()

    issues:
    validate .json file extension?
    '''
    with open(path, 'wt') as f:
        json.dump(dct, f, **kwargs)





# read json - read json from disk to memory
def load_json(path, **kwargs):
    '''
    read json from disk

    path -- file path

    issue: validate .json file extension?
    '''
    with open(path, 'rt') as f:
        dct = json.load(f, **kwargs)
    return dct



def integer_keys(dct):
    '''
    For use as value for object_hook= parameter in json.load(), to convert 
    (i.e. decode) any json dict keys that represent digits but were 
    serialized as strings (as per the json standard) back into int type


    # Issue: convert all single numerics (floats and negatives too) back, 
    # numerics in lists and in values position are already converted
    '''

    if any(k.isdigit() for k in dct):
        return {int(k) if k.isdigit() else k:v for k,v in dct.items()}
    return dct



class NumpyJSONEncoder(json.JSONEncoder):
    '''
    value for parameter cls= in json.dump, to convert non-json serializable 
    obj (particularly numpy objs) types to json serializable types.

    See for explanation: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
   
    Issue: overly braod - if not np array or np numeric, convert to string.. be more specific
    
    
    '''
    def default(self, obj):        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return np.array(obj).tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else: # is this too broad...?            
            return str(obj)
        return json.JSONEncoder.default(self, obj)
        # why not!!!! no good reason I think
        #return super().default(self, obj)





#BRIAN: 
# write log json: handle if file doesn’t exist yet, and is empty; 
#*with key validation handle if key is string ,try catch load_jsonlog conversion from string .. or use some native json module default()
# * handele more complex objs…(else return str(obJ)? 
# * handle jsone encode error partial writing to disk
# * also loook up existing json logging options



def safe_update_log(update_func):
    '''
    decorator (i.e. factory func) to safe load and safe save updates to the log
    
    any func decorated by safe_update_log() should assume the presence 
    `master_log` (in self(future) or passed as arg (current)), make 
    modifications to `master_log` and return modified `master_log`
    '''

    def inner_func(self, *args, **kwargs):
        '''
        deal with redundent loading and saving of the log

        The log json is in dict of dicts format with top-lvel keys all ints (run_ids)

        Issue: make a cached copy of the log im meory to avoid having to 
        read from memory each time we write some thing # when only updating 
        one run_id, save cache in memory to not have to read if same is edited twice in a row.

        Issue: validate dict of dicts format with top-lvel keys all ints, already done?

        Issue: allow regular cloud back up

        Issue: only overwrite the relevant run id... hard to do without 
        reading the whole file and even if possible hard to insert text in 
        a text file. must write EOL separated jsons with tab separated 
        run_ids? so that I can selectively read in a line? or first line is comma 
        separated run_ids that index the following new lines... OR EVEN BETTER 
        first line is tab separated sets of comma separated run_id, json char 
        (or is it byte... yes byte) length, so that to use seek to read selectively 
        form byte stream, you get the seek start position of the  run_id by summing 
        the json byte lengths (plus EOL bytes) for all the previous run_ids, and 
        similarly for end seek position 
        SEE: https://stackoverflow.com/questions/2081836/reading-specific-lines-only 
        AND https://stackoverflow.com/questions/7167008/efficiently-finding-the-last-line-in-a-text-file 
        AND https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file 
        AND https://stackoverflow.com/questions/620367/how-to-jump-to-a-particular-line-in-a-huge-text-file 
        with https://docs.python.org/3/library/itertools.html#itertools.dropwhile 
        AND see how skiprows=/nrows= in pd.read_csv is implemented for ideas (it 
        seems maybe implemented in C), or just comma separate and use pd.read_csv 
        as a wrapper

        for inserting... seek() on r+b (or t) and you will overwrite bytes from that position, how to rearange bytes on disk
        maybe asyncrhonous moving bytes from end to beginng of fiel and truncating at end 
        https://stackoverflow.com/questions/4388201/how-to-seek-and-append-to-a-binary-file-in-python

        The actualy slowness is in json_normalize not in read/write to disk of file

        '''

        ## Load existing logs and update active log
        # if log file is not blank

        master_log = self.safe_load_log()
        old_log = master_log.copy()
        
        # instead of passing explicity... why not assign master_log to self 
        # and pass it that way... that way update_func only recieves explicitly 
        # its purpose specific args 
        new_master_log = update_func(self, master_log=master_log, *args, **kwargs)
    
        ## save updated log
        self.safe_save_log(new_master_log, old_log)


    return inner_func



#class test: pass 
#self = test()
#self.__dict__ = locals()
logfile = '../test.json'
class JSONExperimentLogger:
    '''

    run_id - 
    run id must be set by calling (and only possible by calling!? or 
    specifying in .update_log() ) .set_run_id() before updating the log 
    with .update_log()


    Issue: # update the log from tabular format (i.e array of value at indexed 
    by speciifc paths of keys update)... maybe easier with mongodb...)


    Issue: de
    '''


    def __init__(self, logfile=None, cls=NumpyJSONEncoder):
        '''
        Params:
            cls: passed to cls in json.dump()
        
        Issue: ADD version history -- use git, gitignore all other files 
        but log files, and just chechout the single file not that whole 
        commit when retireving old versions 
        ALT: faster lighter implementation, just save and overwrite a single 
        back up file for each action (put it in save load?) and add an undo 
        method, so that every action can be at least rolled back once
        '''
        if logfile is None:
            raise ValueError('logfile must be specified')
        elif not logfile.endswith('.json'):
            raise ValueError('logfile must have json file extension')
        self.logfile = logfile
        # write empty json
        if (not os.path.exists(self.logfile)) or (os.path.getsize(self.logfile) == 0):
            save_json({}, self.logfile)

        
        self.cls = cls

    def _setup_git_repo(self):
        '''
        set up repo in the log file folder, check if .git exists and logfile 
        is tracked. make sure gitignore is there and ignoring all other files

        then add _git_commit to the safe update wrapper
        '''
        pass


    def _generate_run_id(self):
        '''
        integrate this code directly into .set_run_id()? Is it used 
        anywhere else?

        Currently, will go back to 0 if you deleted the first runs in a log... do you want this... I think you'd want to continue after the highest number in the log...
        '''
        used_ids = list(self.load_log().keys())
        # if used_nums is empty, do I have to set `used_ids = [-1]` to get 
        # a num starting at 0 below???

        # empty list cast as boolean is False
        run_id = max(used_ids)+1 if used_ids else 0
        
        # old code, that will put the run_id the lowest possible unused number
        # I decided that was not desired behavior
        if False:
            # smallest possible value will be 0
            run_id = np.min([0, min(used_ids)]).astype(int) 
            # to avoid overwriting random nums that were written before this 
            # sequential run num was introduced.
            while run_id in used_ids:
                run_id = run_id + 1

        # somehow the astype(int) above is not working
        return int(run_id)
    
    
    def set_run_id(self, run_id=None):
        if run_id is None:
            run_id = self._generate_run_id()
        assert isinstance(run_id, int)
        self.run_id = run_id

        #start run timestamp
        now = dt.datetime.now().replace(microsecond=0).isoformat(' ')
        self.update_log({'timestamp': now})



    def end_run(self):
        '''
        set run_id to None, so that further calls of .update_log() do not
        alter existing run_ids
        '''
        self.run_id = None
        

    def load_log(self, run_id=None ,object_hook=integer_keys, out='json', **kwargs):
        '''
        Description: read entire log json into memory, optionally return only specific single 
        (or multiple) run logs

        Params:
            out: (str) 'json' or 'df'

            **kwargs: passed to pd.json_normalize



        Issues:
        * valudate .json file ext?
        * currently only single not multiple run num specification supported

        Issues return='df':
        * use run_nums as index?
        * currently only supports selecting single not multiple run_nums

        '''

        # All JSON keys must be JSON string objects. integer_keys() as the 
        # object_hook converts all JSON key strings that represent integers 
        # into python int objects rather than str objects 
        # as would be default json.load() result
             
        if out=='df': 
            # leave integer keys as strings for compatibility with expected 
            # format of pd.json_normalize()
            object_hook=None
        
        outlog = load_json(self.logfile, object_hook=object_hook)
        if isinstance(run_id, int):
            outlog = {run_id: outlog[run_id]} # for compatibilty with read_log_df expectations
            #return outlog[run_num]

        if out == 'df':
            df = json_normalize(list(outlog.values()), **kwargs)
            df.index = outlog.keys()
            df = df.dropna(axis=1, how='all')
            return df

        return outlog


    def safe_load_log(self):
            try:
                #object_hook=integer_keys insures integer keys are converted into python ints
                master_log = load_json(path=self.logfile, object_hook=integer_keys) 
            except json.JSONDecodeError as e:
                msg = 'JSON file misformatted, failed to read.'
                raise json.JSONDecodeError(msg, e.doc, e.pos)
            return master_log


    def safe_save_log(self, master_log, old_log):
        try:
            save_json(master_log, self.logfile, cls=self.cls)
        except TypeError as e:
            #  revert to old json string just in case by aborting 
            # write op when error was raised json wrote a partial, unreadable json string
            save_json(old_log, self.logfile, cls=self.cls)
            # why not raise e directly... oh bc `e` is an instantiated Error 
            # obj while we are trying to instantiate a new error....
            raise type(e)('Failed to write JSON because non-json serializable type was passed.')

   
    @safe_update_log
    def update_log(self, json_log, run_id=None, master_log=None): 
        '''
        Description: take a json log for a single (or multiple) runs, and .update() 
        master json log (on disk?) so that entries for that run number are 
        overwritten by the updated json data
        
        Issue: allows possible overwrite of run log with incorrect information! 
        (must save back ups of log json file regularly!)  
        
        Params: 
            json_log: must be a dict with single (or multiple) integer keys

            kwargs: passed to json.dump(), Ex????

        Issue: currently only single not multiple run num specification supported

        ISSUE: update from df?? so that you can alter as df and then save alteration
            
        '''
        # delete, error will raise later in code
        assert isinstance(master_log, dict) # empty dicts return false   
        
        run_id = run_id if run_id else self.run_id 
        assert isinstance(run_id, numbers.Integral), f'run_id is {run_id}.\
        run_id must be an int.'

        # this could be added to the object hook in load_json()
        # incase run_id does not exist yet in master_log
        master_log = defaultdict(dict, master_log)
    
        if run_id is not None: # allow 0
            # assumes json_log does not include run_id
            master_log[run_id].update(json_log)
        else:
            raise Exception('must speficy run')
            # I could implement accepting a fully formatted run log
            #master_log.update(json_log)  
        
        return master_log


    @safe_update_log
    def rename_key(self, old_key, new_key, master_log=None):
        '''
        Rename all keys named old_key in all run logs

        ISSUE: raise warning/error if old_key not found

        ISSUE: reformat as while loop to search for key in artitrarily deeply nested dicts.

        '''
        assert isinstance(master_log, dict) # empty dicts return false

        # rename keys
        # can I somehow reference in place with out reconstructing? del and assign?
        new_log = {id: {k if k != old_key else new_key:v for k,v in dct.items()} for id, dct in master_log.items()}

        return new_log

        
    @safe_update_log
    def delete_key(self, key, master_log=None):
        '''
        delete every dict entry  key == `key`

        ISSUE: add a warning if key is not found, add a confirmation if it is found

        ISSUE: specificy run id?
        '''
        assert isinstance(master_log, dict) # empty dicts return false

        new_log = {}
        for id, dct in master_log.items():
            new_log[id] = {k:v for k,v in dct.items() if k != key} 
        
        return new_log

    @safe_update_log
    def delete_runs(self, run_id, master_log=None):
        '''
        Description: delete entire run log....

        Params
            run_id: (int or list-like of int)

        #ISSUE: return error if run not found, and confirmation if succesfully deleted
        '''
        # get rid of this.. unecesary, if problem error will get raised in code
        assert isinstance(master_log, dict) # empty dicts return false

        if isinstance(run_id, numbers.Integral):
            run_id = [run_id]
        
        for k in run_id:
            del master_log[k]
        new_log =  master_log # should it be copy?
        
        return new_log

    @safe_update_log
    def add_to_all(self, key, value):
        '''
        add key value pair to all run logs

        Issue: ensure that key does not already exist anywhere
        '''
        pass


# delete logs... or saved artifacts from non high performing models...






def try_args(arg_dict, method):
    '''
    For passing a dict of a super set of acceptable args to a function, 
    removing unacceptable args until a maximal subset of acceptable args 
    is reached.

    Issue: wouldn't a better approach be to get the function signature and 
    match param names? Or better why are we trying to get/set params on 
    functions that do not have built in get/set param functionality
    
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






# location of nbrun_git_clone
sys.path.append("/Users/noah.chasek-macfoy@ibm.com/Desktop/projects/Drone proj/code/")
from nbrun_git_clone.nbrun import run_notebook
# original repo https://github.com/tritemio/nbrun

def run_nb(kw_variants, base_nb, save_html=False, save_ipynb=False,
outname_func=None, savedir=os.getcwd(), combine='prod', timeout=60*60*6):
    '''
    Description: run variants of keyward on a notebook
    
    Params:
        kw_variants: (dict of list) keys should match the nb keys to be set
        in the nb, and values should be a list of all values for the given key
        to cycle through

        outname: (callable) should take nb_kwargs dict (each iteration) and 
        return a string.

        combine:(str) if combine=='prod' the outer product (every combination, like nested for-loops) of all values in every kw list will be run in the notebook. If combine=='zip', kw list must all be of equal length (or length 1 to be broadcast) the lists will be zipped and the combinations at each index position will be passed to the notebook. 

        timeout: (int) seconds to wait for nb to run before throwing
        timeout error

    Issue: Add a try/catch??? so you don't stop if one config fails and 
    you can go back to it..

    '''
    print(savedir)
    #kw_dict
    
    if combine == 'prod':
        variants = itertools.product(*kw_variants.values())
    elif combine == 'zip':
        lens = np.unique([len(i) for i in kw_variants.values() if len(i) != 1])
        assert lens.shape[0] == 1, \
        'All lists of variants must be equal length (or length of 1).'
        # expand value lenght from 1 to given len, for len==1 value lists
        kw_variants = {k: v*lens[0] if len(v)==1 else v for k,v in kw_variants.items()}
        variants = zip(*kw_variants.values())

    for values in variants:
        nb_kwargs = dict(zip(kw_variants.keys(), values))

        outname=''
        if save_html or save_ipynb:
            outname = outname_func(nb_kwargs)

        run_notebook(base_nb, 
                    out_path_ipynb=os.path.join(savedir, outname+'.ipynb'),
                    out_path_html=os.path.join(savedir, outname+'.html'),
                    save_html=save_html,
                    save_ipynb=save_ipynb,
                    timeout=timeout,
                    nb_kwargs=nb_kwargs)
        print('Completed:',nb_kwargs)