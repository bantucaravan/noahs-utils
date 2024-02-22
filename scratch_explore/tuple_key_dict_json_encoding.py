import json
import copy

a = {(1,2,3):{(1,2):'hello'}}

def default(o):
    if isinstance(o, tuple):
        return str(0)
    else:
        return json.JSONEncoder.default(o)


json.dumps(a, default=default) 
# error  
# # does not work bc encoding is top down hierarchical and the dict 
# containing the tuple keys is identified to be encoded first and passed 
# to json.JSONEncoder.encode() so the keys are not diverted to default()


def convert_tup_keys(d):
    '''
    recusive conversion of tuple keys into str.
    Can cause max recursion error if dict contains internal references
    see: https://stackoverflow.com/questions/10756427/loop-through-all-nested-dictionary-values/37026861#37026861
    NB!!!!!: Does NOT handle dicts inside other structures like list of dict
    No-op if dict keys are not tuples.
    '''
    dc = copy.deepcopy(d)
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_tup_keys(v)
        if isinstance(k, tuple):
            del dc[k]
            dc[str(k)] = v
    return dc

convert_tup_keys(a)

class CustomJsonEncoder(json.JSONEncoder):
    '''
    json encoder that replaces tuple-keys in dicts with their str reprs
    using the convert_tup_keys() recursive function.
    '''
    def encode(self, o):
        if isinstance(o, dict):
           o = convert_tup_keys(o)
        return super().encode(o)

json.dumps(a, cls=CustomJsonEncoder)
# works!
