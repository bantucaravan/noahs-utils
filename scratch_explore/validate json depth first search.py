import numpy as np
import json
import datetime as dt

################ Write my own depth first search

#Maybe find some smarter than me established algo inspiriation here: 
# https://www.hackerearth.com/practice/algorithms/graphs/depth-first-search/tutorial/

def validate_iter(child):
    '''
    check if obj is list, dict, or np.ndarray and  if so make iterable, else 
    raise typeerror
    '''
    if isinstance(child, list):
        parent = iter(child)
    elif isinstance(child, dict):
        parent = iter(child.values()) 
    elif isinstance(child, np.ndarray):
        parent = iter(child)
    else:
        raise TypeError
    return parent 

#root = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
root = dct
parent = validate_iter(root)
branch = []
while True:
    try:
        child = next(parent)
        branch.append(parent) # anything in branch is already validated as iterator
        try: 
            # I don't want to iter through strings!!! Should I have an else catch?
            parent = validate_iter(child)
            continue
        except TypeError:
            #validate json on leaf
            # I need to have the key vlaue pair
            print(child)
            continue
    except StopIteration:
        del branch[-1]
        if len(branch) > 0: # this could be a try/expect for index outof bounds error
            parent = branch[-1]
            continue
        else:
            break
    
## Issue: handle non lists: np.arrays, dicts, strings


#### logic

root = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
branch.append(root)
while len(branch) > 0:
    leaf = branch[-1]
#while leaf is iterable
    while leaf if iterable
        if next exists: 
            record parent child
            append parent to branch
            leaf = child
        else (if next is out of bounds):
    
    validate json on leaf
    leaf = branch[-1]

##################
# playground



try:
    json.dumps(np.int32(2))
except TypeError as e:
    print(e.args)

isinstance(np.float64(2), np.number)
isinstance(float(2), np.number)


########### best Method

# template source: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# base docs explanation: https://docs.python.org/3/library/json.html#json.JSONEncoder.default
#NB: creating a encoder class for cls= in json.dump() is NOT necessary, an equivelent func (like default in the encoder class) and passing it to default= in json.dump, see: https://docs.python.org/3/library/json.html#json.dump
# generic use of default= in json.dump() to record/seriealize  complex classes is explored in https://thepythonguru.com/reading-and-writing-json-in-python/#serializing-custom-object

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

json.dumps(np.int32(2), cls=NumpyEncoder)
json.dumps(dt.date.today(), cls=NumpyEncoder)

# downside of func approach is that json.JSONEncoder.default(self, obj) 
# does not throw the error itself, simply returning obj if conditions are 
# not met does not result a TypeError getting raised
def encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    raise TypeError(str(obj) + ' is not JSON serializable')

json.dumps(np.int32(2), default=encode_numpy)
json.dumps([2, np.int32(2)], default=encode_numpy)
json.dumps(dt.date.today(), default=encode_numpy)




