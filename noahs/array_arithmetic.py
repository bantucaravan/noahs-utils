
import pandas as pd
import numpy as np

######### Sparse non-zero/mask operatcions ###############
'''
operations that treat a sparse matrix as a mask effectively and performs 
operations only on the nonzero (or really the explicity stored) elements

'''

# Pandas group by is probably faster and easier

def nzrmin(W):
    # row-wise min of non-zero elements #  0 if no non-zero elems per row.
    a = pd.Series(W.data).groupby(W.nonzero()[0]).min()
    if a.shape[0] != W.shape[0]:
        a = a.reindex(range(W.shape[0])).fillna(0)
    return a.values



# Contribute on SO
#W = W_train
def nzragg(W, method='sum'):
    # row-wise agg of non-zero elements 
    #a = pd.Series(W.data).groupby(W.nonzero()[0]).agg(method)
    # groupby sorts by grouper so that even unsorted indices will return 
    # sorted to align with actual W rows
    # unlike .nonzero() below indexes explicit zeros in addition to non-zeros
    W = W.tocsc()
    a = pd.Series(W.data).groupby(W.indices).agg(method)

    if a.shape[0] != W.shape[0]:
        # reindex --> align vec with rows of original array in 
        # case all zero/implicit rows were skipped
        #  fillna -> default to 0 if no non-zero elems per row.
        a = a.reindex(range(W.shape[0])).fillna(0)
    return a.values
    # give toy example with all zero row
    # give performant large array example


'''
TEsting and Exploration:

mat = W_train
vec = nzragg(mat, 'min')
vec.shape
type(vec)

# exploring row indexes of stored values vs strict non-zeros
mat.nonzero()[0].shape
mat.tocsc().indices.shape
mat.tocsc().data
mat.data
mat.tocsc().shape

'''

'''
# it seems nonzero ignores explict zeros 
true_scores.shape
true_scores.nonzero()[0].shape
true_scores.data = np.where(true_scores.data >= min_score, 1, 0)
true_scores.nonzero()[0].shape
true_scores.data.shape
(true_scores.data == 0 ).sum()

newdata = np.where(true_scores.data >= min_score, 1, 0)
a = sp.csr_matrix((newdata, true_scores.nonzero()), shape=true_scores.shape)
a.nonzero()[0].shape

true_scores.indices.shape
a.indices.shape

%timeit true_scores.tocsc().indices

%timeit true_scores.nonzero()[0]
'''

#Matrix-Vector arithmetic with braodcasting
# from https://stackoverflow.com/questions/20060753/efficiently-subtract-vector-from-matrix-scipy
def nzcolsubstract(mat, vec):
    '''
    This is for csr matrices. mat must be csr
    
    vec must be 1-d  or shape (1, n)
    
    defaut not inplace, mat is .copy()'ed could add optional inplace, so 
    passed obj is affected.

    vec is assumed dense 
    '''
    mat = mat.copy()
    if not isinstance(vec, np.ndarray):
        vec = vec.A
    assert (vec.shape[0]==1) & (vec.ndim==2)
    
    mat.data -= np.repeat(vec[0], np.diff(mat.indptr))
    return mat



def nzcolopprep(mat, vec):
    '''
    This is for csr matrices. mat must be csr
    
    vec must shape (1, n), although it is interpreted as a column.
    would 1-d be ok?
    
    defaut not inplace, mat is .copy()'ed could add optional inplace, so 
    passed obj is affected.

    vec is assumed dense 

    option to pass the  operation directly here:
    op = '/'
    eval(f'a.data {op}  b')

    Issue: add auto converion from other shapes to (1, n)

    Issue: consider passing the op, see above
    '''
    mat = mat.copy()
    if not isinstance(vec, np.ndarray):
        vec = vec.A
    assert (vec.shape[0]==1) & (vec.ndim==2)
    # speed consideration of -= vs - (inplace vs not)
    return mat,  np.repeat(vec[0], np.diff(mat.indptr))

def nzcolsubstract(mat, vec):
    a, b = nzcolopprep(mat,vec)
    # speed consideration of -= vs - (inplace vs not) #downcasting for converted dtype maybe int to float
    a.data -= b
    return a

def nzcoldivide(mat, vec):
    a, b = nzcolopprep(mat,vec)
    #  downcasting of converted float to int required for inplace op 
    # on a.data (if a is dtype int) throws error, thus using non-inplace
    a.data = a.data / b
    return a

'''
#Testing and Exploration
mat = scipy.sparse.csr_matrix([[1, 0, 3],
                    [2, 3, 0],
                    [0, 4, 5]])
vec = scipy.sparse.csr_matrix([1,2,3])

mat.A - vec.A.T
vec = vec.A
vec = vec.T
vec.shape
vec.ndim
mat.data.shape
mat.nonzero()[0].shape

mat.data -= np.repeat(vec.toarray()[0], np.diff(mat.indptr))
mat.data -= np.repeat(vec[0], np.diff(mat.indptr))
mat.A

%time a = nzcolsubstract(mat, vec)
a = nzcoldivide(mat, vec)
a.A

a, b =nzcolopprep(mat,vec)
a.data
'''


        #def minmax_scale_rows(W):
        #    top = nzcolsubstract(W, nzragg(W, 'min')[None,:] )  
        #    bottom = (W.max(axis=1).A - nzragg(W, 'min')[:,None])
        #    return nzcoldivide(top,bottom.T)
        # leaves zeros untouched, and they don't affect the scalling



if False:
    #%time a = nzrmin(W)





    #See for below: https://stackoverflow.com/questions/4373631/sum-array-by-number-in-numpy

    # scipy.ndimage approach
    from scipy import ndimage
    data = np.arange(10000000)
    groups = np.arange(1000).repeat(10000)
    #%time ndimage.sum(data, groups, accum)#range(1000))



    # ufunc .at method, pretty sweet!!!
    #%%time
    data = [1,2,3,4,5,6]*10000
    #ix = list(range(len(a)))*10000
    groups = [0, 1,2,3,4,5]*10000 # must be 0 index (concecutive?)... yeah

    accum = np.zeros(np.max(groups)+1)
    accum
    np.add.at(accum, groups, data)
    accum

    data = W.data
    groups = W.nonzero()[0]



    #%%time
    accum = np.repeat(np.max(data), np.max(groups)+1)
    np.minimum.at(accum, groups, data)
    if accum.shape[0] != W.shape[0]:
        # default has (max) value for full range up to max(groups), add rows if full rank is greater than max(groups) i.e. bottom rows are all zero
        # replace max in positions of accum where there was no idx in groups (maybe set to max+1 so it is a value that could not other wise exist in data)
        pass
    
    accum


    #all(a == accum) !!!

