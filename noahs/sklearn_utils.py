from sklearn.compose import ColumnTransformer



#kwargs = {'poly': {'input_features': poly_cols}, 'ohe': {'input_features': ['state']}}
def get_feature_names(ct,fitted=True, **kwargs):
    '''
    A version of get feature names (exclusively for ColumnTransformer)
    that allows passing kwargs to each tranformers get_feature_names 
    method, particularly `input_features`. Closely based on the 
    sklearn code:
    https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/compose/_column_transformer.py#L343
    
    Params:
        kwargs: param kw should be underlying transformer name, and param value
        should be a dict of kwargs to be passed to the .get_feature_names() 
        method of specified transformer.

    Issue: handle getting names of passthrough columns, function 
    transformer (Identity function) with replace_strings==True, also 
    what is the string if replace_strings==False?
    '''
    assert isinstance(ct, ColumnTransformer)
    feat_names = []
    # c are the colum names it was printedt on
    for name, trans, c, _ in ct._iter(fitted=fitted, replace_strings=True):
        if name in kwargs:
            feat_names.extend(trans.get_feature_names(**kwargs[name]))
        elif hasattr(trans, 'get_feature_names'):
            feat_names.extend(trans.get_feature_names())
        else:
            print('Not implemented')
            

    return feat_names