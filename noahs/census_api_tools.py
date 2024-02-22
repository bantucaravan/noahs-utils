import pandas as pd
import numpy as np
import requests


import re




# NB this table
#api_query = 'https://api.census.gov/data/2018/acs/acs5/profile?get=group(DP05)&for=state:*'
#df = get_table_with_names(api_query)


def get_table_with_names(api_query, multiindex=True):
    '''
    api_query: url of query

    Issue: split colmans up bc true col name is delimited by '!!'

    Issue: allow auto cutting of margin of error cols and percent cols, 
    like types='moe', 'est', 'perc', or 'all' (or some combo... in a list..)

    Issue: fix - implementation assumes groups only, no variables in 
    query, and assumes only single group
    '''

    # assumes groups only, no variables in query
    # assumes only single group
    group = re.findall(r'group\((\w+)\)', api_query)[0]

    base_url = api_query.split('?')[0]
    group_vars_url = base_url+'/groups/'+group+'.json'
    group_vars = requests.get(group_vars_url).json()
    key_name_map = {k:v['label'] for k,v in group_vars['variables'].items()}


    df = pd.read_json(api_query)
    # reset header
    header = df.iloc[0,:]
    df = df.iloc[1:,:]
    df.columns = header
    df.columns = [key_name_map[i] if (i in key_name_map) else i for i in df.columns]

    # parses column names into hierarchical multiindex
    if multiindex:    
        splits = df.columns.str.split('!!')
        maxlen = pd.Series(splits).apply(len).max()

        # pad each list with last list elem
        for levels in splits:
            diff = maxlen-len(levels)
            extra = [levels[-1]] * diff
            levels.extend(extra) 

        assert all(pd.Series(splits).apply(len) == maxlen)
        col_idx = np.array(splits.tolist()).T
        df.columns = col_idx.tolist()

    return df
