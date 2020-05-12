import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pair-wise scatter plots for variables
# use to look for potential relationships btw vars
'''
sns.pairplot()
'''
#OR

#for 1 col against all others (with single legend)
def plot_y_vs_all(y_var, data, exclude=[], ncols=5, title=None, **sns_kwargs):
    '''
    y_var:(str) col label of y variable, must exist in data
    data: df to take y_var and other vars from
    exclude: list of col labels not to plot as x vars. NB: you can stratify 
    (i.e. use hue=) by variables excluded by exclude=
    ncols: number of columns in plot grid
    sns_kwargs: NB: you can stratify (i.e. use hue=) by variables excluded 
    by exclude=
    '''
    cols = data.columns.drop([y_var]+exclude)
    for i in range(0,len(cols), ncols):
        pg = sns.pairplot(data, y_vars=y_var, x_vars=cols[i:i+ncols], **sns_kwargs)
        if hasattr(pg, '_legend'): # could try/catch attributeerror
            pg._legend.remove()
        if i==0:
            if hasattr(pg, '_legend'):
                plt.gca().legend(handles=pg._legend.legendHandles, ncol=12, 
                bbox_to_anchor=(0.2,1.6))
            if title:
                pg.fig.suptitle(title, y=1.1, size='xx-large')
    
    plt.show()





'''
# (the beginnings of a) pretty correlation plot
# use to see how correlated your features are. highly correlated features 
# are usually bad, they make many feature importance/interpretation 
# models  useless and they provide little addition information while 
# providing the opportunity for overfitting to noise.
corrmat = data.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'recovered')['recovered'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''