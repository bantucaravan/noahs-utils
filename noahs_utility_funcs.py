from IPython.display import display
import pandas as pd


#Inspo: https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook

# fix 8 space default

def allcols(df, rows=False):
    if not rows:
        with pd.option_context('display.max_columns', df.shape[-1]):
            display(df)

    if rows:
        with pd.option_context('display.max_columns', df.shape[-1], 'display.max_rows', df.shape[0]):
            display(df)


