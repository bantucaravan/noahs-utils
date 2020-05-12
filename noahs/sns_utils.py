## include date axis utils here




# use to convert array of datetime values inputted as axis values
def days_since_series_start(series):
    a =(series.index - series.index.min()) / np.timedelta64(1,'D')
    return a.values
# alt option: matplotlib.dates.date2num(<datetime 1-d array>)

# use after plot func is directly called, to convert numbered axis to date repr
def sns_datelabels(data, ax):
    lbls = [data.index.min() + pd.DateOffset(days=i) for i in ax.get_xticks()]
    ax.set_xticklabels(lbls, rotation=-25, ha='left')
# alt: option a num2date or other options from:
#  https://matplotlib.org/3.1.1/api/dates_api.html



