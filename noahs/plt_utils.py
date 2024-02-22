import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/51898101/how-do-i-stagger-or-offset-x-axis-labels-in-matplotlib
def offset_xlabels(ax):
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    return ax








# FixedWidthGrid

# .iter_axes()
# '''iterate thru ax objs in fixed width grid'''