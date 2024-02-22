

from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix as cm_sklearn
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import KFold

import inspect
import json
import os
import re
import warnings
import pickle
import copy
from collections import defaultdict



############################################
##########  Image processing

def plot_image(pixels, ax=None):
    """
    From https://raw.githubusercontent.com/hellodanylo/ucla-deeplearning/master/02_cnn/utils.py
    Simply plots an image from its pixels.
    Pixel values must be either integers in [0, 255], or floats in [0, 1].
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(pixels)
    ax.axis('off')
    return ax
    #plt.show()