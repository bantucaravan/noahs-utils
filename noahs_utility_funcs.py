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









