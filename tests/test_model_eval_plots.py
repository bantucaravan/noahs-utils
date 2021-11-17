# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import os
import sys
import numpy as np

pckg_root = Path(__file__).parent.parent
sys.path.insert(0, pckg_root) # pckg root as first location to search for pckages
os.chdir(pckg_root) # set working dir (for file reads)
print(pckg_root)

from noahs.model_eval_plots import calibration_plot

# I should really hard code a specific input arrays because I don't want to 
# be testing numpy implicitly

# test is inspired by https://stackoverflow.com/questions/27948126/how-can-i-write-unit-tests-against-code-that-uses-matplotlib


def test_calibration_plot():

    np.random.seed(123)
    # fix these dists (positive erros for probs>.5 and negavtive for below)
    true = np.random.beta(2,2, size=500)
    errors = (np.random.beta(2,5, size=500) + .5) / 6 # re-center the errors, and scale down for smaller errors
    pred = (true + errors).clip(0, 1)

    fig = calibration_plot(true, pred, n_bins=4, return_fig=True)

    ax1, ax2 = fig.axes
    curve1, curve2 = ax1.lines
    rect1, rect2, rect3, rect4 = ax2.patches

    arr_test = curve1.get_xydata().round(8)
    arr_true = np.array([[0.28364284, 0.15247724],
        [0.47093964, 0.34016321],
        [0.67172111, 0.5388648 ],
        [0.90224502, 0.77301829]])
    np.testing.assert_array_equal(arr_test, arr_true)

    arr_test = curve2.get_xydata().round(8)
    arr_true = np.array([[0.        , 0.        ],
        [0.11111111, 0.11111111],
        [0.22222222, 0.22222222],
        [0.33333333, 0.33333333],
        [0.44444444, 0.44444444],
        [0.55555556, 0.55555556],
        [0.66666667, 0.66666667],
        [0.77777778, 0.77777778],
        [0.88888889, 0.88888889],
        [1.        , 1.        ]])
    np.testing.assert_array_equal(arr_test, arr_true)

    assert (67 == rect1.get_height()) & (0.2135372424997635 == rect1.get_width())

    assert (154 == rect2.get_height()) & (0.2135372424997635 == rect2.get_width())

    assert (136 == rect3.get_height()) & (0.2135372424997635 == rect3.get_width())

    assert (143 == rect4.get_height()) & (0.21363724249976349 == rect4.get_width())


