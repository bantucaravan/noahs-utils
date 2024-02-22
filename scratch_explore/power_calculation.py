from statsmodels.stats.power import TTestPower, TTestIndPower
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


power_analysis = TTestPower() # TTestIndPower() # 


power_analysis.solve_power(effect_size=.8, nobs1=None, alpha=.05, power=.8)


# look at how power changes when effect size gets larger (holding alpha and smaple size constant)
power_analysis.plot_power(dep_var='effect_size',
                          nobs=[10, 20, 30, 40, 50],
                          effect_size=np.arange(0,1,.001),
)
plt.show()