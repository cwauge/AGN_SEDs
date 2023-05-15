import numpy as np
import matplotlib.pyplot as plt
from bootstrap_err import BootStrap


# x = np.random.normal(1, 2, 10)
x = np.linspace(0,15,10)
y = (0.5*x + 10)+np.random.normal(0, 0.25, 10)
x_outliers = np.random.normal(1, 2, 5)
y_outliers = (0.25*x_outliers + 10)+np.random.normal(0, 5, 5)

# x = np.append(x, x_outliers)
# y = np.append(y, y_outliers)

x_err = abs(np.random.normal(0, 0.25, 10))
y_err = abs(np.random.normal(0, 0.25, 10))
y_outlier_err = abs(np.random.normal(0, 3, 5))
# y_err = np.append(y_err, y_outlier_err)

x_line = np.arange(-4, 6)
y_line = 0.5*x_line + 10


boot = BootStrap(x, y, x_err, y_err, 1000)
boot.plot_fits()
boot.plot_interp(np.linspace(0,15,1000),7.8,True)
