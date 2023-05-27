import numpy as np
import matplotlib.pyplot as plt


def plot_fit(x,y,deg,xlo,xhi,plot_color='k',lw=2.5):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    xrange = np.linspace(xlo,xhi,100)
    z, cov = np.polyfit(x,y,deg,cov=True)
    p = np.poly1d(z)

    y_out = p(xrange)
    x_out = xrange

    plt.plot(x_out,y_out,'--',color=plot_color,lw=lw)
    # print(x)
    # print(y)
    print('fit: ',z)
    print('error: ',np.sqrt(np.diag(cov)))
    return z
