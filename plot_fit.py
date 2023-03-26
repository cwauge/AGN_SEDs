import numpy as np
import matplotlib.pyplot as plt


def plot_fit(x,y,deg,xlo,xhi,plot_color='k'):
    xrange = np.linspace(xlo,xhi,100)
    z = np.polyfit(x,y,deg)
    p = np.poly1d(z)

    y_out = p(xrange)
    x_out = xrange

    plt.plot(x_out,y_out,'--',color=plot_color,lw=2.5)
    print('fit: ',z)
    return z
