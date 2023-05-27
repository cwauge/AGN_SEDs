import numpy as np


def mult_err(x, y, xerr, yerr):
    x = np.asarray(x)
    y = np.asarray(y)

    R = x*y

    Rerr = abs(R)*np.sqrt((xerr/x)**2+(yerr/y)**2)
    return Rerr

def div_err(x, y, xerr, yerr):
    x = np.asarray(x)
    y = np.asarray(y)

    R = x/y

    Rerr = abs(R)*np.sqrt((xerr/x)**2+(yerr/y)**2)
    return Rerr

def log_err(x, xerr):
    x = np.asarray(x)

    Rerr = (1./2.3)*(xerr/x)
    return Rerr

def anti_log_err(x, xerr):
    x = np.asarray(x)

    Rerr = 2.303*x*xerr
    return Rerr

