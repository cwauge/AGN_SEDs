import numpy as np

def remove_outliers(x,y,bin_range):
    xmed = np.arange(bin_range[0],bin_range[1],0.5)
    x_out = x.copy()
    y_out = y.copy()
    x_out = []
    y_out = []

    for i in range(len(xmed)-1):
        select = (x > xmed[i]) & (x < xmed[i+1])
        x_s = x[select]
        y_s = y[select]

        mean = np.mean(y_s)
        std = np.std(y_s)

        bad = np.where(np.logical_or(y_s > mean+std, y_s < mean-std))
        good = np.where(np.logical_and(y_s > mean-std, y_s < mean+std))[0]
        print('mean: ',mean)
        print('std: ', std)

        for i in range(len(good)):
            x_out.append(x_s[good[i]])
            y_out.append(y_s[good[i]])

    return np.asarray(x_out), np.asarray(y_out)




