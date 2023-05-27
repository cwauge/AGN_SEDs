import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import interpolate


class BootStrap():

    def __init__(self,x,y,xerr,yerr,numb):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        self.numb = numb

        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 3.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 4
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 4
        plt.rcParams['xtick.minor.size'] = 3.
        plt.rcParams['xtick.minor.width'] = 2.
        plt.rcParams['ytick.minor.size'] = 3.
        plt.rcParams['ytick.minor.width'] = 2.
        plt.rcParams['hatch.linewidth'] = 2.5


    def random(self,data,data_err):
        self.rand_data = np.random.normal(data,data_err,len(data))

    def fit_boot(self,x_data,y_data,deg):
        x_fit = np.linspace(min(x_data[~np.isnan(x_data)]),max(x_data[~np.isnan(x_data)]))
        
        z = np.polyfit(x_data,y_data,deg)
        P = np.poly1d(z)

        y_fit = P(x_fit)

        return x_fit, y_fit

    def interp_boot(self,x_data,y_data,value,log=True):
        self.f_interp_boot = interpolate.interp1d(x_data,y_data,kind='linear',fill_value='extrapolate')
        if log:
            try:
                out_value = 10**self.f_interp_boot(np.log10(value))
            except:
                out_value = np.nan
        else:
            out_value = self.f_interp_boot(value)

        return out_value
        
    def iterate_fit(self,numb):
        x_out, y_out = [], []
        for _ in range(numb): 
            self.random(self.y,self.yerr)
            xfit, yfit = self.fit_boot(self.x,self.rand_data,1)
            x_out.append(xfit)
            y_out.append(yfit)

        return x_out, y_out

    def iterate_interp(self,value,numb,log,xrange=[np.nan],output='value'):
        interp_value = []
        interp_line = []
        interp_xrange = []
        for _ in range(numb):
            self.random(self.y,self.yerr)
            out_value = self.interp_boot(self.x,self.rand_data,value,log=log)
            interp_value.append(out_value)

            if output != 'value':
                yrange_interp = self.f_interp_boot(xrange)
                interp_line.append(yrange_interp)
                interp_xrange.append(xrange)

        if output == 'value':
            return np.asarray(interp_value)
        elif output == 'line':
            return np.asarray(interp_xrange), np.asarray(interp_line)
        elif output == 'both':
            return np.asarray(interp_xrange), np.asarray(interp_line), np.asarray(interp_value)

    def plot_fits(self):
        xfit, yfit = self.fit_boot(self.x,self.y,1)
        xboot, yboot = self.iterate_fit(self.numb)
        seg = np.stack((xboot,yboot),axis=2)
        collection = LineCollection(seg,color='orange',alpha=0.1)

        fig = plt.figure(figsize=(9,9),facecolor='white')
        ax = plt.subplot(111)
        ax.add_collection(collection)
        plt.plot(self.x,self.y,'.',color='k')
        plt.errorbar(self.x,self.y,yerr=self.yerr,c='gray',fmt='None')
        plt.plot(xfit,yfit,color='red',label='Fit to data')

        plt.show()

    def plot_interp(self,xrange,value,line=False):
        if line:
            interp_xline, interp_yline, interp_value = self.iterate_interp(value, self.numb, False, xrange,'both')
            seg = np.stack((interp_xline,interp_yline),axis=2)
            collection = LineCollection(seg,color='orange',alpha=0.1)
        else:
            interp_value = self.iterate_interp(value, self.numb, False, xrange,'value')

        value_x = np.ones(np.shape(interp_value))*value

        fig = plt.figure(figsize=(9,9),facecolor='white')
        ax = plt.subplot(111)
        if line:
            ax.add_collection(collection)
        plt.plot(value_x,interp_value,'x',color='r')
        plt.plot(self.x,self.y,'.',color='k')
        plt.errorbar(self.x, self.y, yerr=self.yerr, c='gray', fmt='None')
        plt.show()
