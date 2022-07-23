import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection


def main(ID, z, wavelength, Lum, L):
    plot = Plotter(ID, z, wavelength, Lum, L)

class Plotter():

    def __init__(self,ID,z,wavelength,Lum,L):
        self.ID = np.asarray(ID)
        self.z = np.asarray(z)
        self.wavelength = np.asarray(wavelength)
        self.Lum = np.asarray(Lum)
        self.L = np.asarray(L)

        plt.rcParams['font.size'] = 13
        plt.rcParams['axes.lionewidth'] = 2.5
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['xtick.major.width'] = 3
        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['ytick.major.width'] = 3

    def multilines(self,xs,ys,cs,ax=None,**kwargs):
        ax = plt.gca() if ax is None else ax # find axes
        segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
        lc = LineCollection(segments, **kwargs)
        lc.set_array(np.asarray(cs)) # set coloring of line segments
        ax.add_collection(lc) # ad lines to axes and rescale
        ax.autoscale()
        return lc

    def PlotSED(self,point_x=np.nan,point_y=np.nan,save=False):
        fig, ax = plt.subplots(figsize=(10,8))

        ax.plot(self.wavelength,self.Lum)
        ax.plot(self.wavelength,self.Lum,'x',c='k')
        ax.plot(point_x,point_y,'x',c='r')

        ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
        ax.set_ylabel(r'$\lambda$L$_\lambda$')
        ax.set_xscale('log')
        ax.set_yscal('log')
        ax.set_xlim(5E-5,7E2)
        ax.set_ylim(1E-4,1E2)
        ax.text(0.05,0.7,f'L = {self.L}',transform=ax.transAxes)
        plt.grid()
        if save:
            plt.savefig(f'/Users/connor_auge/Desktop/{self.ID}_SED.pdf')
        plt.show()





