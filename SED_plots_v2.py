from ast import arg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from matplotlib.collections import LineCollection
from scipy import interpolate

from RunSEDv3 import B1



def main(ID, z, wavelength, Lum, L):
    plot = Plotter(ID, z, wavelength, Lum, L)

class Plotter():

    def __init__(self,ID,z,wavelength,Lum,L,norm,up_check):
        self.ID = np.asarray(ID)
        self.z = np.asarray(z)
        self.wavelength = np.asarray(wavelength)
        self.Lum = np.asarray(Lum)
        self.L = np.asarray(L)
        self.norm = np.asarray(norm)
        self.up_check = np.asarray(up_check)

        plt.rcParams['font.size'] = 25
        plt.rcParams['axes.linewidth'] = 3.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 4
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 4
        plt.rcParams['xtick.minor.size'] = 3.
        plt.rcParams['xtick.minor.width'] = 2.
        plt.rcParams['ytick.minor.size'] = 3.
        plt.rcParams['ytick.minor.width'] = 2.

    def multilines(self,xs,ys,cs,ax=None,**kwargs):
        ax = plt.gca() if ax is None else ax # find axes
        segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
        lc = LineCollection(segments, **kwargs)
        lc.set_array(np.asarray(cs)) # set coloring of line segments
        ax.add_collection(lc) # ad lines to axes and rescale
        ax.autoscale()
        return lc

    def median_sed(self,x_in,y_in,Norm=True,connect_point=False):
        '''Function to generate the median line for array of SEDs to be plotted'''
        x_out = np.nanmedian(x_in, axis=0)
        y_out = 10**np.nanmedian(y_in,axis=0)
        if Norm:
            y_out /= np.nanmedian(self.norm)

        plt.plot(x_out,y_out,c='k',lw=6)
        if connect_point:
            return x_out[-1], y_out[-1]

    def median_FIR_sed(self,xfir,yfir,Norm=True,connect=[np.nan,np.nan],upper='upper lims'):
        '''Function to plot the median FIR SED'''
        if upper == 'upper lims':
            yfir = yfir # if upper is True, only used detections to determine median FIR. Default is to use detections + upperlimts
        elif upper == 'data only':
            yfir = yfir[self.up_check == 0]
        else:
            print('Specify if FIR upper limits should be included in median calc or only data. Options are: upper lims,    data only')
            return
        x_out = np.nanmean(xfir,axis=0)
        y_out = np.nanmean(yfir,axis=0)
        if Norm: # If Norm is True, normalize FIR SED. Default is to normalize
            y_out /= np.nanmedian(self.norm)
        if ~np.isnan(connect[0]):
            # x_out = np.append(connect[0],x_out)
            # y_out = np.append(connect[1], y_out)
            x_out[0] = connect[0]
            y_out[0] = connect[1]
        plt.plot(x_out,y_out,c='k',lw=6,ls='--')

    
    def PlotSED(self,point_x=np.nan,point_y=np.nan,save=False):
        fig, ax = plt.subplots(figsize=(10,8))

        ax.plot(self.wavelength,self.Lum)
        ax.plot(self.wavelength,self.Lum,'x',c='k')
        ax.plot(point_x,point_y,'x',c='r')
        ax.plot(self.wfir,self.ffir,c='gray',lw=4)

        ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
        ax.set_ylabel(r'$\lambda$L$_\lambda$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(5E-5,7E2)
        # ax.set_ylim(1E-4,1E2)
        ax.set_title(self.ID)
        ax.text(0.05,0.7,f'L = {np.log10(self.norm)}',transform=ax.transAxes)
        plt.grid()
        if save:
            plt.savefig(f'/Users/connor_auge/Desktop/{self.ID}_SED.pdf')
        plt.show()

    def Plot_FIR_SED(self,wfir=[np.nan],ffir=[np.nan]):
        self.wfir = wfir
        self.ffir = ffir

    def multi_SED(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]],opt_p=[np.nan,np.nan],Median_line=True,FIR_med=True,FIR_upper='upper lims'):
        '''Function to overplot all normalized SEDs with each line mapping to a colorbar'''
        # Set colorbar limits
        clim1 = 43
        clim2 = 45.5
        cmap = 'rainbow_r' # set colormap

        x = self.wavelength[self.L >= clim1-0.1] # remove sources with L outside colorbar range
        y = self.Lum[self.L >= clim1-0.1] 
        L = self.L[self.L >= clim1-0.1]

        median_x = np.asarray(median_x)
        median_y = np.asarray(median_y)
        wfir = np.asarray(wfir)
        ffir = np.asarray(ffir)
        
        # Normalize the FIR luminosity
        if  len(self.norm) == len(ffir):
            ffir_norm = ffir.T/self.norm
            ffir_norm = ffir_norm.T
        else:
            ffir_norm = [[np.nan]] 
        ffir_norm = np.asarray(ffir_norm)

        # Begin plot
        fig, ax = plt.subplots(figsize=(20,15))
        ax.set_aspect(1)
        ax.set_xlabel(r'Rest Wavelength [$\mu$m]')
        ax.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim(1E-3,200)
        # ax.set_xlim(1E-10,1E5)
        # ax.set_xticks([1E-4,1E-3,1E-2,1E-1,1E0,1E1,1E2,1E3,1E4])
        ax.text(0.15, 0.85, f'n = {len(L)}', transform=ax.transAxes)
        
        # Plot the FIR upper limit segments 
        upper_seg = np.stack((wfir, ffir_norm), axis=2)
        upper_all = LineCollection(upper_seg,color='gray',alpha=0.3)
        ax.add_collection(upper_all)

        # use multilines function to plot all SEDs mapped to colorbar based on L
        lc = self.multilines(x, y, L, lw=2.5, cmap=cmap, alpha=0.75, rasterized=True) 
        axcb1 = fig.colorbar(lc, fraction=0.046, pad=0.04)  # make colorbar
        axcb1.mappable.set_clim(clim1,clim2) # initialize colorbar limits
        axcb1.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]')

        # plot data points for indvidual filters and optional point
        # ax.plot(x, y, 'x', color='k')
        if ~np.isnan(opt_p[0]):
            ax.plot(opt_p[0],opt_p[1]/self.norm,'x',color='r')


        # Plot median line
        if Median_line:
            ax.plot(np.nanmedian(x[:,:2],axis=0),np.nanmedian(y[:,:2],axis=0),c='k',lw=6)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x,median_y,connect_point=True)
                self.median_FIR_sed(wfir,ffir,connect=[x_connect,y_connect],upper=FIR_upper)
            else:
                self.median_sed(median_x,median_y)

        plt.ylim(5E-4,5E2)
        plt.xlim(7E-5,700)
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(f'/Users/connor_auge/Desktop/{savestring}.pdf')
        plt.show()

    # def multi_SED_bins(self, savestring, bin, field, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], opt_p=[np.nan, np.nan], Median_line=True, FIR_med=True, FIR_upper='upper lims'):
    #     '''Function to overplot all normalized SEDs with each line mapping to a colorbar and separated into three bins'''

    #     if bin == 'redshift':
    #         b1 = self.z <= 0.6
    #         b2 = (self.z > 0.6) & (self.z <= 0.9)
    #         b3 = (self.z > 0.9) & (self.z <= 1.2)
    #         t1 = '0 < z < 0.6'
    #         t2 = '0.6 < 0.9'
    #         t3 = '0.9 < 1.2'

    #     elif bin == 'field':
    #         b1 = field == 'g'
    #         b2 = field == 'c'
    #         b3 = field == 's'
    #         t1 = 'GOODS-N/S'
    #         t2 = 'COSMOS'
    #         t3 = 'Stripe82X'
    #     else:
    #         print('Specify bins. Options are: redshift,    field')
    #         return

    #     # remove sources with L outside colorbar range
    #     x = self.wavelength[self.L >= clim1-0.1]
    #     y = self.Lum[self.L >= clim1-0.1]
    #     L = self.L[self.L >= clim1-0.1]

    #     x1, x2, x3 = x[b1], x[b2], x[b3]
    #     y1, y2, y3 = y[b1], y[b2], y[b3] 
    #     L1, L2, L3 = L[b1], L[b2], L[b3]
    #     wfir1, wfir2, wfir3 = wfir[b1], wfir[b2], wfir[b3]
    #     ffir1, ffir2, ffir3 = ffir[b1], ffir[b2], ffir[b3]
    #     median_x1, median_x2, median_x3, = median_x[b1], median_x[b2], median_x[b3]
    #     median_y1, median_y2, median_y3, = median_y[b1], median_y[b2], median_y[b3],

    #     # Set colorbar limits
    #     clim1 = 43
    #     clim2 = 45.5
    #     cmap = 'rainbow_r'  # set colormap

    #     xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
    #     yticks = [0.001,0.01,0.1,1,10,100]
    #     xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

    #     # mosaic = '''123'''
    #     # fig = plt.figure(figsize=(12,8))
    #     # axd = fig.subplot_mosaic(mosaic)
    #     # lc = axd['1'].self.multilines
        
    #     # Set up Plot
    #     fig, ax = plt.figure(figsize=(18,6))
    #     gs = fig.add_gridspec(nrows=1,ncols=3,bottom=0.01,top=0.95,left=0.08,right=1.05,wspace=-0.15)

    #     ax1 = fig.add_subplot(gs[0])
    #     ax1.set_aspect(1)
    #     ax1.set_xscale('log')
    #     ax1.set_yscale('log')
    #     ax1.set_xlim(6E-5, 7E2)
    #     ax1.set_ylim(1E-4,120)
    #     ax1.set_xticks(xticks)
    #     ax1.set_yticks(yticks)
    #     ax1.set_xticklabels(xticks_labels)
    #     ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
    #     ax1.set_title(t1)
    #     ax1.grid()
    #     ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')

    #     ax2 = fig.add_subplot(gs[1])
    #     ax2.set_aspect(1)
    #     ax2.set_xscale('log')
    #     ax2.set_yscale('log')
    #     ax2.set_xlim(6E-5, 7E2)
    #     ax2.set_ylim(1E-4, 120)
    #     ax2.set_xticks(xticks)
    #     ax2.set_yticks(yticks)
    #     ax2.set_xticklabels(xticks_labels)
    #     ax2.text(0.05, 0.8, f'n = {len(x2)}', transform=ax2.transAxes)
    #     ax2.set_title(t2)
    #     ax2.grid()
    #     ax2.set_xlabel(r'Rest Wavelength [$\mu$m]')

    #     ax3 = fig.add_subplot(gs[2])
    #     ax3.set_aspect(1)
    #     ax3.set_xscale('log')
    #     ax3.set_yscale('log')
    #     ax3.set_xlim(6E-5, 7E2)
    #     ax3.set_ylim(1E-4, 120)
    #     ax3.set_xticks(xticks)
    #     ax3.set_yticks(yticks)
    #     ax3.set_xticklabels(xticks_labels)
    #     ax3.text(0.05, 0.8, f'n = {len(x3)}', transform=ax3.transAxes)
    #     ax3.grid()
    #     ax3.set_title(t3)

    #     # Plot data
    #     upper_seg1 = np.stack((wfir1,ffir1), axis=2)
    #     upper_all1 = LineCollection(upper_seg1, color='gray', alpha=0.3)
    #     ax1.add_collection(upper_all1)
    #     lc1 = self.multilines(x1,y1,L1,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
    #     axcb1 = fig.colorbar(lc1)
    #     axcb1.mappable.set_climi(clim1,clim2)
    #     axcb1.remove()
    #     # Plot median line
    #     if Median_line:
    #         ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0), c='k', lw=6)
    #         if FIR_med:
    #             x_connect, y_connect = self.median_sed(median_x1, median_y1, connect_point=True)
    #             self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper)
    #         else:
    #             self.median_sed(median_x1, median_y1)

    #     upper_seg2 = np.stack((wfir2,ffir2), axis=2)
    #     upper_all2 = LineCollection(upper_seg2, color='gray', alpha=0.3)
    #     ax2.add_collection(upper_all2)
    #     lc2 = self.multilines(x2,y2,L2,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
    #     axcb2 = fig.colorbar(lc2)
    #     axcb2.mappable.set_climi(clim1, clim2)
    #     axcb2.remove()
    #     # Plot median line
    #     if Median_line:
    #         ax.plot(np.nanmedian(x2[2:,:2],axis=0),np.nanmedian(y2[:,:2],axis=0),c='k',lw=6)
    #         if FIR_med:
    #             x_connect, y_connect = self.median_sed(median_x2,median_y2,connect_point=True)
    #             self.median_FIR_sed(wfir2,ffir2,connect=[x_connect,y_connect],upper=FIR_upper)
    #         else:
    #             self.median_sed(median_x2,median_y2)

    #     upper_seg3 = np.stack((wfir3,ffir3), axis=2)
    #     upper_all3 = LineCollection(upper_seg3, color='gray', alpha=0.3)
    #     ax3.add_collection(upper_all3)
    #     lc3 = self.multilines(x3,y3,L3,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
    #     axcb3 = fig.colorbar(lc3)
    #     axcb3.mappable.set_climi(clim1, clim2)
    #     axcb3.remove()
    #     # Plot median line
    #     if Median_line:
    #         ax.plot(np.nanmedian(x3[:,:2],axis=0),np.nanmedian(y3[:,:2],axis=0),c='k',lw=6)
    #         if FIR_med:
    #             x_connect, y_connect = self.median_sed(median_x3,median_y3,connect_point=True)
    #             self.median_FIR_sed(wfir3,ffir3,connect=[x_connect,y_connect],upper=FIR_upper)
    #         else:
    #             self.median_sed(median_x3,median_y3)

    #     plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
    #     plt.show()


    # def median_SED_plot(self, savestring, median_x, median_y, wfir, ffir, shape):
    #     '''Function to plot the median SED of SEDs separated by defined SED shape and bined into three z bins'''

    #     z1 = self.z <= 0.6
    #     z2 = (self.z > 0.6) & (self.z <= 0.9)
    #     z3 = (self.z > 0.9) & (self.z <= 1.2)
    
    # def Lum_Lum_1panel(self, savestring, X, Y, Norm_opt, Median, L, norm_val = 1.0, norm=[np.nan], x=[np.nan], y=[np.nan], up_check=[np.nan]):
    #     '''Function to make a single panel Luminosity - Luminosity Plot'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting class to generate a variety of different plots based on the output of the AGN class from SED_v8.py')
    parser.add_argument('ID', help='Source ID', type=str)
    parser.add_argument('--redshift','-z',help='best redshift measurement', type=float)
    parser.add_argument('--wavelength','-x',help='restframe wavelenght in microns')
    parser.add_argument('--Lum','-y',help='Normalized luminosity at each wavelength in erg/s (lambdaL_labmda)')
    parser.add_argument('--L','-l',help='additional Luminosity value, such as Lx or Lbol', type=float)
    parser.add_argument('--norm','-Lone',help='nomalization luminosity')

    args = parser.parse_args()
    main(args.ID,args.redshift,args.wavelength,args.Lum,args.L,args.norm)