from ast import arg
from stringprep import map_table_b2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import Lit_functions
from matplotlib.collections import LineCollection
from scipy import interpolate



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

    def solar_log(self,x):
        return x - np.log10(3.8E33)

    def ergs_log(self,x):
        return x + np.log10(3.8E33)

    def solar(self, x):
        return x/3.8E33

    def ergs(self, x):
        return x*3.8E33

    def multilines(self,xs,ys,cs,ax=None,**kwargs):
        ax = plt.gca() if ax is None else ax # find axes
        segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
        lc = LineCollection(segments, **kwargs)
        lc.set_array(np.asarray(cs)) # set coloring of line segments
        ax.add_collection(lc) # ad lines to axes and rescale
        ax.autoscale()
        return lc

    def median_sed(self,x_in,y_in,Norm=True,connect_point=False,Bin=False,bin_in=None,color='k',lw=6):
        '''Function to generate the median line for array of SEDs to be plotted'''
        x_out = np.nanmedian(x_in, axis=0)
        y_out = 10**np.nanmedian(y_in,axis=0)
        if Norm:
            if Bin:
                y_out /= np.nanmedian(self.norm[bin_in])
            else:
                y_out /= np.nanmedian(self.norm)

        plt.plot(x_out,y_out,lw=lw,color=color)
        if connect_point:
            return x_out[-1], y_out[-1]

    def median_FIR_sed(self,xfir,yfir,Norm=True,connect=[np.nan,np.nan],upper='upper lims',Bin=False,bin_in=None,color='k',lw=6,ls='-',line=True,ms=10):
        '''Function to plot the median FIR SED'''
        if upper == 'upper lims':
            yfir = yfir # if upper is True, only used detections to determine median FIR. Default is to use detections + upperlimts
        elif upper == 'data only':
            if Bin:
                yfir = yfir[self.up_check[bin_in] == 0]
            else:
                yfir = yfir[self.up_check == 0]
        else:
            print('Specify if FIR upper limits should be included in median calc or only data. Options are: upper lims,    data only')
            return
        x_out = np.nanmean(xfir,axis=0)
        y_out = np.nanmean(yfir,axis=0)
        if Norm: # If Norm is True, normalize FIR SED. Default is to normalize
            if Bin:
                y_out /= np.nanmedian(self.norm[bin_in])
            else:
                y_out /= np.nanmedian(self.norm)
        if ~np.isnan(connect[0]):
            # x_out = np.append(connect[0],x_out)
            # y_out = np.append(connect[1], y_out)
            x_out[0] = connect[0]
            y_out[0] = connect[1]
        if line:
            plt.plot(x_out,y_out,lw=lw,ls=ls,color=color)
        else:
            plt.plot(x_out, y_out, marker='v', color=color,ms=ms)

    def percentile_lines(self,x_in,y_in,Norm=True,connect_point=False,Bin=False,bin_in=None,fill=False,color='k',lw=3):
        '''Function to plot the 25 and 75 percentile lines'''
        x_out = np.nanmedian(x_in, axis=0)
        y_out_25 = 10**np.nanpercentile(y_in,25,axis=0)
        y_out_75 = 10**np.nanpercentile(y_in,75,axis=0)
        if Norm:
            if Bin:
                y_out_25 /= np.nanpercentile(self.norm[bin_in],25)
                y_out_75 /= np.nanpercentile(self.norm[bin_in],75)
            else:
                y_out_25 /= np.nanpercentile(self.norm,25)
                y_out_75 /= np.nanpercentile(self.norm,75)
        plt.plot(x_out, y_out_25, c=color, lw=lw, ls='--')
        plt.plot(x_out, y_out_75, c=color, lw=lw, ls='--')
        if fill:
            plt.fill_between(x_out, y_out_25, y_out_75, color=color, alpha=0.15)
        if connect_point:
            return x_out[-1], y_out_25[-1], y_out_75[-1]

    def percentile_lines_FIR(self,xfir,yfir,Norm=True,connect=[np.nan,np.nan,np.nan],upper='upper lims',fill=False,Bin=False,bin_in=None,color='k',lw=3):
        '''Function to plot the 25 and 75 percentile lines'''
        if upper == 'upper lims':
            yfir = yfir  # if upper is True, only used detections to determine median FIR. Default is to use detections + upperlimts
        elif upper == 'data only':
            if Bin:
                yfir = yfir[self.up_check[bin_in] == 0]
            else:
                yfir = yfir[self.up_check == 0]
        else:
            print('Specify if FIR upper limits should be included in median calc or only data. Options are: upper lims,    data only')
            return

        x_out = np.nanmean(xfir, axis=0)
        y_out_25 = np.nanpercentile(yfir, 25, axis=0)
        y_out_75 = np.nanpercentile(yfir, 75, axis=0)
        if Norm:  # If Norm is True, normalize FIR SED. Default is to normalize
            if Bin:
                y_out_25 /= np.nanpercentile(self.norm[bin_in], 25)
                y_out_75 /= np.nanpercentile(self.norm[bin_in], 75)

            else:
                y_out_25 /= np.nanpercentile(self.norm, 25)
                y_out_75 /= np.nanpercentile(self.norm, 75)
        if ~np.isnan(connect[0]):
            x_out[0] = connect[0]
            y_out_25[0] = connect[1]
            y_out_75[0] = connect[2]
        plt.plot(x_out, y_out_25, lw=lw, ls='--', color=color)
        plt.plot(x_out, y_out_75, lw=lw, ls='--', color=color)
        if fill:
            plt.fill_between(x_out, y_out_25, y_out_75, color=color, alpha=0.15)
   
    def PlotSED(self,point_x=np.nan,point_y=np.nan,save=False):
        fig, ax = plt.subplots(figsize=(10,8))

        ax.plot(self.wavelength,self.Lum)
        ax.plot(self.wavelength,self.Lum,'x',c='k')
        ax.plot(point_x,point_y,'x',c='r')
        # ax.plot(self.wfir,self.ffir,c='gray',lw=4)

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

    def multi_SED(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]],opt_p=[np.nan,np.nan],Median_line=True,FIR_med=True,FIR_upper='upper lims',percent=False):
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

        wfir_seg = np.delete(wfir, 0 , 1)
        ffir_seg = np.delete(ffir_norm, 0, 1)

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
        # ax.text(0.15, 0.85, f'n = {len(L)}', transform=ax.transAxes)
        
        # Plot the FIR upper limit segments 
        upper_seg = np.stack((wfir_seg, ffir_seg), axis=2)
        upper_all = LineCollection(upper_seg,color='gray',alpha=0.3)
        # ax.add_collection(upper_all)

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
            # ax.plot(np.nanmedian(x[:,:2],axis=0),np.nanmedian(y[:,:2],axis=0),c='k',lw=6)
            self.median_sed(x[:,:2],np.log10(y[:,:2]),Norm=False)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x,median_y,connect_point=True)
                self.median_FIR_sed(wfir,ffir,connect=[x_connect,y_connect],upper=FIR_upper)
            else:
                self.median_sed(median_x,median_y)
        if percent:
            x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x, median_y,connect_point=True,fill=True)
            self.percentile_lines_FIR(wfir,ffir, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, fill=True)
            self.percentile_lines(x[:,:2],np.log10(y[:,:2]),Norm=False,fill=True)

        # x, y = np.append(median_x,wfir), np.append(median_y,ffir)

        # f_interp = interpolate.interp1d(np.log10(x),y,kind='linear',fill_value='extrapolate') 

        # x_interp = np.arange(1E-1,350,0.5)
        # y_interp = f_interp(np.log10(x_interp))

        # plt.plot(x_interp,y_interp,color='r')
        # print(x_interp)
        # print(y_interp)
        # print(x)
        # print(y)

        # plt.axvline(0.11,color='b',lw=3)
        # plt.axvline(0.5,color='b',lw=3)
        # plt.fill_between([0.11,0.5],[1E-2,1E-2],[1E2,1E2],color='b',alpha=0.3)
        # ax.text(0.056 , 0.85, f'Accretion', transform=ax.transAxes,fontsize=30, weight='bold')
        # ax.text(0.056, 0.8 , f'Disk', transform=ax.transAxes,fontsize=30, weight='bold')
 
        # plt.axvline(1.8, color='orange', lw=3)
        # plt.axvline(10, color='orange', lw=3)
        # plt.fill_between([1.8, 10],[1E-2,1E-2],[1E2,1E2], color='orange', alpha=0.3)
        # ax.text(0.37 , 0.85, f'Dusty', transform=ax.transAxes,fontsize=30, weight='bold')
        # ax.text(0.37 , 0.8, f'Torus', transform=ax.transAxes,fontsize=30, weight='bold')

        # plt.axvline(30, color='red', lw=3)
        # plt.axvline(450, color='red', lw=3)
        # plt.fill_between([30, 450],[1E-2,1E-2],[1E2,1E2], color='red', alpha=0.3)
        # ax.text(0.69, 0.85, f'Cold', transform=ax.transAxes,fontsize=30, weight='bold')
        # ax.text(0.69, 0.8, f'Dust', transform=ax.transAxes,fontsize=30, weight='bold')
 

        plt.ylim(5E-4,5E2)
        plt.xlim(7E-5,700)
        # plt.ylim(1E-1, 40)
        # plt.xlim(7E-2,500)
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def multi_SED_bins(self, savestring, bin, field, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], opt_p=[np.nan, np.nan], Median_line=True, FIR_med=True, FIR_upper='upper lims'):
        '''Function to overplot all normalized SEDs with each line mapping to a colorbar and separated into three bins'''

        if bin == 'redshift':
            b1 = self.z <= 0.6
            b2 = (self.z > 0.6) & (self.z <= 0.9)
            b3 = (self.z > 0.9) & (self.z <= 1.2)
            t1 = 'z < 0.6'
            t2 = '0.6 < z < 0.9'
            t3 = '0.9 < z < 1.2'

        elif bin == 'field':
            b1 = field == 'g'
            b2 = field == 'c'
            b3 = field == 's'
            t1 = 'GOODS-N/S'
            t2 = 'COSMOS'
            t3 = 'Stripe82X'
        else:
            print('Specify bins. Options are: redshift,    field')
            return

        if  len(self.norm) == len(ffir):
            ffir_norm = ffir.T/self.norm
            ffir_norm = ffir_norm.T
        else:
            ffir_norm = [[np.nan]] 
        ffir_norm = np.asarray(ffir_norm)

        # Set colorbar limits
        clim1 = 43
        clim2 = 45.5
        cmap = 'rainbow_r'  # set colormap

        # remove sources with L outside colorbar range
        x = self.wavelength[self.L >= clim1-0.1]
        y = self.Lum[self.L >= clim1-0.1]
        L = self.L[self.L >= clim1-0.1]

        wfir_seg = np.delete(wfir, 0 , 1)
        ffir_seg = np.delete(ffir_norm, 0, 1)

        x1, x2, x3 = x[b1], x[b2], x[b3]
        y1, y2, y3 = y[b1], y[b2], y[b3] 
        L1, L2, L3 = L[b1], L[b2], L[b3]
        wfir1, wfir2, wfir3 = wfir[b1], wfir[b2], wfir[b3]
        ffir1, ffir2, ffir3 = ffir_norm[b1], ffir_norm[b2], ffir_norm[b3]
        wfir1_seg, wfir2_seg, wfir3_seg = wfir_seg[b1], wfir_seg[b2], wfir_seg[b3]
        ffir1_seg, ffir2_seg, ffir3_seg = ffir_seg[b1], ffir_seg[b2], ffir_seg[b3]
        median_x1, median_x2, median_x3, = median_x[b1], median_x[b2], median_x[b3]
        median_y1, median_y2, median_y3, = median_y[b1], median_y[b2], median_y[b3]

        xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
        yticks = [0.001,0.01,0.1,1,10,100]
        xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

        # mosaic = '''123'''
        # fig = plt.figure(figsize=(12,8))
        # axd = fig.subplot_mosaic(mosaic)
        # lc = axd['1'].self.multilines
        
        # Set up Plot
        fig = plt.figure(figsize=(24,8))
        gs = fig.add_gridspec(nrows=1,ncols=3,bottom=0.1,top=0.9,left=0.1,right=1,wspace=-0.15)

        ax1 = fig.add_subplot(gs[0], aspect='equal', adjustable='box')
        # ax1.set_box_aspect(1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
        ax1.set_title(t1)
        ax1.grid()
        ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')

        # Plot data
        upper_seg1 = np.stack((wfir1_seg,ffir1_seg), axis=2)
        upper_all1 = LineCollection(upper_seg1, color='gray', alpha=0.3)
        ax1.add_collection(upper_all1)
        lc1 = self.multilines(x1,y1,L1,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb1 = fig.colorbar(lc1)
        axcb1.mappable.set_clim(clim1, clim2)
        axcb1.remove()
        # Plot median line
        if Median_line:
            ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x1, median_y1, connect_point=True, Bin=True, bin_in=b1, lw=3)
                self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b1, lw=3, ls='--')
            else:
                self.median_sed(median_x1, median_y1,Bin=True, bin_in = b1, lw=3)
        plt.ylim(5E-4, 5E2)
        plt.xlim(7E-5, 700)

        ax2 = fig.add_subplot(gs[1], aspect='equal', adjustable='box')
        # ax2.set_box_aspect(1)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticklabels([])
        ax2.text(0.05, 0.8, f'n = {len(x2)}', transform=ax2.transAxes)
        ax2.set_title(t2)
        ax2.grid()
        ax2.set_xlabel(r'Rest Wavelength [$\mu$m]')

        upper_seg2 = np.stack((wfir2_seg,ffir2_seg), axis=2)
        upper_all2 = LineCollection(upper_seg2, color='gray', alpha=0.3)
        ax2.add_collection(upper_all2)
        lc2 = self.multilines(x2,y2,L2,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb2 = fig.colorbar(lc2)
        axcb2.mappable.set_clim(clim1, clim2)
        axcb2.remove()
        # Plot median line
        if Median_line:
            ax2.plot(np.nanmedian(x2[2:,:2],axis=0),np.nanmedian(y2[:,:2],axis=0),c='k',lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x2, median_y2, connect_point=True, Bin=True, bin_in=b2, lw=3)
                self.median_FIR_sed(wfir2,ffir2,connect=[x_connect,y_connect],upper=FIR_upper, Norm=False, Bin=True, bin_in=b2, lw=3, ls='--')
            else:
                self.median_sed(median_x2, median_y2, Bin=True, bin_in=b2, lw=3)
        plt.ylim(5E-4, 5E2)
        plt.xlim(7E-5, 700)

        ax3 = fig.add_subplot(gs[2], aspect='equal', adjustable='box')
        # ax3.set_box_aspect(1)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xticks(xticks)
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([])
        ax3.set_xticklabels(xticks_labels)
        ax3.text(0.05, 0.8, f'n = {len(x3)}', transform=ax3.transAxes)
        ax3.grid()
        ax3.set_title(t3)

        upper_seg3 = np.stack((wfir3_seg,ffir3_seg), axis=2)
        upper_all3 = LineCollection(upper_seg3, color='gray', alpha=0.3)
        ax3.add_collection(upper_all3)
        lc3 = self.multilines(x3,y3,L3,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb3 = fig.colorbar(lc3)
        axcb3.mappable.set_clim(clim1, clim2)
        axcb3.remove()
        # Plot median line
        if Median_line:
            ax3.plot(np.nanmedian(x3[:,:2],axis=0),np.nanmedian(y3[:,:2],axis=0),c='k',lw=3)
            if FIR_med:
                if bin=='redshift':
                    x_connect, y_connect = self.median_sed(median_x3, median_y3, connect_point=True, Bin=True, bin_in=b3,lw=3)
                    self.median_FIR_sed(wfir3,ffir3,connect=[x_connect,y_connect],upper=FIR_upper, Norm=False,Bin=True, bin_in=b3,lw=3, ls='--')
                else:
                    # wfir3 = np.delete(wfir3, 0, 1)
                    # ffir3 = np.delete(ffir3, 0, 1)
                    self.median_sed(median_x3, median_y3, Bin=True, bin_in=b3,lw=3)
                    # self.median_FIR_sed(wfir3,ffir3,upper=FIR_upper, Norm=False,Bin=True, bin_in=b3,lw=3)
            else:
                self.median_sed(median_x3,median_y3, Bin=True, bin_in=b3,lw=3)
        plt.ylim(5E-4, 5E2)
        plt.xlim(7E-5, 700)
        
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def median_SED_plot(self, savestring, median_x, median_y, wfir, ffir, shape, FIR_upper='upper lims',ls='-'):
        '''Function to plot the median SED of SEDs separated by defined SED shape and bined into three z bins'''

        x = self.wavelength
        y = self.Lum

        z1 = self.z <= 0.6
        z2 = (self.z > 0.6) & (self.z <= 0.9)
        z3 = (self.z > 0.9) & (self.z <= 1.2)

        z1_b1, z2_b1, z3_b1 = z1[shape == 1], z2[shape == 1], z3[shape == 1]
        z1_b2, z2_b2, z3_b2 = z1[shape == 2], z2[shape == 2], z3[shape == 2]
        z1_b3, z2_b3, z3_b3 = z1[shape == 3], z2[shape == 3], z3[shape == 3]
        z1_b4, z2_b4, z3_b4 = z1[shape == 4], z2[shape == 4], z3[shape == 4]
        z1_b5, z2_b5, z3_b5 = z1[shape == 5], z2[shape == 5], z3[shape == 5]

        median_x1_b1, median_x2_b1, median_x3_b1 = median_x[shape == 1][z1_b1], median_x[shape == 1][z2_b1], median_x[shape == 1][z3_b1]
        median_x1_b2, median_x2_b2, median_x3_b2 = median_x[shape == 2][z1_b2], median_x[shape == 2][z2_b2], median_x[shape == 2][z3_b2]
        median_x1_b3, median_x2_b3, median_x3_b3 = median_x[shape == 3][z1_b3], median_x[shape == 3][z2_b3], median_x[shape == 3][z3_b3]
        median_x1_b4, median_x2_b4, median_x3_b4 = median_x[shape == 4][z1_b4], median_x[shape == 4][z2_b4], median_x[shape == 4][z3_b4]
        median_x1_b5, median_x2_b5, median_x3_b5 = median_x[shape == 5][z1_b5], median_x[shape == 5][z2_b5], median_x[shape == 5][z3_b5]

        median_y1_b1, median_y2_b1, median_y3_b1 = median_y[shape == 1][z1_b1], median_y[shape == 1][z2_b1], median_y[shape == 1][z3_b1]
        median_y1_b2, median_y2_b2, median_y3_b2 = median_y[shape == 2][z1_b2], median_y[shape == 2][z2_b2], median_y[shape == 2][z3_b2]
        median_y1_b3, median_y2_b3, median_y3_b3 = median_y[shape == 3][z1_b3], median_y[shape == 3][z2_b3], median_y[shape == 3][z3_b3]
        median_y1_b4, median_y2_b4, median_y3_b4 = median_y[shape == 4][z1_b4], median_y[shape == 4][z2_b4], median_y[shape == 4][z3_b4]
        median_y1_b5, median_y2_b5, median_y3_b5 = median_y[shape == 5][z1_b5], median_y[shape == 5][z2_b5], median_y[shape == 5][z3_b5]

        wfir1_b1, wfir2_b1, wfir3_b1 = wfir[shape == 1][z1_b1], wfir[shape == 1][z2_b1], wfir[shape == 1][z3_b1]
        wfir1_b2, wfir2_b2, wfir3_b2 = wfir[shape == 2][z1_b2], wfir[shape == 2][z2_b2], wfir[shape == 2][z3_b2]
        wfir1_b3, wfir2_b3, wfir3_b3 = wfir[shape == 3][z1_b3], wfir[shape == 3][z2_b3], wfir[shape == 3][z3_b3]
        wfir1_b4, wfir2_b4, wfir3_b4 = wfir[shape == 4][z1_b4], wfir[shape == 4][z2_b4], wfir[shape == 4][z3_b4]
        wfir1_b5, wfir2_b5, wfir3_b5 = wfir[shape == 5][z1_b5], wfir[shape == 5][z2_b5], wfir[shape == 5][z3_b5]

        ffir1_b1, ffir2_b1, ffir3_b1 = ffir[shape == 1][z1_b1], ffir[shape == 1][z2_b1], ffir[shape == 1][z3_b1]
        ffir1_b2, ffir2_b2, ffir3_b2 = ffir[shape == 2][z1_b2], ffir[shape == 2][z2_b2], ffir[shape == 2][z3_b2]
        ffir1_b3, ffir2_b3, ffir3_b3 = ffir[shape == 3][z1_b3], ffir[shape == 3][z2_b3], ffir[shape == 3][z3_b3]
        ffir1_b4, ffir2_b4, ffir3_b4 = ffir[shape == 4][z1_b4], ffir[shape == 4][z2_b4], ffir[shape == 4][z3_b4]
        ffir1_b5, ffir2_b5, ffir3_b5 = ffir[shape == 5][z1_b5], ffir[shape == 5][z2_b5], ffir[shape == 5][z3_b5]

        x1_b1, x2_b1, x3_b1 = x[shape == 1][z1_b1], x[shape == 1][z2_b1], x[shape == 1][z3_b1]
        x1_b2, x2_b2, x3_b2 = x[shape == 2][z1_b2], x[shape == 2][z2_b2], x[shape == 2][z3_b2]
        x1_b3, x2_b3, x3_b3 = x[shape == 3][z1_b3], x[shape == 3][z2_b3], x[shape == 3][z3_b3]
        x1_b4, x2_b4, x3_b4 = x[shape == 4][z1_b4], x[shape == 4][z2_b4], x[shape == 4][z3_b4]
        x1_b5, x2_b5, x3_b5 = x[shape == 5][z1_b5], x[shape == 5][z2_b5], x[shape == 5][z3_b5]

        y1_b1, y2_b1, y3_b1 = y[shape == 1][z1_b1], y[shape == 1][z2_b1], y[shape == 1][z3_b1]
        y1_b2, y2_b2, y3_b2 = y[shape == 2][z1_b2], y[shape == 2][z2_b2], y[shape == 2][z3_b2]
        y1_b3, y2_b3, y3_b3 = y[shape == 3][z1_b3], y[shape == 3][z2_b3], y[shape == 3][z3_b3]
        y1_b4, y2_b4, y3_b4 = y[shape == 4][z1_b4], y[shape == 4][z2_b4], y[shape == 4][z3_b4]
        y1_b5, y2_b5, y3_b5 = y[shape == 5][z1_b5], y[shape == 5][z2_b5], y[shape == 5][z3_b5]

        norm1_b1, norm2_b1, norm3_b1 = self.norm[shape == 1][z1_b1], self.norm[shape == 1][z2_b1], self.norm[shape == 1][z3_b1]
        norm1_b2, norm2_b2, norm3_b2 = self.norm[shape == 2][z1_b2], self.norm[shape == 2][z2_b2], self.norm[shape == 2][z3_b2]
        norm1_b3, norm2_b3, norm3_b3 = self.norm[shape == 3][z1_b3], self.norm[shape == 3][z2_b3], self.norm[shape == 3][z3_b3]
        norm1_b4, norm2_b4, norm3_b4 = self.norm[shape == 4][z1_b4], self.norm[shape == 4][z2_b4], self.norm[shape == 4][z3_b4]
        norm1_b5, norm2_b5, norm3_b5 = self.norm[shape == 5][z1_b5], self.norm[shape == 5][z2_b5], self.norm[shape == 5][z3_b5] 


        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        yticks = [42, 43, 44, 45, 46]
        xticks = [1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2]
        ytick_labels = ['42', '43', '44', '45', '46']
        xticks_labels = [r'10$^{-4}$', '', r'10$^{-2}$', '', r'10$^{0}$', '', r'10$^{2}$']

        def solar(x):
            return x/3.8E33

        def ergs(x):
            return x*3.8E33

        fig = plt.figure(figsize=(24,7))
        gs = fig.add_gridspec(nrows=1, ncols=3)
        gs.update(wspace=0.08) # set the spacing between axes
        gs.update(left=0.06,right=0.94,top=0.9,bottom=0.15)

        ax1 = plt.subplot(gs[0])
        # x_connect, y_connect = self.median_sed(median_x1_b1, median_y1_b1, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1_b1, median_y1_b1, Norm=False,connect_point=True,color=c1,lw=4)
        self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')
        # self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1_b1, median_y1_b1, Norm=False,connect_point=True,fill=True,color=c1,lw=2)
        # self.percentile_lines_FIR(wfir1_b1,ffir1_b1, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c1,lw=2)
        ax1.plot(np.nanmedian(x1_b1[:, :2], axis=0),np.nanmedian(y1_b1[:, :2], axis=0)*np.nanmedian(norm1_b1), c=c1, lw=4)

        # x_connect, y_connect = self.median_sed(median_x1_b2, median_y1_b2, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1_b2, median_y1_b2, Norm=False,connect_point=True,color=c2,lw=4)
        self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')
        # self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1_b2, median_y1_b2, Norm=False,connect_point=True,fill=True,color=c2,lw=2)
        # self.percentile_lines_FIR(wfir1_b2,ffir1_b2, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c2,lw=2)
        ax1.plot(np.nanmedian(x1_b2[:, :2], axis=0),np.nanmedian(y1_b2[:, :2], axis=0)*np.nanmedian(norm1_b2), c=c2, lw=4)

        # x_connect, y_connect = self.median_sed(median_x1_b3, median_y1_b3, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1_b3, median_y1_b3, Norm=False,connect_point=True,color=c3,lw=4)
        self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')
        # self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1_b3, median_y1_b3, Norm=False,connect_point=True,fill=True,color=c3,lw=2)
        # self.percentile_lines_FIR(wfir1_b3,ffir1_b3, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c3,lw=2)
        ax1.plot(np.nanmedian(x1_b3[:, :2], axis=0),np.nanmedian(y1_b3[:, :2], axis=0)*np.nanmedian(norm1_b3), c=c3, lw=4)

        # x_connect, y_connect = self.median_sed(median_x1_b4, median_y1_b4, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1_b4, median_y1_b4, Norm=False,connect_point=True,color=c4,lw=4)
        self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,ls='--')
        # self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1_b4, median_y1_b4, Norm=False,connect_point=True,fill=True,color=c4,lw=2)
        # self.percentile_lines_FIR(wfir1_b4,ffir1_b4, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c4,lw=2)
        ax1.plot(np.nanmedian(x1_b4[:, :2], axis=0),np.nanmedian(y1_b4[:, :2], axis=0)*np.nanmedian(norm1_b4), c=c4, lw=4)

        # x_connect, y_connect = self.median_sed(median_x1_b5, median_y1_b5, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1_b5, median_y1_b5, Norm=False,connect_point=True,color=c5,lw=4)
        self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,ls='--')
        # self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1_b5, median_y1_b5, Norm=False,connect_point=True,fill=True,color=c5,lw=2)
        # self.percentile_lines_FIR(wfir1_b5,ffir1_b5, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c5,lw=2)
        ax1.plot(np.nanmedian(x1_b5[:, :2], axis=0),np.nanmedian(y1_b5[:, :2], axis=0)*np.nanmedian(norm1_b5), c=c5, lw=4)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim(1E42, 1E46)
        ax1.set_xlim(7E-5, 700)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.set_ylabel(r'$\lambda$ L$_\lambda$ [erg/s]')
        ax1.grid()

        ax2 = plt.subplot(gs[1])
        # x_connect, y_connect = self.median_sed(median_x2_b1, median_y2_b1, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir2_b1, ffir2_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x2_b1, median_y2_b1, Norm=False,connect_point=True,color=c1,lw=4)
        self.median_FIR_sed(wfir2_b1, ffir2_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2_b1, median_y2_b1, Norm=False,connect_point=True,fill=True,color=c1,lw=2)
        # self.percentile_lines_FIR(wfir2_b1,ffir2_b1, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c1,lw=2)
        ax2.plot(np.nanmedian(x2_b1[:, :2], axis=0),np.nanmedian(y2_b1[:, :2], axis=0)*np.nanmedian(norm2_b1), c=c1, lw=4)

        # x_connect, y_connect = self.median_sed(median_x2_b2, median_y2_b2, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir2_b2, ffir2_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x2_b2, median_y2_b2, Norm=False,connect_point=True,color=c2,lw=4)
        self.median_FIR_sed(wfir2_b2, ffir2_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2_b2, median_y2_b2, Norm=False,connect_point=True,fill=True,color=c2,lw=2)
        # self.percentile_lines_FIR(wfir2_b2,ffir2_b2, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c2,lw=2)
        ax2.plot(np.nanmedian(x2_b2[:, :2], axis=0),np.nanmedian(y2_b2[:, :2], axis=0)*np.nanmedian(norm2_b2), c=c2, lw=4)
       
        # x_connect, y_connect = self.median_sed(median_x2_b3, median_y2_b3, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir2_b3, ffir2_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x2_b3, median_y2_b3, Norm=False,connect_point=True,color=c3,lw=4)
        self.median_FIR_sed(wfir2_b3, ffir2_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2_b3, median_y2_b3, Norm=False,connect_point=True,fill=True,color=c3,lw=2)
        # self.percentile_lines_FIR(wfir2_b3,ffir2_b3, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c3,lw=2)
        ax2.plot(np.nanmedian(x2_b3[:, :2], axis=0),np.nanmedian(y2_b3[:, :2], axis=0)*np.nanmedian(norm2_b3), c=c3, lw=4)

        # x_connect, y_connect = self.median_sed(median_x2_b4, median_y2_b4, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir2_b4, ffir2_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x2_b4, median_y2_b4, Norm=False,connect_point=True,color=c4,lw=4)
        self.median_FIR_sed(wfir2_b4, ffir2_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2_b4, median_y2_b4, Norm=False,connect_point=True,fill=True,color=c4,lw=2)
        # self.percentile_lines_FIR(wfir2_b4,ffir2_b4, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c1,lw=2)
        ax2.plot(np.nanmedian(x2_b4[:, :2], axis=0),np.nanmedian(y2_b4[:, :2], axis=0)*np.nanmedian(norm2_b4), c=c4, lw=4)

        # x_connect, y_connect = self.median_sed(median_x2_b5, median_y2_b5, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir2_b5, ffir2_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x2_b5, median_y2_b5, Norm=False,connect_point=True,color=c5,lw=4)
        self.median_FIR_sed(wfir2_b5, ffir2_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2_b5, median_y2_b5, Norm=False,connect_point=True,fill=True,color=c5,lw=2)
        # self.percentile_lines_FIR(wfir2_b5,ffir2_b5, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c5,lw=2)
        ax2.plot(np.nanmedian(x2_b5[:, :2], axis=0),np.nanmedian(y2_b5[:, :2], axis=0)*np.nanmedian(norm2_b5), c=c5, lw=4)

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(1E42, 1E46)
        ax2.set_xlim(7E-5, 700)
        ax2.set_yticklabels([])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticks_labels)
        ax2.set_xlabel(r'Rest Wavelength [$\mu$m]')
        ax2.grid()

        ax3 = plt.subplot(gs[2])
        # x_connect, y_connect = self.median_sed(median_x3_b1, median_y3_b1, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir3_b1, ffir3_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x3_b1, median_y3_b1, Norm=False,connect_point=True,color=c1,lw=4)
        self.median_FIR_sed(wfir3_b1, ffir3_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3_b1, median_y3_b1, Norm=False,connect_point=True,fill=True,color=c1,lw=1)
        # self.percentile_lines_FIR(wfir3_b1,ffir3_b1, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c1,lw=1)
        ax3.plot(np.nanmedian(x3_b1[:, :2], axis=0),np.nanmedian(y3_b1[:, :2], axis=0)*np.nanmedian(norm3_b1), c=c1, lw=4)

        # x_connect, y_connect = self.median_sed(median_x3_b2, median_y3_b2, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir3_b2, ffir3_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x3_b2, median_y3_b2, Norm=False,connect_point=True,color=c2,lw=4)
        self.median_FIR_sed(wfir3_b2, ffir3_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3_b2, median_y3_b2, Norm=False,connect_point=True,fill=True,color=c2,lw=1)
        # self.percentile_lines_FIR(wfir3_b2,ffir3_b2, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c2,lw=1)
        ax3.plot(np.nanmedian(x3_b2[:, :2], axis=0),np.nanmedian(y3_b2[:, :2], axis=0)*np.nanmedian(norm3_b2), c=c2, lw=4)

        # x_connect, y_connect = self.median_sed(median_x3_b3, median_y3_b3, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir3_b3, ffir3_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x3_b3, median_y3_b3, Norm=False,connect_point=True,color=c3,lw=4)
        self.median_FIR_sed(wfir3_b3, ffir3_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3_b3, median_y3_b3, Norm=False,connect_point=True,fill=True,color=c3,lw=1)
        # self.percentile_lines_FIR(wfir3_b3,ffir3_b3, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c3,lw=1)
        ax3.plot(np.nanmedian(x3_b3[:, :2], axis=0),np.nanmedian(y3_b3[:, :2], axis=0)*np.nanmedian(norm3_b3), c=c3, lw=4)

        # x_connect, y_connect = self.median_sed(median_x3_b4, median_y3_b4, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir3_b4, ffir3_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x3_b4, median_y3_b4, Norm=False,connect_point=True,color=c4,lw=4)
        self.median_FIR_sed(wfir3_b4, ffir3_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3_b4, median_y3_b4, Norm=False,connect_point=True,fill=True,color=c4,lw=1)
        # self.percentile_lines_FIR(wfir3_b4,ffir3_b4, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c4,lw=1)
        ax3.plot(np.nanmedian(x3_b4[:, :2], axis=0),np.nanmedian(y3_b4[:, :2], axis=0)*np.nanmedian(norm3_b4), c=c4, lw=4)

        # x_connect, y_connect = self.median_sed(median_x3_b5, median_y3_b5, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir3_b5, ffir3_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        x_connect, y_connect = self.median_sed(median_x3_b5, median_y3_b5, Norm=False,connect_point=True,color=c5,lw=4)
        self.median_FIR_sed(wfir3_b5, ffir3_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,ls='--')
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3_b5, median_y3_b5, Norm=False,connect_point=True,fill=True,color=c5,lw=1)
        # self.percentile_lines_FIR(wfir3_b5,ffir3_b5, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c5,lw=1)
        ax3.plot(np.nanmedian(x3_b5[:, :2], axis=0),np.nanmedian(y3_b5[:, :2], axis=0)*np.nanmedian(norm3_b5), c=c5, lw=4)

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_ylim(1E42, 1E46)
        ax3.set_xlim(7E-5, 700)
        ax3.set_yticklabels([])
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xticks_labels)
        secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
        secax3.set_ylabel(r'$\lambda$ L$_\lambda$ [L$_{\odot}$]')
        ax3.grid()

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def median_SED_1panel(self, savestring, median_x, median_y, wfir, ffir, shape, FIR_upper='upper lims',ls='-'):
        '''Function to plot the median SED of SEDs separated by defined SED shape and bined into three z bins'''
        plt.rcParams['font.size'] = 35
        plt.rcParams['axes.linewidth'] = 4.5
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 5
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 5
        plt.rcParams['xtick.minor.size'] = 4.
        plt.rcParams['xtick.minor.width'] = 3.
        plt.rcParams['ytick.minor.size'] = 4.
        plt.rcParams['ytick.minor.width'] = 3.

        x = self.wavelength
        y = self.Lum

        b1 = shape == 1
        b2 = shape == 2
        b3 = shape == 3
        b4 = shape == 4
        b5 = shape == 5

        median_x1 = median_x[b1]
        median_x2 = median_x[b2]
        median_x3 = median_x[b3]
        median_x4 = median_x[b4]
        median_x5 = median_x[b5]

        median_y1 = median_y[b1]
        median_y2 = median_y[b2]
        median_y3 = median_y[b3]
        median_y4 = median_y[b4]
        median_y5 = median_y[b5]

        wfir1 = wfir[b1]
        wfir2 = wfir[b2]
        wfir3 = wfir[b3]
        wfir4 = wfir[b4]
        wfir5 = wfir[b5]

        ffir1 = ffir[b1]
        ffir2 = ffir[b2]
        ffir3 = ffir[b3]
        ffir4 = ffir[b4]
        ffir5 = ffir[b5]

        x1 = x[b1]
        x2 = x[b2]
        x3 = x[b3]
        x4 = x[b4]
        x5 = x[b5]

        y1 = y[b1]
        y2 = y[b2]
        y3 = y[b3]
        y4 = y[b4]
        y5 = y[b5]

        norm1 = self.norm[b1]
        norm2 = self.norm[b2]
        norm3 = self.norm[b3]
        norm4 = self.norm[b4]
        norm5 = self.norm[b5] 


        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        yticks = [42, 43, 44, 45, 46]
        # xticks = [1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2]
        xticks = [1E-1, 1E0, 1E1, 1E2]
        ytick_labels = ['42', '43', '44', '45', '46']
        # xticks_labels = [r'10$^{-4}$', '', r'10$^{-2}$', '', r'10$^{0}$', '', r'10$^{2}$']
        xticks_labels = [r'10$^{-4}$', '', r'10$^{-2}$', '', r'10$^{0}$', '', r'10$^{2}$']

        fig = plt.figure(figsize=(18,12))
        gs = fig.add_gridspec(nrows=1, ncols=1)
        # gs.update(wspace=0.08) # set the spacing between axes
        # gs.update(left=0.125,right=0.95,top=0.9,bottom=0.1)

        # median_y1[-1] = median_y1[-1]*0.9

        ax1 = plt.subplot(gs[0], aspect='equal', adjustable='box')
        # x_connect, y_connect = self.median_sed(median_x1_b1, median_y1_b1, Norm=False,connect_point=True,color='k',lw=4.5)
        # self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # self.median_FIR_sed(wfir1_b1, ffir1_b1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=False,connect_point=True,color=c1,lw=4)
        self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')
        # self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,line=False)
        # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x1, median_y1, Norm=False,connect_point=True,fill=True,color=c1,lw=2)
        # self.percentile_lines_FIR(wfir1,ffir1, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c1,lw=2)
        # ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0)*np.nanmedian(norm1), c=c1, lw=4)

        # # x_connect, y_connect = self.median_sed(median_x1_b2, median_y1_b2, Norm=False,connect_point=True,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # # self.median_FIR_sed(wfir1_b2, ffir1_b2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x2, median_y2, Norm=False,connect_point=True,color=c2,lw=4)
        self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')
        # # self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,line=False)
        # # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x2, median_y2, Norm=False,connect_point=True,fill=True,color=c2,lw=2)
        # # self.percentile_lines_FIR(wfir2,ffir2, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c2,lw=2)
        # ax1.plot(np.nanmedian(x2[:, :2], axis=0),np.nanmedian(y2[:, :2], axis=0)*np.nanmedian(norm2), c=c2, lw=4)

        # # x_connect, y_connect = self.median_sed(median_x1_b3, median_y1_b3, Norm=False,connect_point=True,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,ls=ls)
        # # self.median_FIR_sed(wfir1_b3, ffir1_b3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x3, median_y3, Norm=False,connect_point=True,color=c3,lw=4)
        self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')
        # # self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,line=False)
        # # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x3, median_y3, Norm=False,connect_point=True,fill=True,color=c3,lw=2)
        # # self.percentile_lines_FIR(wfir3,ffir3, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c3,lw=2)
        # ax1.plot(np.nanmedian(x3[:, :2], axis=0),np.nanmedian(y3[:, :2], axis=0)*np.nanmedian(norm3), c=c3, lw=4)

        # # x_connect, y_connect = self.median_sed(median_x1_b4, median_y1_b4, Norm=False,connect_point=True,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b4, ffir1_b4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x4, median_y4, Norm=False,connect_point=True,color=c4,lw=4)
        self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,ls='--')
        # # self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,line=False)
        # # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x4, median_y4, Norm=False,connect_point=True,fill=True,color=c4,lw=2)
        # # self.percentile_lines_FIR(wfir4,ffir4, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c4,lw=2)
        # ax1.plot(np.nanmedian(x4[:, :2], axis=0),np.nanmedian(y4[:, :2], axis=0)*np.nanmedian(norm4), c=c4, lw=4)

        # # x_connect, y_connect = self.median_sed(median_x1_b5, median_y1_b5, Norm=False,connect_point=True,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5)
        # # self.median_FIR_sed(wfir1_b5, ffir1_b5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='k',lw=4.5,line=False,ms=12)
        x_connect, y_connect = self.median_sed(median_x5, median_y5, Norm=False,connect_point=True,color=c5,lw=4)
        self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,ls='--')
        # # self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,line=False)
        # # x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x5, median_y5, Norm=False,connect_point=True,fill=True,color=c5,lw=2)
        # # self.percentile_lines_FIR(wfir5,ffir5, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, Norm=False, fill=True,color=c5,lw=2)
        # ax1.plot(np.nanmedian(x5[:, :2], axis=0),np.nanmedian(y5[:, :2], axis=0)*np.nanmedian(norm5), c=c5, lw=4)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.set_ylim(3E41, 3E46)
        ax1.set_ylim(1E42,1E46)
        ax1.set_xlim(5E-2, 450)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.set_ylabel(r'$\lambda$ L$_\lambda$ [erg/s]')
        ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')
        secax1 = ax1.secondary_yaxis('right', functions=(self.solar, self.ergs))
        secax1.set_yticks([9, 10, 11, 12, 13])
        secax1.set_ylabel(r'$\lambda$ L$_\lambda$ [L$_{\odot}$]')
        ax1.grid()

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def plot_medians(self,savestring, F1, uv, mir, fir):
        '''Function to plot the median values between different luminosities'''
        Lx = self.L

        b1 = self.z < 0.6
        b2 = (self.z > 0.6) & (self.z < 0.9)
        b3 = self.z > 0.9

        uv1, uv2, uv3 = np.log10(uv[b1]), np.log10(uv[b2]), np.log10(uv[b3])
        mir1, mir2, mir3 = np.log10(mir[b1]), np.log10(mir[b2]), np.log10(mir[b3])
        fir1, fir2, fir3 = np.log10(fir[b1]), np.log10(fir[b2]), np.log10(fir[b3])
        Lx1, Lx2, Lx3 = Lx[b1], Lx[b2], Lx[b3]

        y11 = uv1 - Lx1
        y12 = uv2 - Lx2
        y13 = uv3 - Lx3

        y21 = mir1 - Lx1
        y22 = mir2 - Lx2
        y23 = mir3 - Lx3

        y31 = fir1 - Lx1
        y32 = fir2 - Lx2
        y33 = fir3 - Lx3

        mb11 = Lx1 < 43.5
        mb12 = (Lx1 > 43.5) & (Lx1 < 44)
        mb13 = (Lx1 > 44) & (Lx1 < 44.5)
        mb14 = (Lx1 > 44.5) & (Lx1 < 45)
        mb15 = (Lx1 > 45)

        mb21 = Lx2 < 43.5
        mb22 = (Lx2 > 43.5) & (Lx2 < 44)
        mb23 = (Lx2 > 44) & (Lx2 < 44.5)
        mb24 = (Lx2 > 44.5) & (Lx2 < 45)
        mb25 = (Lx2 > 45)

        mb31 = Lx3 < 43.5
        mb32 = (Lx3 > 43.5) & (Lx3 < 44)
        mb33 = (Lx3 > 44) & (Lx3 < 44.5)
        mb34 = (Lx3 > 44.5) & (Lx3 < 45)
        mb35 = (Lx3 > 45)

        x11, x12, x13, x14, x15 = np.nanmedian(Lx1[mb11]), np.nanmedian(Lx1[mb12]), np.nanmedian(Lx1[mb13]), np.nanmedian(Lx1[mb14]), np.nanmedian(Lx1[mb15])
        x21, x22, x23, x24, x25 = np.nanmedian(Lx2[mb21]), np.nanmedian(Lx2[mb22]), np.nanmedian(Lx2[mb23]), np.nanmedian(Lx2[mb24]), np.nanmedian(Lx2[mb25])
        x31, x32, x33, x34, x35 = np.nanmedian(Lx3[mb31]), np.nanmedian(Lx3[mb32]), np.nanmedian(Lx3[mb33]), np.nanmedian(Lx3[mb34]), np.nanmedian(Lx3[mb35])

        y11_1, y11_2, y11_3, y11_4, y11_5 = np.nanmedian(y11[mb11]), np.nanmedian(y11[mb12]), np.nanmedian(y11[mb13]), np.nanmedian(y11[mb14]), np.nanmedian(y11[mb15])
        y12_1, y12_2, y12_3, y12_4, y12_5 = np.nanmedian(y12[mb21]), np.nanmedian(y12[mb22]), np.nanmedian(y12[mb23]), np.nanmedian(y12[mb24]), np.nanmedian(y12[mb25])
        y13_1, y13_2, y13_3, y13_4, y13_5 = np.nanmedian(y13[mb31]), np.nanmedian(y13[mb32]), np.nanmedian(y13[mb33]), np.nanmedian(y13[mb34]), np.nanmedian(y13[mb35])

        y21_1, y21_2, y21_3, y21_4, y21_5 = np.nanmedian(y21[mb11]), np.nanmedian(y21[mb12]), np.nanmedian(y21[mb13]), np.nanmedian(y21[mb14]), np.nanmedian(y21[mb15])
        y22_1, y22_2, y22_3, y22_4, y22_5 = np.nanmedian(y22[mb21]), np.nanmedian(y22[mb22]), np.nanmedian(y22[mb23]), np.nanmedian(y22[mb24]), np.nanmedian(y22[mb25])
        y23_1, y23_2, y23_3, y23_4, y23_5 = np.nanmedian(y23[mb31]), np.nanmedian(y23[mb32]), np.nanmedian(y23[mb33]), np.nanmedian(y23[mb34]), np.nanmedian(y23[mb35])

        y31_1, y31_2, y31_3, y31_4, y31_5 = np.nanmedian(y31[mb11]), np.nanmedian(y31[mb12]), np.nanmedian(y31[mb13]), np.nanmedian(y31[mb14]), np.nanmedian(y31[mb15])
        y32_1, y32_2, y32_3, y32_4, y32_5 = np.nanmedian(y32[mb21]), np.nanmedian(y32[mb22]), np.nanmedian(y32[mb23]), np.nanmedian(y32[mb24]), np.nanmedian(y32[mb25])
        y33_1, y33_2, y33_3, y33_4, y33_5 = np.nanmedian(y33[mb31]), np.nanmedian(y33[mb32]), np.nanmedian(y33[mb33]), np.nanmedian(y33[mb34]), np.nanmedian(y33[mb35])

        print(y11_5)
        print(y12_5)
        print(y13_5)

        xticks = [42,43,44,45,46]
        yticks = [-2,-1,0,1,2]

        fig = plt.figure(figsize=(21, 10))
        ax1 = plt.subplot(131, aspect='equal', adjustable='box')

        ax1.scatter(x11, y11_1, color='blue', marker='s', s=120,label=r'a = 0.25$\mu$m')
        ax1.scatter(x12, y11_2, color='blue', marker='s', s=120)
        ax1.scatter(x13, y11_3, color='blue', marker='s', s=120)
        ax1.scatter(x14, y11_4, color='blue', marker='s', s=120)
        ax1.scatter(x15, y11_5, color='blue', marker='s', s=120)

        ax1.scatter(x11, y21_1, color='red', marker='o', s=120,label=r'a = 6$\mu$m')
        ax1.scatter(x12, y21_2, color='red', marker='o', s=120)
        ax1.scatter(x13, y21_3, color='red', marker='o', s=120)
        ax1.scatter(x14, y21_4, color='red', marker='o', s=120)
        ax1.scatter(x15, y21_5, color='red', marker='o', s=120)

        ax1.scatter(x11, y31_1, color='orange', marker='P', s=120,label=r'a = 100$\mu$m')
        ax1.scatter(x12, y31_2, color='orange', marker='P', s=120)
        ax1.scatter(x13, y31_3, color='orange', marker='P', s=120)
        ax1.scatter(x14, y31_4, color='orange', marker='P', s=120)
        ax1.scatter(x15, y31_5, color='orange', marker='P', s=120)

        ax1.set_title('z < 0.6')
        ax1.set_ylabel(r'log L$_{\mathrm{a}}$/L$_{\mathrm{X}}$')
        ax1.legend(fontsize=15)
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xlim(42.,46.)
        ax1.set_ylim(-2,2)
        ax1.grid()

        ax2 = plt.subplot(132, aspect='equal', adjustable='box')
        ax2.scatter(x21, y12_1, color='blue', marker='s',s=120)
        ax2.scatter(x22, y12_2, color='blue', marker='s',s=120)
        ax2.scatter(x23, y12_3, color='blue', marker='s',s=120)
        ax2.scatter(x24, y12_4, color='blue', marker='s',s=120)
        ax2.scatter(x25, y12_5, color='blue', marker='s',s=120)

        ax2.scatter(x21, y22_1, color='red', marker='o',s=120)
        ax2.scatter(x22, y22_2, color='red', marker='o',s=120)
        ax2.scatter(x23, y22_3, color='red', marker='o',s=120)
        ax2.scatter(x24, y22_4, color='red', marker='o',s=120)
        ax2.scatter(x25, y22_5, color='red', marker='o',s=120)

        ax2.scatter(x21, y32_1, color='orange', marker='P',s=120)
        ax2.scatter(x22, y32_2, color='orange', marker='P',s=120)
        ax2.scatter(x23, y32_3, color='orange', marker='P',s=120)
        ax2.scatter(x24, y32_4, color='orange', marker='P',s=120)
        ax2.scatter(x25, y32_5, color='orange', marker='P',s=120)

        ax2.set_title('0.6 < z < 0.9')
        ax2.set_xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
        ax2.set_yticklabels([])
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_xlim(42., 46.)
        ax2.set_ylim(-2, 2)
        ax2.grid()

        ax3 = plt.subplot(133, aspect='equal', adjustable='box')
        ax3.scatter(x31, y13_1, color='blue', marker='s',s=120)
        ax3.scatter(x32, y13_2, color='blue', marker='s',s=120)
        ax3.scatter(x33, y13_3, color='blue', marker='s',s=120)
        ax3.scatter(x34, y13_4, color='blue', marker='s',s=120)
        ax3.scatter(x35, y13_5, color='blue', marker='s',s=120)

        ax3.scatter(x31, y23_1, color='red', marker='o',s=120)
        ax3.scatter(x32, y23_2, color='red', marker='o',s=120)
        ax3.scatter(x33, y23_3, color='red', marker='o',s=120)
        ax3.scatter(x34, y23_4, color='red', marker='o',s=120)
        ax3.scatter(x35, y23_5, color='red', marker='o',s=120)

        ax3.scatter(x31, y33_1, color='orange', marker='P',s=120)
        ax3.scatter(x32, y33_2, color='orange', marker='P',s=120)
        ax3.scatter(x33, y33_3, color='orange', marker='P',s=120)
        ax3.scatter(x34, y33_4, color='orange', marker='P',s=120)
        ax3.scatter(x35, y33_5, color='orange', marker='P',s=120)

        ax3.set_title('0.9 < z < 1.2')
        ax3.set_yticklabels([])
        ax3.set_xticks(xticks)
        ax3.set_yticks(yticks)
        ax3.set_xlim(42., 46.)
        ax3.set_ylim(-2, 2)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_ratio_3panels(self,savestring,X,Y,median,F1,uv,mir,fir,shape,L=None):
        '''Function to plot the ratio of two luminosites as a function of the denominator'''

        bs1 = shape == 1
        bs2 = shape == 2
        bs3 = shape == 3
        bs4 = shape == 4
        bs5 = shape == 5

        if X == 'Lx':
            x = self.L
            xlabel = r'log L$_{\mathrm{X}}$'
            xunits = ' [erg/s]'
            xticks = [43.5,44.5,45.5]
            xlim = [42.75,45.75]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)

        elif X == 'Lbol':
            x = L
            xlabel = r'log L$_{\mathrm{bol}}' 
            xunits = ' [erg/s]'
            xticks = [43.5,44.5,45.5,46.5]
            xlim = [42.5,46.5]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)
        
        else:
            print('Provide valid X option. Options are:    Lx,    Lbol')
            return

        if Y == 'UV-MIR-FIR':
            # y1 = np.log10(uv) - x
            # y2 = np.log10(mir) - x
            # y3 = np.log10(fir) - x

            y1 = np.log10(uv)
            y2 = np.log10(mir)
            y3 = np.log10(fir)

            # ylabel1 = r'log L (0.25$\mu$m)/L$_{\mathrm{X}}$'  
            # ylabel2 = r'log L (6$\mu$m)/L$_{\mathrm{X}}$'
            # ylabel3 = r'log L (100$\mu$m)/L$_{\mathrm{X}}$'  
            ylabel1 = r'log L (0.25$\mu$m)'
            ylabel2 = r'log L (6$\mu$m)'
            ylabel3 = r'log L (100$\mu$m)'

            # yticks1 = [-1,0,1]   
            # yticks2 = [-1,0,1]
            # yticks3 = [0,1,2]
            # ylim1 = [-1.5,1.5]
            # ylim2 = [-1.5,1.5]
            # ylim3 = [-1, 2]

            yticks1 = [43,44,45]
            ylim1 = [42.5,45.7]
            yticks2 = [43,44,45]
            ylim2 = [42.5,45.7]
            yticks3 = [ 43, 44, 45]
            ylim3 = [42.5, 45.7]

        else:
            print('Provide valid Y option. Options are:    UV-MIR-FIR')

        
        # Set median points for X-axis bins
        xmed1, y1med1, y2med1, y3med1, y3med_detect1 = np.nanmean(x[bx1]), np.nanmean(y1[bx1]), np.nanmean(y2[bx1]), np.nanmean(y3[bx1]), np.nanmean(y3[bx1][self.up_check[bx1] == 0])
        xmed2, y1med2, y2med2, y3med2, y3med_detect2 = np.nanmean(x[bx2]), np.nanmean(y1[bx2]), np.nanmean(y2[bx2]), np.nanmean(y3[bx2]), np.nanmean(y3[bx2][self.up_check[bx2] == 0])
        xmed3, y1med3, y2med3, y3med3, y3med_detect3 = np.nanmean(x[bx3]), np.nanmean(y1[bx3]), np.nanmean(y2[bx3]), np.nanmean(y3[bx3]), np.nanmean(y3[bx3][self.up_check[bx3] == 0])
        xmed4, y1med4, y2med4, y3med4, y3med_detect4 = np.nanmean(x[bx4]), np.nanmean(y1[bx4]), np.nanmean(y2[bx4]), np.nanmean(y3[bx4]), np.nanmean(y3[bx4][self.up_check[bx4] == 0])
        xmed5, y1med5, y2med5, y3med5, y3med_detect5 = np.nanmean(x[bx5]), np.nanmean(y1[bx5]), np.nanmean(y2[bx5]), np.nanmean(y3[bx5]), np.nanmean(y3[bx5][self.up_check[bx5] == 0])

        y1std1, y2std1, y3std1, y3std_detect1 = np.std(y1[bx1]), np.std(y2[bx1]), np.std(y3[bx1]), np.std(y3[bx1][self.up_check[bx1] == 0])
        y1std2, y2std2, y3std2, y3std_detect2 = np.std(y1[bx2]), np.std(y2[bx2]), np.std(y3[bx2]), np.std(y3[bx2][self.up_check[bx2] == 0])
        y1std3, y2std3, y3std3, y3std_detect3 = np.std(y1[bx3]), np.std(y2[bx3]), np.std(y3[bx3]), np.std(y3[bx3][self.up_check[bx3] == 0])
        y1std4, y2std4, y3std4, y3std_detect4 = np.std(y1[bx4]), np.std(y2[bx4]), np.std(y3[bx4]), np.std(y3[bx4][self.up_check[bx4] == 0])
        y1std5, y2std5, y3std5, y3std_detect5 = np.std(y1[bx5]), np.std(y2[bx5]), np.std(y3[bx5]), np.std(y3[bx5][self.up_check[bx5] == 0])

        xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        y1med = np.array([y1med1, y1med2, y1med3, y1med4, y1med5])
        y2med = np.array([y2med1, y2med2, y2med3, y2med4, y2med5])
        y3med = np.array([y3med1, y3med2, y3med3, y3med4, y3med5])
        y3med_detect = np.array([y3med_detect1, y3med_detect2, y3med_detect3, y3med_detect4, y3med_detect5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5])
        y2std = np.array([y2std1, y2std2, y2std3, y2std4, y2std5])
        y3std = np.array([y3std1, y3std2, y3std3, y3std4, y3std5])
        y3std_detect = np.array([y3std_detect1, y3std_detect2, y3std_detect3, y3std_detect4, y3std_detect5])

        stern_Lx = Lit_functions.Stern_MIR(np.arange(42, 48))

        fig = plt.figure(figsize=(21, 8))
        gs = fig.add_gridspec(ncols=3,nrows=1,left=0.04,right=0.99,top=0.86,bottom=0.14,wspace=0.09)
        ax1 = plt.subplot(gs[0], aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0],xlim[1])
        ax1.set_ylim(ylim1[0],ylim1[1])
        ax1.set_ylabel(ylabel1)
        ax1.set_yticks(yticks1)
        ax1.set_xticks(xticks)
        ax1.grid()
        ax1.scatter(x,y1,color='gray',marker='P',s=10,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax1.plot(xmed, y1med, marker='s', color='k',ms=10)
            ax1.errorbar(xmed,y1med,xerr=xstd,yerr=y1std,color='k')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        # secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')

        ax2 = plt.subplot(gs[1], aspect='equal', adjustable='box')
        ax2.set_xlim(xlim[0],xlim[1])
        ax2.set_ylim(ylim2[0],ylim2[1])
        ax2.set_ylabel(ylabel2)
        ax2.set_xlabel(xlabel+xunits)
        ax2.set_yticks(yticks2)
        ax2.set_xticks(xticks)
        ax2.grid()
        ax2.scatter(x, y2, color='gray', marker='P', s=10,rasterized=True)
        # ax2.plot(stern_Lx,np.arange(42,48),color='r')
        if median == 'X-axis' or median == 'Both':
            ax2.plot(xmed, y2med, marker='s', color='k', ms=10)
            ax2.errorbar(xmed, y2med, xerr=xstd, yerr=y2std, color='k')
        secax2 = ax2.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax2.set_xlabel(xlabel+r' [L$_{\odot}$]')

        ax3 = plt.subplot(gs[2], aspect='equal', adjustable='box')
        ax3.set_xlim(xlim[0],xlim[1])
        ax3.set_ylim(ylim3[0],ylim3[1])
        ax3.set_ylabel(ylabel3)
        ax3.set_yticks(yticks3)
        ax3.set_xticks(xticks)
        ax3.grid()

        if 'FIR' in Y:
            ax3.scatter(x[self.up_check == 1], y3[self.up_check == 1], color='k',marker=11,s=20, alpha=0.6,rasterized=True)
            ax3.scatter(x[self.up_check == 1], y3[self.up_check == 1], color='k', marker=2,s=20, alpha=0.6)
            ax3.scatter(x[self.up_check == 0], y3[self.up_check == 0], color='gray', marker='P', s=10, rasterized=True)
        else:
            ax3.scatter(x, y3, color='gray', marker='P', s=10,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax3.plot(xmed, y3med, marker='s', color='k', ms=10)
            ax3.errorbar(xmed, y3med, xerr=xstd, yerr=y3std, color='k')
            # if 'FIR' in Y:
                # ax3.plot(xmed, y3med_detect, marker='s', color='k', ms=10, alpha=0.4)
                # ax3.errorbar(xmed, y3med_detect, xerr=xstd, yerr=y3std_detect, color='k', alpha=0.4)
        secax3 = ax3.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        # secax3.set_xlabel(xlabel+r' [L$_{\odot}$]')

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_ratio_1panel(self,savestring,X,Y,median,F1,uv,mir,fir,shape,L=None):
        '''Function to plot the ratio of two luminosites as a function of the denominator'''
        bs1 = shape == 1
        bs2 = shape == 2
        bs3 = shape == 3
        bs4 = shape == 4
        bs5 = shape == 5

        if X == 'Lx':
            x = self.L
            xlabel = r'log L$_{\mathrm{X}}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{X}}$'
            xticks = [43.5,44.5,45.5]
            xlim = [42.75,45.75]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)
 
        elif X == 'Lbol':
            x = L
            xlabel = r'log L$_{\mathrm{bol}}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{bol}}$'
            xticks = [44.5,45.5,46.5]
            xlim = [43.75,46.75]

            bx1 = (x > 44) & (x < 44.5)
            bx2 = (x > 44.5) & (x < 45)
            bx3 = (x > 45) & (x < 45.5)
            bx4 = (x > 45.5) & (x < 46)
            bx5 = (x > 46)
        
        else:
            print('Provide valid X option. Options are:    Lx,    Lbol')
            return

        if Y == 'UV':
            y = np.log10(uv) - x

            ylabel = r'log L (0.25$\mu$m)/'+xvar  
            yticks = [-1,0,1]   
            ylim = [-1, 2]
            
        elif Y == 'MIR':
            y = np.log10(mir) - x

            ylabel = r'log L (6$\mu$m)/'+xvar
            yticks = [-1, 0, 1]
            ylim = [-1, 2]

        elif Y == 'FIR':
            y = np.log10(fir) - x

            ylabel = r'log L (100$\mu$m)/'+xvar  
            yticks = [-1,0,1]   
            ylim = [-1, 2]

        elif Y == 'Lbol':
            if X != 'Lbol':
                y = L - self.L
                ylabel = r'log L$_{\mathrm{bol}}$/'+xvar  
                yticks = [-1,0,1]   
                ylim = [-1, 2]
            else:
                print('X and Y variable cannot be the same. Specify new X or Y variable.')
                return

        elif Y == 'Lbol/Lx':
            y = L - self.L
            ylabel = r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$' 
            yticks = [0, 1, 2, 3]   
            ylim = [0, 2, 3] 

        elif Y == 'Lx/Lbol':
            # if X != 'Lx':
            y = self.L - L
            ylabel = r'log L$_{\mathrm{X}}$/'+xvar  
            yticks = [-1,0,1]   
            ylim = [-1, 2]
            # else:
                # print('X and Y variable cannot be the same. Specify new X or Y variable.')
                # return

        else:
            print('Provide valid Y option. Options are:    UV,    MIR,    FIR,    Lbol,    Lbol/Lx,    Lx')

        # Set median points for X-axis bins
        xmed1, ymed1 = np.nanmean(x[bx1]), np.nanmean(y[bx1])
        xmed2, ymed2 = np.nanmean(x[bx2]), np.nanmean(y[bx2])
        xmed3, ymed3 = np.nanmean(x[bx3]), np.nanmean(y[bx3])
        xmed4, ymed4 = np.nanmean(x[bx4]), np.nanmean(y[bx4])
        xmed5, ymed5 = np.nanmean(x[bx5]), np.nanmean(y[bx5])

        y1std1 = np.std(y[bx1])
        y1std2 = np.std(y[bx2])
        y1std3 = np.std(y[bx3])
        y1std4 = np.std(y[bx4])
        y1std5 = np.std(y[bx5])
  

        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            xmed = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        ymed = np.array([ymed1, ymed2, ymed3, ymed4, ymed5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5])

        durras_K = Lit_functions.Durras_Lbol(np.arange(42,48,0.25),typ='Lbol')
        hopkins_K = Lit_functions.Hopkins_Lbol(np.arange(42,48,0.25),band='Lx')

        # check = Lit_functions.Durras_Lbol(L,typ='Lbol')

        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(111)#, aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0],xlim[1])
        ax1.set_ylim(ylim[0],ylim[1])
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel+xunits)
        ax1.set_yticks(yticks)
        ax1.set_xticks(xticks)
        ax1.grid()
        ax1.plot(np.arange(42,48,0.25),np.log10(durras_K),color='r',label='Duras+2020')
        ax1.plot(np.arange(42,48,0.25),np.log10(hopkins_K),color='b',label='Hopkins+2007')
        # ax1.plot(L,np.log10(check),'.',color='b')
        ax1.scatter(x,y,color='gray',marker='+',s=30,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax1.plot(xmed, ymed, marker='s', color='k',ms=10)
            ax1.errorbar(xmed,ymed,xerr=xstd,yerr=y1std,color='k')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')
        ax1.legend(fontsize=15)

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_scatter_3panels(self,savestring,X,Y,median,F1,uv,mir,fir,shape,L=None):
        if X == 'Lx':
            x = self.L
            x1 = self.L
            x2 = self.L
            x3 = self.L

            xlabel1 = r' '
            xlabel2 = r'log L$_{\mathrm{X}}$'
            xlabel3 = r' '
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{X}}$'
            xticks = [42.5, 43.5, 44.5, 45.5]
            xlim = [42.5, 46]

            b1x1 = (x1 > 43) & (x1 < 43.5)
            b1x2 = (x1 > 43.5) & (x1 < 44)
            b1x3 = (x1 > 44) & (x1 < 44.5)
            b1x4 = (x1 > 44.5) & (x1 < 45)
            b1x5 = (x1 > 45)

            b2x1 = (x2 > 43) & (x2 < 43.5)
            b2x2 = (x2 > 43.5) & (x2 < 44)
            b2x3 = (x2 > 44) & (x2 < 44.5)
            b2x4 = (x2 > 44.5) & (x2 < 45)
            b2x5 = (x2 > 45)

            b3x1 = (x3 > 43) & (x3 < 43.5)
            b3x2 = (x3 > 43.5) & (x3 < 44)
            b3x3 = (x3 > 44) & (x3 < 44.5)
            b3x4 = (x3 > 44.5) & (x3 < 45)
            b3x5 = (x3 > 45)

        elif X == 'Lbol':
            L = np.log10(L)
            x = L
            x1 = L
            x2 = L
            x3 = L

            xlabel = r'log L$_{\mathrm{bol}}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{bol}}$'
            xticks = [44.5, 45.5, 46.5]
            xlim = [43.75, 46.75]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)

        elif X == 'UV-MIR-FIR':
            x1 = np.log10(uv)
            x2 = np.log10(mir)
            x3 = np.log10(fir)

            xlabel1 = (r'log L (0.25$\mu$m)')
            xlabel2 = (r'log L (6$\mu$m)')
            xlabel3 = (r'log L (100$\mu$m)')
            xunits = ' [erg/s]'
            xvar1 = r'L (0.25$\mu$m)'
            xvar2 = r'L (6$\mu$m)'
            xvar3 = r'L (100$\mu$m)'
            xticks = [43,44,45]
            xlim = [42.25 ,45.75]

            b1x1 = (x1 > 42.5) & (x1 < 43)
            b1x2 = (x1 > 43) & (x1 < 43.5)
            b1x3 = (x1 > 43.5) & (x1 < 44)
            b1x4 = (x1 > 44) & (x1 < 44.5)
            b1x5 = (x1 > 44.5) & (x1 < 45)

            b2x1 = (x2 > 43) & (x2 < 43.5)
            b2x2 = (x2 > 43.5) & (x2 < 44)
            b2x3 = (x2 > 44) & (x2 < 44.5)
            b2x4 = (x2 > 44.5) & (x2 < 45)
            b2x5 = (x2 > 45) & (x2 < 45.5)

            b3x1 = (x3 > 43) & (x3 < 43.5)
            b3x2 = (x3 > 43.5) & (x3 < 44)
            b3x3 = (x3 > 44) & (x3 < 44.5)
            b3x4 = (x3 > 44.5) & (x3 < 45)
            b3x5 = (x3 > 45) & (x3 < 45.5)

        if Y == 'Lx':
            y = self.L
            y1 = self.L
            y2 = self.L
            y3 = self.L

            ylabel = r'log L$_{\mathrm{X}}$'
            yunits = ' [erg/s]'
            yvar = r'L$_{\mathrm{X}}$'
            yticks = [42.5 ,43.5, 44.5, 45.5]
            ylim = [42.25, 45.75]

        elif Y == 'UV-MIR-FIR':
            y1 = np.log10(uv)
            y2 = np.log10(mir)
            y3 = np.log10(fir)

            ylabel1 = (r'log L (0.25$\mu$m)')
            ylabel2 = (r'log L (6$\mu$m)')
            ylabel3 = (r'log L (100$\mu$m)')
            ylabel = (r'log L (a $\mu$m)')
            yunits = ' [erg/s]'
            yvar1 = r'L (0.25$\mu$m)'
            yvar2 = r'L (6$\mu$m)'
            yvar3 = r'L (100$\mu$m)'
            yticks = [43, 44, 45]
            ylim = [42.25, 45.75]


        # Set median points for X-axis bins
        y1med1, y2med1, y3med1 = np.nanmean(y1[b1x1]), np.nanmean(y2[b2x1]), np.nanmean(y3[b3x1])
        y1med2, y2med2, y3med2 = np.nanmean(y1[b1x2]), np.nanmean(y2[b2x2]), np.nanmean(y3[b3x2])
        y1med3, y2med3, y3med3 = np.nanmean(y1[b1x3]), np.nanmean(y2[b2x3]), np.nanmean(y3[b3x3])
        y1med4, y2med4, y3med4 = np.nanmean(y1[b1x4]), np.nanmean(y2[b2x4]), np.nanmean(y3[b3x4])
        y1med5, y2med5, y3med5 = np.nanmean(y1[b1x5]), np.nanmean(y2[b2x5]), np.nanmean(y3[b3x5])

        y1std1, y2std1, y3std1 = np.std(y1[b1x1]), np.std(y2[b2x1]), np.std(y3[b3x1]) 
        y1std2, y2std2, y3std2 = np.std(y1[b1x2]), np.std(y2[b2x2]), np.std(y3[b3x2]) 
        y1std3, y2std3, y3std3 = np.std(y1[b1x3]), np.std(y2[b2x3]), np.std(y3[b3x3]) 
        y1std4, y2std4, y3std4 = np.std(y1[b1x4]), np.std(y2[b2x4]), np.std(y3[b3x4]) 
        y1std5, y2std5, y3std5 = np.std(y1[b1x5]), np.std(y2[b2x5]), np.std(y3[b3x5]) 

        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            x1med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        elif X == 'UV-MIR-FIR':
            x1med = np.array([42.75, 43.25, 43.75, 44.25, 44.75])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])

        y1med = np.array([y1med1, y1med2, y1med3, y1med4, y1med5])
        y2med = np.array([y2med1, y2med2, y2med3, y2med4, y2med5])
        y3med = np.array([y3med1, y3med2, y3med3, y3med4, y3med5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5])
        y2std = np.array([y2std1, y2std2, y2std3, y2std4, y2std5])
        y3std = np.array([y3std1, y3std2, y3std3, y3std4, y3std5])

        print(y1std)
        print(y2std)
        print(y3std)
        stern_Lx = Lit_functions.Stern_MIR(np.arange(42, 48, 0.25))

        fig = plt.figure(figsize=(21, 8))
        gs = fig.add_gridspec(ncols=3,nrows=1,left=0.06,right=0.96,top=0.86,bottom=0.14,wspace=0.0)
        ax1 = plt.subplot(gs[0], aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_ylabel(ylabel+yunits)
        ax1.set_xlabel(xlabel1+xunits)
        ax1.set_yticks(yticks) 
        ax1.set_xticks(xticks)
        ax1.grid()

        ax1.scatter(x1,y1,color='gray',marker='+',s=30,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax1.plot(x1med, y1med, marker='s', color='k',ms=10)
            ax1.errorbar(x1med,y1med,xerr=xstd,yerr=y1std,color='k')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel1+r' [L$_{\odot}$]')
        # if Y != 'Nh':
        #     secax1 = ax1.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_yticks([9,10,11,12,13])
        #     secax1.set_ylabel(ylabel+r' [L$_{\odot}$]')

        ax2 = plt.subplot(gs[1], aspect='equal', adjustable='box')
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(ylim[0], ylim[1])
        # ax2.set_ylabel(ylabel+yunits)
        ax2.set_yticklabels([])
        ax2.set_xlabel(xlabel2+xunits)
        ax2.set_yticks(yticks)
        ax2.set_xticks(xticks)
        ax2.grid()

        ax2.plot(np.arange(42,48,0.25),stern_Lx,color='r',label='Stern 2015')
        # ax2.plot(stern_Lx,np.arange(42,48,0.25),color='r',label='Stern 2015')
        ax2.scatter(x2,y2,color='gray',marker='+',s=30,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax2.plot(x2med, y2med, marker='s', color='k',ms=10)
            ax2.errorbar(x2med,y2med,xerr=xstd,yerr=y2std,color='k')
        secax2 = ax2.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax2.set_xticks([9,10,11,12,13])
        secax2.set_xlabel(xlabel2+r' [L$_{\odot}$]')
        # if Y != 'Nh':
        #     secax2 = ax2.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax2.set_yticks([9,10,11,12,13])
        #     secax2.set_ylabel(ylabel+r' [L$_{\odot}$]')
        ax2.legend(fontsize=15)

        ax3 = plt.subplot(gs[2], aspect='equal', adjustable='box')
        ax3.set_xlim(xlim[0], xlim[1])
        ax3.set_ylim(ylim[0], ylim[1])
        # ax3.set_ylabel(ylabel+yunits)
        ax3.set_yticklabels([])
        ax3.set_xlabel(xlabel3+xunits)
        ax3.set_yticks(yticks)
        ax3.set_xticks(xticks)
        ax3.grid()

        if 'FIR' in X:
            ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='k',marker=8,s=35,edgecolors=None, alpha=0.5,rasterized=True)
            ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='gray', marker=1,s=35, alpha=0.75)
            ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='gray', marker='+', s=30, rasterized=True)
        else:
            ax3.scatter(x, y3, color='gray', marker='+', s=30,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax3.plot(x3med, y3med, marker='s', color='k',ms=10)
            ax3.errorbar(x3med,y3med,xerr=xstd,yerr=y3std,color='k')
        secax3 = ax3.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax3.set_xticks([9,10,11,12,13])
        secax3.set_xlabel(xlabel3+r' [L$_{\odot}$]')
        if Y != 'Nh':
            secax3 = ax3.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax3.set_yticks([9,10,11,12,13])
            secax3.set_ylabel(ylabel+r' [L$_{\odot}$]')

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()


    def L_scatter_1panel(self,savestring,X,Y,median,F1,uv,mir,fir,Nh,shape,L=None):
        '''Function to plot the ratio of two luminosites as a function of the denominator'''
        bs1 = shape == 1
        bs2 = shape == 2
        bs3 = shape == 3
        bs4 = shape == 4
        bs5 = shape == 5

        if X == 'Lx':
            x = self.L
            xlabel = r'log L$_{\mathrm{X}}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{X}}$'
            xticks = [43.5,44.5,45.5]
            xlim = [42.75,46.25]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)

        elif X == 'Lbol':
            x = np.log10(L)
            xlabel = r'log L$_{\mathrm{bol}}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\mathrm{bol}}$'
            xticks = [44.5,45.5,46.5]
            xlim = [43.75,46.75]

            bx1 = (x > 43) & (x < 43.5)
            bx2 = (x > 43.5) & (x < 44)
            bx3 = (x > 44) & (x < 44.5)
            bx4 = (x > 44.5) & (x < 45)
            bx5 = (x > 45)
        
        else:
            print('Provide valid X option. Options are:    Lx,    Lbol,    MIR')
            return

        if Y == 'Nh':
            y = np.log10(Nh)
            ylabel = r'log N$_{\mathrm{H}}$'
            yunits = r' [cm$^{-2}$]'
            yticks = [20,21,22,23,24]
            ylim = [19.75,24.25]
        
        elif Y == 'UV':
            y = np.log10(uv)
            ylabel = r'log L (0.25$\mu$m)'
            yunits = ' [erg/s]'
            yticks = [43,44,45,46]   
            ylim = [43, 46.5]
            
        elif Y == 'MIR':
            y = np.log10(mir)
            ylabel = r'log L (6$\mu$m)'
            yunits = ' [erg/s]'
            yticks = [43, 44, 45, 46]
            ylim = [43, 46.5]

        elif Y == 'FIR':
            y = np.log10(fir)
            ylabel = r'log L (100$\mu$m)'  
            yunits = ' [erg/s]'
            yticks = [43,44,45,46]   
            ylim = [43, 46.5]

        elif Y == 'Lbol':
            y = np.log10(L)
            ylabel = r'log L$_{\mathrm{bol}}$' 
            yunits = ' [erg/s]' 
            yticks = [43, 44, 45, 46]
            ylim = [43, 46.5]

        elif Y == 'Lx':
            y = self.L
            ylabel = r'log L$_{\mathrm{X}}$'
            yunits = ' [erg/s]'
            yticks = [41, 42, 43, 44, 45, 46]
            ylim = [41, 46]

        else:
            print('Provide valid Y option. Options are:    UV,    MIR,    FIR,    Lbol,    Lbol/Lx,    Lx')

        y[y == 0.] = np.nan
        # Set median points for X-axis bins
        xmed1, ymed1 = np.nanmean(x[bx1]), np.nanmean(y[bx1])
        xmed2, ymed2 = np.nanmean(x[bx2]), np.nanmean(y[bx2])
        xmed3, ymed3 = np.nanmean(x[bx3]), np.nanmean(y[bx3])
        xmed4, ymed4 = np.nanmean(x[bx4]), np.nanmean(y[bx4])
        xmed5, ymed5 = np.nanmean(x[bx5]), np.nanmean(y[bx5])

        y1std1 = np.std(y[bx1])
        y1std2 = np.std(y[bx2])
        y1std3 = np.std(y[bx3])
        y1std4 = np.std(y[bx4])
        y1std5 = np.std(y[bx5])

        print(min(y),max(y))

        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            xmed = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        else: 
            xmed = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        ymed = np.array([ymed1, ymed2, ymed3, ymed4, ymed5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5])

        stern_Lx = Lit_functions.Stern_MIR(np.arange(42,48))
        durras_Lx = Lit_functions.Durras_Lbol(np.arange(42,48),typ='Lbol')

        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(111, aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0],xlim[1])
        ax1.set_ylim(ylim[0],ylim[1])
        ax1.set_ylabel(ylabel+yunits)
        ax1.set_xlabel(xlabel+xunits)
        ax1.set_yticks(yticks)
        ax1.set_xticks(xticks)
        ax1.grid()
        # ax1.plot(np.arange(42,48),np.log10(durras_Lx),color='r')
        ax1.plot(np.arange(42,48),stern_Lx,color='r')
        ax1.scatter(x,y,color='gray',marker='P',s=10,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax1.plot(xmed, ymed, marker='s', color='k',ms=10)
            ax1.errorbar(xmed,ymed,xerr=xstd,yerr=y1std,color='k')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')
        if Y != 'Nh':
            secax1 = ax1.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax1.set_yticks([9,10,11,12,13])
            secax1.set_ylabel(ylabel+r' [L$_{\odot}$]')

        plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_hist(self,savestring,x,xlabel=None,xlim=[np.nan,np.nan],bins=[np.nan,np.nan,np.nan],median=True,std=False):
        #75bbfb
        plt.figure(figsize=(9, 9))
        # ax1 = plt.subplot(111, aspect='equal', adjustable='box')
        n = plt.hist(x,bins=np.arange(bins[0],bins[1],bins[2]),color='#75bbfb',zorder=0)
        if median:
            plt.axvline(np.nanmedian(x),color='k',ls='--',lw=3)
            if std:
                plt.axvline(np.std(x)+np.nanmean(x),color='k',lw=1)
                plt.axvline(np.nanmean(x)-np.std(x),color='k',lw=1)
                plt.fill_between([np.nanmean(x)-np.std(x), np.std(x)+np.nanmean(x)],[0,0],[max(n[0])+max(n[0])*0.1,max(n[0])+max(n[0])*0.1],color='gray',alpha=0.4)
        plt.xlabel(xlabel)
        plt.grid()
        plt.ylim(0,max(n[0])+max(n[0])*0.1)
        
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

        print('mean: ', np.nanmean(x))
        print('std: ', np.std(x))

    def L_hist_zbins(self,savestring,x,xlabel=None,xlim=[np.nan,np.nan],bins=[np.nan,np.nan,np.nan],median=True,std=False):
        b1 = self.z < 0.6
        b2 = (self.z > 0.6) & (self.z < 0.9)
        b3 = self.z > 0.9
        
        plt.figure(figsize=(9,9))
        n1 = plt.hist(x[b1], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color='#1E62E5',lw=4,alpha=0.8,label= 'z < 0.6')
        n2 = plt.hist(x[b2], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color='#04BF0C',lw=4,alpha=0.8,label= '0.6 < z < 0.9')
        n3 = plt.hist(x[b3], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color='#D23737',lw=4,alpha=0.8,label= '0.9 < z < 1.2')
        n = np.append(n1[0],n2[0])
        n = np.append(n,n3[0])
        if median:
            plt.axvline(np.nanmean(x[b1]), color='#1E62E5', ls='--', lw=3)
            plt.axvline(np.nanmean(x[b2]), color='#04BF0C', ls='--', lw=3)
            plt.axvline(np.nanmean(x[b3]), color='#D23737', ls='--', lw=3)
        plt.xlabel(xlabel)
        plt.grid()
        plt.ylim(0,max(n)+max(n)*0.1)
        plt.legend()

        print('bin 1: ',np.nanmean(x[b1]))
        print('bin 2: ',np.nanmean(x[b2]))
        print('bin 3: ',np.nanmean(x[b3]))

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()


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