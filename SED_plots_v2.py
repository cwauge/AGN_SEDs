from ast import arg
from stringprep import map_table_b2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import argparse
import Lit_functions
from matplotlib.collections import LineCollection
from scipy import interpolate
from plot_fit import plot_fit
from sample_change import remove_outliers



def main(ID, z, wavelength, Lum, L):
    plot = Plotter(ID, z, wavelength, Lum, L)

class Plotter():

    def __init__(self,ID,z,wavelength,Lum,L,norm,up_check):
        self.ID = np.asarray(ID)
        self.z = np.around(z,2)
        self.wavelength = np.asarray(wavelength)
        self.Lum = np.asarray(Lum)
        self.L = np.asarray(L)
        self.norm = np.asarray(norm)
        self.up_check = np.asarray(up_check)

        plt.rcParams['font.size'] = 20
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

    def median_sed(self,x_in,y_in,Norm=True,connect_point=False,Bin=False,bin_in=None,color='k',lw=6, label=None,scale=False,scale_F=None):
        '''Function to generate the median line for array of SEDs to be plotted'''
        x_out = np.nanmedian(x_in, axis=0)
        y_out = 10**np.nanmedian(y_in,axis=0)
        if Norm:
            if Bin:
                if scale:
                    print('y 1: ', len(y_out))
                    y_out /=np.nanmedian(self.norm[bin_in])/scale_F
                    print('y 2: ',len(y_out))
                else:
                    y_out /= np.nanmedian(self.norm[bin_in])
            else:
                y_out /= np.nanmedian(self.norm)

        if label is None:
            plt.plot(x_out,y_out,lw=lw,color=color)
        else:
            plt.plot(x_out,y_out,lw=lw,color=color,label=label)
        if connect_point:
            return x_out[-1], y_out[-1]

    def median_FIR_sed(self,xfir,yfir,Norm=True,connect=[np.nan,np.nan],upper='upper lims',Bin=False,bin_in=None,color='k',lw=6,ls='-',line=True,ms=10,scale=False,scale_F=None):
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
        if scale:    
            y_out = np.nanmean(yfir,axis=0)*scale_F
        else:
            y_out = np.nanmean(yfir, axis=0)

        if Norm: # If Norm is True, normalize FIR SED. Default is to normalize
            if Bin:
                    y_out /= np.nanmedian(self.norm[bin_in])
            else:
                y_out /= np.nanmedian(self.norm)
        if ~np.isnan(connect[0]):
            x_out[0] = connect[0]
            y_out[0] = connect[1]
        if line:
            # print('x: ',x_out)
            # print('y: ',y_out)
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
   
    def PlotSED(self,point_x=np.nan,point_y=np.nan,fir_x=[np.nan],fir_y=[np.nan],temp_x=[np.nan],temp_y=[np.nan],xsub=[np.nan],ysub=[np.nan],save=False):
        name = self.ID
        name = name.tolist()
        name = name.replace('/','-')
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(self.wavelength,self.Lum,color='b',lw=2.5)

        ax.plot(self.wavelength,self.Lum,'o',c='k',ms=5)
        ax.plot(point_x,point_y,'x',c='r')
        # ax.plot(self.wfir,self.ffir,c='gray',lw=4)
        if any(fir_y) != np.nan:
            ax.plot(fir_x, fir_y/self.norm, color='gray',lw=4)
            ax.plot(fir_x, fir_y, 'o',color='b')
        ax.plot(temp_x,temp_y/self.norm,color='orange')
        ax.plot(xsub,ysub/self.norm,color='black')
        # print(xsub)
        # print(ysub/self.norm)

        ax.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)
        ax.set_ylabel(r'$\lambda$L$_\lambda$',fontsize=22)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_xlim(5E-5,7E2)
        ax.set_ylim(1E-4,1E2)
        # ax.set_title(self.ID)
        ax.text(0.05,0.8,r'log $L_{\rm X}$ = '+str(round(np.log10(self.L),2)),transform=ax.transAxes,fontsize=20)
        ax.text(0.05,0.7,f'z = {self.z}',transform=ax.transAxes,fontsize=20)
        plt.xlim(5E-5, 7E2)
        # plt.ylim(1E-4, 1E2)
        # plt.ylim(1E42,1E46)
        plt.grid()
        plt.title(self.ID)
        if save:
            plt.savefig(f'/Users/connor_auge/Desktop/test_goals_data/{name}_SED_new.pdf')
        plt.show()

    def Plot_FIR_SED(self,wfir=[np.nan],ffir=[np.nan]):
        self.wfir = wfir
        self.ffir = ffir

    def multi_SED(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]],opt_p=[np.nan,np.nan],Median_line=True,FIR_med=True,FIR_upper='upper lims',percent=False,GOALS=False,wave_labels=False,temp_comp=False,temp_comp_x=[np.nan],temp_comp_y=[np.nan]):
        '''Function to overplot all normalized SEDs with each line mapping to a colorbar'''
        # Set colorbar limits
        print(self.L)
        if GOALS:
            clim1 = 42.8
            clim2 = 44.25
            fir_ls = '-'
        else:
            clim1 = 45.2 + np.log10(12)
            clim2 = 46 + np.log10(12)
            fir_ls = '--'
        cmap = 'rainbow_r' # set colormap
        # cmap = 'YlGnBu' # set colormap
        # cmap = 'winter' # set colormap

        # self.L += np.log10(12)


        x = self.wavelength[self.L >= clim1] # remove sources with L outside colorbar range
        y = self.Lum[self.L >= clim1] 
        L = self.L[self.L >= clim1]
        z = self.z[self.L >= clim1]

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
        # ax.set_aspect(1)
        ax.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=26)
        ax.set_ylabel(r'Normalized $\lambda$ L$_\lambda$',fontsize=26)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticklabels([r'10$^{-4}$',r'10$^{-3}$','0.01','0.1','1.0','10','100'])
        ax.set_xticks([1E-4,1E-3,1E-2,1E-1,1E0,1E1,1E2])
        ax.text(0.15, 0.85, f'n = {len(L)}', transform=ax.transAxes)
        # ax.text(0.15, 0.824, f'S82X n = 529', transform=ax.transAxes)
        # ax.text(0.15, 0.798, f'COSMOS n = 624', transform=ax.transAxes)
        # ax.text(0.15, 0.772, f'GOODS n = 93', transform=ax.transAxes)



        if wave_labels:
            font = matplotlib.font_manager.FontProperties()
            font.set_weight('bold')
            ax.text(1.E-4, 1.15E2, 'X-ray', fontproperties=font)
            ax.text(0.065,1.15E2,'UV', fontproperties=font)
            ax.text(0.2,1.15E2,'Optical', fontproperties=font)
            ax.text(3.5,1.15E2,'MIR', fontproperties=font)
            ax.text(65, 1.15E2, 'FIR', fontproperties=font)
        
        # Plot the FIR upper limit segments 
        upper_seg = np.stack((wfir_seg, ffir_seg), axis=2)
        # upper_all = LineCollection(upper_seg,color='gray',alpha=0.3) 
        # if ~GOALS:
            # ax.add_collection(upper_all)

        # use multilines function to plot all SEDs mapped to colorbar based on L
        lc = self.multilines(x, y, L, lw=2.5, cmap=cmap, alpha=0.5, rasterized=True) 
        axcb1 = fig.colorbar(lc, fraction=0.046, pad=0.04)  # make colorbar
        axcb1.mappable.set_clim(clim1,clim2) # initialize colorbar limits
        # axcb1.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]',fontsize=22)
        axcb1.set_label(r'log L$_{\rm bol}$ [erg s$^{-1}$]',fontsize=24)

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
                self.median_FIR_sed(wfir,ffir,connect=[x_connect,y_connect],upper=FIR_upper,ls=fir_ls)
            else:
                self.median_sed(median_x,median_y)
        if percent:
            x_connect_p, y_connect_p1, y_connect_p2 = self.percentile_lines(median_x, median_y,connect_point=True,fill=True)
            self.percentile_lines_FIR(wfir,ffir, connect=[x_connect_p,y_connect_p1, y_connect_p2], upper=FIR_upper, fill=True)
            self.percentile_lines(x[:,:2],np.log10(y[:,:2]),Norm=False,fill=True)

        if temp_comp:
            ax.plot(temp_comp_x,temp_comp_y,'--',color='r',lw=6)

        # plt.axvline(x=6,color='k',ls='--',lw=2.8)
        # plt.axvline(x=5.1660E-5,color='k',ls='--',lw=2.8)

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

        # plt.axvline(6.0,color='k')
        plt.ylim(1E-3,200)
        plt.xlim(2E-5,550)
        plt.grid()
        plt.tight_layout()
        
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def multi_SED_bins(self, savestring, bin, field, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], opt_p=[np.nan, np.nan], Median_line=True, FIR_med=True, FIR_upper='upper lims',scale=False):
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

        elif bin == 'Lx':
            b1 = self.L < 43.75
            b2 = (self.L > 43.75) & (self.L < 44.5)
            b3 = self.L > 44.5
            t1 = r'43 < log L$_{\rm X}$ < 43.75'
            t2 = r'43.75 < log L$_{\rm X}$ < 44.5'
            t3 = r'44.5 < log L$_{\rm X}$'
            z1 = self.z[b1]
            z2 = self.z[b2]
            z3 = self.z[b3]

        else:
            print('Specify bins. Options are: redshift,    field,    Lx')
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

        if scale:
            b1_scale = np.nanmedian(self.norm[b1],axis=0)/np.nanmedian(self.norm[b2],axis=0)
            b3_scale = np.nanmedian(self.norm[b3],axis=0)/np.nanmedian(self.norm[b2],axis=0)
        else:
            b1_scale = 1.0
            b3_scale = 1.0

        print('scale: ', b1_scale, b3_scale)


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
        gs = fig.add_gridspec(nrows=1,ncols=3,bottom=0.1,top=0.9,left=0.1,right=0.9,wspace=0.05)

        ax1 = fig.add_subplot(gs[0], aspect='equal', adjustable='box')
        # ax1.set_box_aspect(1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
        if bin == 'redshift':
            ax1.text(0.05,0.72,r'$L_{\rm X, med}$ = '+str(round(np.nanmedian(L1),2)),transform=ax1.transAxes)
        if bin == 'Lx':
            ax1.text(0.05,0.72,r'$z_{\rm med}$ = '+str(round(np.nanmedian(z1),2)),transform=ax1.transAxes)
        ax1.set_title(t1)
        ax1.grid()
        ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')

        # Plot data
        upper_seg1 = np.stack((wfir1_seg,ffir1_seg*b1_scale), axis=2)
        upper_all1 = LineCollection(upper_seg1, color='gray', alpha=0.3)
        ax1.add_collection(upper_all1)
        lc1 = self.multilines(x1,y1*b1_scale,L1,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb1 = fig.colorbar(lc1)
        axcb1.mappable.set_clim(clim1, clim2)
        axcb1.remove()
        # Plot median line
        if Median_line:
            ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0)*b1_scale, c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x1, median_y1, connect_point=True, Bin=True, bin_in=b1, lw=3, scale=True, scale_F=b1_scale)
                self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b1, lw=3, ls='--',scale=True,scale_F=b1_scale)
            else:
                self.median_sed(median_x1, median_y1,Bin=True, bin_in = b1, lw=3, scale=True, scale_F=b1_scale)
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
        if bin == 'redshift':
            ax2.text(0.05,0.72,r'$L_{\rm X, med}$ = '+str(round(np.nanmedian(L2),2)),transform=ax2.transAxes)
        if bin == 'Lx':
            ax2.text(0.05,0.72,r'$z_{\rm med}$ = '+str(round(np.nanmedian(z2),2)),transform=ax2.transAxes)
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
        if bin == 'redshift':
            ax3.text(0.05,0.72,r'$L_{\rm X, med}$ = '+str(round(np.nanmedian(L3),2)),transform=ax3.transAxes)
        if bin == 'Lx':
            ax3.text(0.05,0.72,r'$z_{\rm med}$ = '+str(round(np.nanmedian(z3),2)),transform=ax3.transAxes) 
        ax3.grid()
        ax3.set_title(t3)

        upper_seg3 = np.stack((wfir3_seg,ffir3_seg*b3_scale), axis=2)
        upper_all3 = LineCollection(upper_seg3, color='gray', alpha=0.3)
        ax3.add_collection(upper_all3)
        lc3 = self.multilines(x3,y3*b3_scale,L3,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb3 = fig.colorbar(lc3)
        axcb3.mappable.set_clim(clim1, clim2)
        axcb3.remove()
        # Plot median line
        if Median_line:
            ax3.plot(np.nanmedian(x3[:,:2],axis=0),np.nanmedian(y3[:,:2],axis=0)*b3_scale,c='k',lw=3)
            if FIR_med:
                if bin=='redshift' or 'Lx':
                    x_connect, y_connect = self.median_sed(median_x3, median_y3, connect_point=True, Bin=True, bin_in=b3,lw=3,scale=True,scale_F=b3_scale)
                    self.median_FIR_sed(wfir3,ffir3,connect=[x_connect,y_connect],upper=FIR_upper, Norm=False,Bin=True, bin_in=b3,lw=3, ls='--',scale=True,scale_F=b3_scale)
                else:
                    self.median_sed(median_x3, median_y3, Bin=True, bin_in=b3,lw=3,scale=True,scale_F=b3_scale)
            else:
                self.median_sed(median_x3,median_y3, Bin=True, bin_in=b3,lw=3,scale=True,scale_F=b3_scale)
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

    def median_SED_1panel(self, savestring, median_x, median_y, wfir, ffir, shape, FIR_upper='upper lims',ls='-',bins='shape',compare=False,comp_med_x=None,comp_med_y=None,comp_wfir=None,comp_ffir=None,plot_temp=False,temp_x=[np.nan],temp_y=[np.nan]):
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

        if bins == 'shape':
            b1 = shape == 1
            b2 = shape == 2
            b3 = shape == 3
            b4 = shape == 4
            b5 = shape == 5

            bin1_name = 'Panel 1'
            bin2_name = 'Panel 2'
            bin3_name = 'Panel 3'
            bin4_name = 'Panel 4'
            bin5_name = 'Panel 5'

            c1 = '#377eb8'
            c2 = '#984ea3'
            c3 = '#4daf4a'
            c4 = '#ff7f00'
            c5 = '#e41a1c'

        elif bins == 'Lx_5':
            b5 = self.L < 43.5
            b4 = (self.L > 43.5) & (self.L < 44)
            b3 = (self.L > 44) & (self.L < 44.5)
            b2 = (self.L > 44.5) & (self.L < 45)
            b1 = self.L > 45

            bin1_name = r'log L$_{\rm X}$ > 45'
            bin2_name = r'44.5 < log L$_{\rm X}$ < 45'
            bin3_name = r'44 < log L$_{\rm X}$ < 44.5'
            bin4_name = r'43.5 < log L$_{\rm X}$ < 44'
            bin5_name = r'log L$_{\rm X}$ < 43.5'

            c1 = '#377eb8'
            c2 = '#984ea3'
            c3 = '#4daf4a'
            c4 = '#ff7f00'
            c5 = '#e41a1c'

        elif bins == 'Lx_3':
            b1 = self.L < 43.75
            b2 = (self.L > 43.75) & (self.L < 44.5)
            b3 = self.L > 44.5
            b4 = self.L < 0
            b5 = self.L < 0

            bin1_name = r'43 < log L$_{\rm X}$ < 43.75'
            bin2_name = r'43.75 < L$_{\rm X}$ < 44.5'
            bin3_name = r'44.5 < L$_{\rm X}$ < 45.5'
            # bin4_name = r' '
            # bin5_name = r' '

            c1 = 'blue'
            c2 = 'green'
            c3 = 'red'


        else:
            print('Invalid bins option. Options are:   shape,    Lx_5,    Lx_3')
            return
        
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

        # x1 = x[b1]
        # x2 = x[b2]
        # x3 = x[b3]
        # x4 = x[b4]
        # x5 = x[b5]

        # y1 = y[b1]
        # y2 = y[b2]
        # y3 = y[b3]
        # y4 = y[b4]
        # y5 = y[b5]

        # norm1 = self.norm[b1]
        # norm2 = self.norm[b2]
        # norm3 = self.norm[b3]
        # norm4 = self.norm[b4]
        # norm5 = self.norm[b5] 

        # yticks = [42, 43, 44, 45, 46]
        # # xticks = [1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1, 1E2]
        # xticks = [1E-1, 1E0, 1E1, 1E2]
        # ytick_labels = ['42', '43', '44', '45', '46']
        # # xticks_labels = [r'10$^{-4}$', '', r'10$^{-2}$', '', r'10$^{0}$', '', r'10$^{2}$']
        # xticks_labels = [r'10$^{-1}$', r'10$^{0}$', r'10$^{1}$', r'10$^{2}$']

        fig = plt.figure(figsize=(18,12))
        gs = fig.add_gridspec(nrows=1, ncols=1)
        # gs.update(wspace=0.08) # set the spacing between axes
        # gs.update(left=0.125,right=0.95,top=0.9,bottom=0.1)

        # median_y1[-1] = median_y1[-1]*0.9

        ax1 = plt.subplot(gs[0])#, aspect='equal', adjustable='box')

        if plot_temp:
            print(temp_y)
            ax1.plot(temp_x,temp_y,color='gray',alpha=0.7,lw=5,label='Assef et al. 2010 AGN')

        if bins == 'Lx_3':
            x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=False,connect_point=True,color=c3,lw=4,label=bin1_name)
            self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')
            x_connect, y_connect = self.median_sed(median_x2, median_y2, Norm=False,connect_point=True,color=c2,lw=4,label=bin2_name)
            self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')
            x_connect, y_connect = self.median_sed(median_x3, median_y3, Norm=False,connect_point=True,color=c1,lw=4,label=bin3_name)
            self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')

        else:
            x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=False,connect_point=True,color=c1,lw=4,label=bin1_name)
            self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c1,lw=4,ls='--')

            x_connect, y_connect = self.median_sed(median_x2, median_y2, Norm=False,connect_point=True,color=c2,lw=4,label=bin2_name)
            self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c2,lw=4,ls='--')

            x_connect, y_connect = self.median_sed(median_x3, median_y3, Norm=False,connect_point=True,color=c3,lw=4,label=bin3_name)
            self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c3,lw=4,ls='--')

            x_connect, y_connect = self.median_sed(median_x4, median_y4, Norm=False,connect_point=True,color=c4,lw=4,label=bin4_name)
            self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c4,lw=4,ls='--')

            x_connect, y_connect = self.median_sed(median_x5, median_y5, Norm=False,connect_point=True,color=c5,lw=4,label=bin5_name)
            self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color=c5,lw=4,ls='--')

            if compare:
                print(comp_med_x)
                print(comp_med_y)
                x_connect, y_connect = self.median_sed(comp_med_x, comp_med_y, Norm=False,connect_point=True,color='gray',lw=4,label='GOALS')
                self.median_FIR_sed(comp_wfir, comp_ffir, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,color='gray',lw=4,ls='-')
 


        ax1.set_xscale('log')
        ax1.set_yscale('log')
        # ax1.set_ylim(3E41, 3E46)
        # ax1.set_ylim(1E42,1E46)
        # ax1.set_xlim(5E-2, 450)
        # ax1.set_xticks(xticks)
        # ax1.set_xticklabels(xticks_labels)
        ax1.set_ylabel(r'$\lambda$ L$_\lambda$ [erg/s]')
        ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')
        secax1 = ax1.secondary_yaxis('right', functions=(self.solar, self.ergs))
        secax1.set_yticks([9, 10, 11, 12, 13])
        secax1.set_ylabel(r'$\lambda$ L$_\lambda$ [L$_{\odot}$]')
        ax1.grid()
        ax1.legend(loc='lower right',fontsize=21)

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

    def L_ratio_1panel(self,savestring,X,Y,median,F1,uv,mir,fir,shape,L=None,compare=False,comp_x=None,comp_y=None,sample=False,spec_type=[np.nan],med2x=[np.nan],med2y=[np.nan],med2xerr=[np.nan],med2yerr=[np.nan]):
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
            xlabel = r'log L$_{\mathrm{bol - gal,e}}$'
            xunits = r'/$(\rm{erg \; s^{-1}})$'
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
            # y = L - (self.L+np.log10(0.611))
            y = L - self.L
            ylabel = r'log L$_{\mathrm{bol-gal,e}}$/L$_{\mathrm{X}}$' 
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

        # xmed1, ymed1 = np.nanmedian(x[bx1]), np.nanmedian(y[bx1])
        # xmed2, ymed2 = np.nanmedian(x[bx2]), np.nanmedian(y[bx2])
        # xmed3, ymed3 = np.nanmedian(x[bx3]), np.nanmedian(y[bx3])
        # xmed4, ymed4 = np.nanmedian(x[bx4]), np.nanmedian(y[bx4])
        # xmed5, ymed5 = np.nanmedian(x[bx5]), np.nanmedian(y[bx5])

        y1std1 = np.std(y[bx1])
        y1std2 = np.std(y[bx2])
        y1std3 = np.std(y[bx3])
        y1std4 = np.std(y[bx4])
        y1std5 = np.std(y[bx5])

        # y_per25_1 = ymed1 - np.nanpercentile(y[bx1],25) 
        # y_per25_2 = ymed2 - np.nanpercentile(y[bx2],25) 
        # y_per25_3 = ymed3 - np.nanpercentile(y[bx3],25) 
        # y_per25_4 = ymed4 - np.nanpercentile(y[bx4],25) 
        # y_per25_5 = ymed5 - np.nanpercentile(y[bx5],25) 

        # y_per75_1 = np.nanpercentile(y[bx1],75) - ymed1 
        # y_per75_2 = np.nanpercentile(y[bx2],75) - ymed2 
        # y_per75_3 = np.nanpercentile(y[bx3],75) - ymed3 
        # y_per75_4 = np.nanpercentile(y[bx4],75) - ymed4 
        # y_per75_5 = np.nanpercentile(y[bx5],75) - ymed5   

        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            xmed = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        ymed = np.array([ymed1, ymed2, ymed3, ymed4, ymed5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5])
        # y1std = np.array([[y_per25_1,y_per75_1], [y_per25_2,y_per75_2], [y_per25_3,y_per75_3], [y_per25_4,y_per75_4], [y_per25_5,y_per75_5]]).T

        durras_K = Lit_functions.Durras_Lbol(np.arange(42,48,0.25),typ='Lbol')
        hopkins_K = Lit_functions.Hopkins_Lbol(np.arange(42,48,0.25),band='Lx')

        # check = Lit_functions.Durras_Lbol(L,typ='Lbol')
        print('x med: ',xmed)
        print('y med: ',ymed)
        print('y std: ', y1std)
        print('Duras: ',np.log10(Lit_functions.Durras_Lbol(xmed,typ='Lbol')))

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
        if sample:
            ax1.scatter(x[spec_type==1],y[spec_type==1],color='orange',marker='+',s=30,rasterized=True,label='Type 1')
            ax1.scatter(x[spec_type==2],y[spec_type==2],color='green',marker='+',s=30,rasterized=True,label='Type 2') 
        else:
            ax1.scatter(x,y,color='gray',marker='+',s=30,rasterized=True)
        if median == 'X-axis' or median == 'Both':
            ax1.plot(xmed, ymed, marker='s', color='k', ms=12, linestyle='')
            ax1.errorbar(xmed, ymed, xerr=xstd, yerr=y1std, color='k', linestyle='')

            ax1.plot(med2x, med2y, marker='o', color='k', ms=12, linestyle='',alpha=0.5)
            ax1.errorbar(med2x, med2y, xerr=med2xerr, yerr=med2yerr, color='k', linestyle='',alpha=0.5)

            # plot_fit(xmed,ymed,2,44,46.5)

        if compare:
            ax1.scatter(comp_x,comp_y,marker='X',color='r',s=100,label='GOALS')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel+r'/L$_{\odot}$')
        ax1.legend(fontsize=15)

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Research/Disertation/AGN_SEDs/AHA_SED_paper1/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_scatter_3panels(self, savestring, X, Y, median, F1, uv, mir, fir, shape, L=None, uv_err=None, mir_err=None, fir_err=None, error=False, compare=False, comp_L=None, comp_uv=None, comp_mir=None, comp_fir=None, stack_color=False, stack_bins=None, F100_ratio=None, field=None, fir_field=False):
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
            xticks = [43,43.5,44,44.5,45,45.5]
            # xlim = [42.5, 46]
            xlim = [42.5,46.25]

            b1x1 = (x1 > 43) & (x1 < 43.5)
            b1x2 = (x1 > 43.5) & (x1 < 44)
            b1x3 = (x1 > 44) & (x1 < 44.5)
            b1x4 = (x1 > 44.5) & (x1 < 45)
            b1x5 = (x1 > 45)
            b1x6 = (x1 > 100)

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
            x1err = uv_err
            x2err = mir_err
            x3err = fir_err
            xF1 = np.log10(F100_ratio)

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
            b1x6 = (x1 > 45) & (x1 <45.5)

            b2x1 = (x2 > 43) & (x2 < 43.5)
            b2x2 = (x2 > 43.5) & (x2 < 44)
            b2x3 = (x2 > 44) & (x2 < 44.5)
            b2x4 = (x2 > 44.5) & (x2 < 45)
            b2x5 = (x2 > 45) & (x2 < 45.5)

            # b2x1 = (x2 > 42.5) & (x2 < 43)
            # b2x2 = (x1 > 43) & (x2 < 43.5)
            # b2x3 = (x2 > 43.5) & (x2 < 44)
            # b2x4 = (x2 > 44) & (x2 < 44.5)
            # b2x5 = (x2 > 44.5) & (x2 < 45)

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
            y1err = uv_err
            y2err = mir_err
            y3err = fir_err

            ylabel1 = (r'log L (0.25$\mu$m)')
            ylabel2 = (r'log L (6$\mu$m)')
            ylabel3 = (r'log L (100$\mu$m)')
            ylabel = (r'log L (a $\mu$m)')
            yunits = ' [erg/s]'
            yvar1 = r'L (0.25$\mu$m)'
            yvar2 = r'L (6$\mu$m)'
            yvar3 = r'L (100$\mu$m)'
            yticks = [42,43,44,45,46,47]
            # ylim = [42.25, 45.75]
            ylim = [41,47]

        # Set median points for X-axis bins
        y1med1, y2med1, y3med1 = np.nanmedian(y1[b1x1]), np.nanmedian(y2[b2x1]), np.nanmedian(y3[b3x1])
        y1med2, y2med2, y3med2 = np.nanmedian(y1[b1x2]), np.nanmedian(y2[b2x2]), np.nanmedian(y3[b3x2])
        y1med3, y2med3, y3med3 = np.nanmedian(y1[b1x3]), np.nanmedian(y2[b2x3]), np.nanmedian(y3[b3x3])
        y1med4, y2med4, y3med4 = np.nanmedian(y1[b1x4]), np.nanmedian(y2[b2x4]), np.nanmedian(y3[b3x4])
        y1med5, y2med5, y3med5 = np.nanmedian(y1[b1x5]), np.nanmedian(y2[b2x5]), np.nanmedian(y3[b3x5])
        y1med6 = np.nanmedian(y1[b1x6])


        # y1med1, y2med1, y3med1 = np.nanmedian(y1[b1x1][y1[b1x1] > 42]), np.nanmedian(y2[b2x1][y2[b2x1] > 42]), np.nanmedian(y3[b3x1][y3[b3x1] > 43])
        # y1med2, y2med2, y3med2 = np.nanmedian(y1[b1x2][y1[b1x2] > 43]), np.nanmedian(y2[b2x2][y2[b2x2] > 43]), np.nanmedian(y3[b3x2][y3[b3x2] > 43])
        # y1med3, y2med3, y3med3 = np.nanmedian(y1[b1x3][y1[b1x3] > 43]), np.nanmedian(y2[b2x3][y2[b2x3] > 43]), np.nanmedian(y3[b3x3][y3[b3x3] > 43])
        # y1med4, y2med4, y3med4 = np.nanmedian(y1[b1x4][y1[b1x4] > 43]), np.nanmedian(y2[b2x4][y2[b2x4] > 43]), np.nanmedian(y3[b3x4][y3[b3x4] > 43])
        # y1med5, y2med5, y3med5 = np.nanmedian(y1[b1x5][y1[b1x5] > 43]), np.nanmedian(y2[b2x5][y2[b2x5] > 43]), np.nanmedian(y3[b3x5][y3[b3x5] > 43])
        # y1med6 = np.nanmedian(y1[b1x6])

        # y1std1, y2std1, y3std1 = np.std(y1[b1x1]), np.std(y2[b2x1]), np.std(y3[b3x1]) 
        # y1std2, y2std2, y3std2 = np.std(y1[b1x2]), np.std(y2[b2x2]), np.std(y3[b3x2]) 
        # y1std3, y2std3, y3std3 = np.std(y1[b1x3]), np.std(y2[b2x3]), np.std(y3[b3x3]) 
        # y1std4, y2std4, y3std4 = np.std(y1[b1x4]), np.std(y2[b2x4]), np.std(y3[b3x4]) 
        # y1std5, y2std5, y3std5 = np.std(y1[b1x5]), np.std(y2[b2x5]), np.std(y3[b3x5]) 
        # y1std6 = np.std(y1[b1x6])

        y1per25_1, y2per25_1, y3per25_1 = y1med1 - np.nanpercentile(y1[b1x1],25), y2med1 - np.nanpercentile(y2[b2x1],25), y3med1 - np.nanpercentile(y3[b3x1],25) 
        y1per25_2, y2per25_2, y3per25_2 = y1med2 - np.nanpercentile(y1[b1x2],25), y2med2 - np.nanpercentile(y2[b2x2],25), y3med2 - np.nanpercentile(y3[b3x2],25) 
        y1per25_3, y2per25_3, y3per25_3 = y1med3 - np.nanpercentile(y1[b1x3],25), y2med3 - np.nanpercentile(y2[b2x3],25), y3med3 - np.nanpercentile(y3[b3x3],25) 
        y1per25_4, y2per25_4, y3per25_4 = y1med4 - np.nanpercentile(y1[b1x4],25), y2med4 - np.nanpercentile(y2[b2x4],25), y3med4 - np.nanpercentile(y3[b3x4],25) 
        y1per25_5, y2per25_5, y3per25_5 = y1med5 - np.nanpercentile(y1[b1x5],25), y2med5 - np.nanpercentile(y2[b2x5],25), y3med5 - np.nanpercentile(y3[b3x5],25) 
        y1per25_6 = np.nanpercentile(y1[b1x6],25)

        y1per75_1, y2per75_1, y3per75_1 = np.nanpercentile(y1[b1x1],75) - y1med1, np.nanpercentile(y2[b2x1],75) - y2med1, np.nanpercentile(y3[b3x1],75) - y3med1 
        y1per75_2, y2per75_2, y3per75_2 = np.nanpercentile(y1[b1x2],75) - y1med2, np.nanpercentile(y2[b2x2],75) - y2med2, np.nanpercentile(y3[b3x2],75) - y3med2 
        y1per75_3, y2per75_3, y3per75_3 = np.nanpercentile(y1[b1x3],75) - y1med3, np.nanpercentile(y2[b2x3],75) - y2med3, np.nanpercentile(y3[b3x3],75) - y3med3 
        y1per75_4, y2per75_4, y3per75_4 = np.nanpercentile(y1[b1x4],75) - y1med4, np.nanpercentile(y2[b2x4],75) - y2med4, np.nanpercentile(y3[b3x4],75) - y3med4 
        y1per75_5, y2per75_5, y3per75_5 = np.nanpercentile(y1[b1x5],75) - y1med5, np.nanpercentile(y2[b2x5],75) - y2med5, np.nanpercentile(y3[b3x5],75) - y3med5 
        y1per75_6 = np.nanpercentile(y1[b1x6],75)

        x1_good, y1_good = remove_outliers(x1,y1,[43,47])
        x2_good, y2_good = remove_outliers(x2,y2,[43,46])
        x3_good, y3_good = remove_outliers(x3,y3,[43,46])


        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            x1med = np.array([43.25, 43.75, 44.25, 44.75, 45.25, np.nan])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        elif X == 'UV-MIR-FIR':
            x1med = np.array([42.75, 43.25, 43.75, 44.25, 44.75, 45.25])
            # x2med = np.array([42.75, 43.25, 43.75, 44.25, 44.75])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])

        y1med = np.array([y1med1, y1med2, y1med3, y1med4, y1med5, y1med6])
        y2med = np.array([y2med1, y2med2, y2med3, y2med4, y2med5])
        y3med = np.array([y3med1, y3med2, y3med3, y3med4, y3med5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        x1std = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        # y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5, y1std6])
        # y2std = np.array([y2std1, y2std2, y2std3, y2std4, y2std5])
        # y3std = np.array([y3std1, y3std2, y3std3, y3std4, y3std5])

        y1std = np.array([[y1per25_1,y1per75_1], [y1per25_2,y1per75_2], [y1per25_3,y1per75_3], [y1per25_4,y1per75_4], [y1per25_5,y1per75_5], [y1per25_6,y1per75_6]]).T
        y2std = np.array([[y2per25_1,y2per75_1], [y2per25_2,y2per75_2], [y2per25_3,y2per75_3], [y2per25_4,y2per75_4], [y2per25_5,y2per75_5]]).T
        y3std = np.array([[y3per25_1,y3per75_1], [y3per25_2,y3per75_2], [y3per25_3,y3per75_3], [y3per25_4,y3per75_4], [y3per25_5,y3per75_5]]).T

        stern_Lx = Lit_functions.Stern_MIR(np.arange(42, 48, 0.25))
        just_Lx = Lit_functions.Just_alpha_ox(np.arange(42, 48, 0.25))

        fig = plt.figure(figsize=(21, 8))
        gs = fig.add_gridspec(ncols=3,nrows=1,left=0.06,right=0.95,top=0.86,bottom=0.14,wspace=0.0)
        ax1 = plt.subplot(gs[0])#, aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_ylabel(ylabel+yunits)
        if X != 'Lx':
            ax1.set_xlabel(xlabel1+xunits)
        ax1.set_yticks(yticks) 
        ax1.set_xticks(xticks)
        ax1.grid()
        if X == 'UV-MIR-FIR':
            ax1.plot(np.arange(42,48,0.25),just_Lx,color='b',label='Just 2007')
        elif Y == 'UV-MIR-FIR':
            ax1.plot(just_Lx,np.arange(42,48,0.25),color='b',label='Just 2007')
        ax1.scatter(x1,y1,color='k',marker='.',s=35,alpha=0.5,rasterized=True)
        if error:
            ax1.errorbar(x1,y1,xerr=x1err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax1.plot(x1med, y1med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax1.errorbar(x1med,y1med,xerr=x1std,yerr=y1std,color='k',markeredgecolor='white',linestyle='')
        if X != 'Lx':
            secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax1.set_xticks([9,10,11,12,13])
            secax1.set_xlabel(xlabel1+r' [L$_{\odot}$]')
        # if Y != 'Nh':
        #     secax1 = ax1.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_yticks([9,10,11,12,13])
        #     secax1.set_ylabel(ylabel+r' [L$_{\odot}$]')
        plot_fit(x1med,y1med,2,43,45.5,'k')
        # plot_fit(x1[x1 < 44],y1[x1 < 44],1,42.5,44,'r')
        # plot_fit(x1[x1 > 44], y1[x1 > 44],1,44,47,'purple')
        # ax1.plot(x1_good,y1_good,'.',color='purple')
        # plot_fit(x1_good, y1_good, 2, 43, 46, 'orange')
        if compare:
            ax1.scatter(np.log10(comp_uv),comp_L,marker='X',c='r',s=100)

        plt.legend(fontsize=15)

        ax2 = plt.subplot(gs[1])#, aspect='equal', adjustable='box')
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(ylim[0], ylim[1])
        # ax2.set_ylabel(ylabel+yunits)
        ax2.set_yticklabels([])
        ax2.set_xlabel(xlabel2+xunits)
        ax2.set_yticks(yticks)
        ax2.set_xticks(xticks)
        ax2.grid()

        if X == 'UV-MIR-FIR':
            ax2.plot(np.arange(42,48,0.25),stern_Lx,color='r',label='Stern 2015')
        elif Y == 'UV-MIR-FIR':
            ax2.plot(stern_Lx, np.arange(42, 48, 0.25),color='r', label='Stern 2015')
        ax2.scatter(x2,y2,color='k',marker='.',alpha=0.5,s=35,rasterized=True)
        if error:
            ax2.errorbar(x2,y2,xerr=x2err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax2.plot(x2med, y2med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax2.errorbar(x2med,y2med,xerr=xstd,yerr=y2std,color='k',markeredgecolor='white',linestyle='')
        if X != 'Lx':
            secax2 = ax2.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax2.set_xticks([9,10,11,12,13])
            secax2.set_xlabel(xlabel2+r' [L$_{\odot}$]')

        # ax2.scatter(x2_good,y2_good,color='b',marker='.',alpha=0.75,s=35,rasterized=True)
        # plot_fit(x2_good, y2_good, 2, 43, 46, 'orange')
        # plot_fit(x2,y2,1,43,46,'purple')

        # if Y != 'Nh':
        #     secax2 = ax2.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax2.set_yticks([9,10,11,12,13])
        #     secax2.set_ylabel(ylabel+r' [L$_{\odot}$]')

        plot_fit(x2med, y2med, 2, 43, 45.5, 'k')
        # ytest1 = 0.54824163*(np.arange(43, 45.5))+19.62959268
        # ytest2 = 1.0*np.arange(43, 45.5)
        # ax2.plot(np.arange(43,45.5),ytest1,color='b')
        # ax2.plot(np.arange(43,45.5),ytest2,color='orange')
        # plot_fit(x2med, y2med, 1, 42, 47, 'r')
        # plot_fit(x2, y2, 1, 42, 47, 'b')

        if compare:
            ax2.scatter(np.log10(comp_mir),comp_L,marker='X',c='r',s=100)
        # plot_fit(10**x2_good,10**y2_good,1,40,48,'orange')
        ax2.legend(fontsize=15)

        ax3 = plt.subplot(gs[2])#, aspect='equal', adjustable='box')
        ax3.set_xlim(xlim[0], xlim[1])
        ax3.set_ylim(ylim[0], ylim[1])
        # ax3.set_ylabel(ylabel+yunits)
        ax3.set_yticklabels([]) 
        if X != 'Lx':
            ax3.set_xlabel(xlabel3+xunits)
        ax3.set_yticks(yticks)
        ax3.set_xticks(xticks)
        ax3.grid()

        # z3 = np.polyfit(x3[self.up_check == 0],y3[self.up_check == 0], 1)
        # p3 = np.poly1d(z3)
        # xrange3 = np.linspace(43.5,47,5)
        # yang = Lit_functions.Yang_FIR_Lx(xrange3)


        if 'FIR' in X:

            stack_x = np.asarray([43.46,43.72,44.068,44.18,44.545,44.95])- np.log10(3)
            stack_y = np.asarray([43.5,43.71,43.857,44.11,44.31,44.569])
            # print('Bin 1: ',len(stack_bins[stack_bins == 0]))
            # print('Bin 2: ',len(stack_bins[stack_bins == 1]))
            # print('Bin 3: ',len(stack_bins[stack_bins == 2]))
            # print('Bin 4: ',len(stack_bins[stack_bins == 3]))
            # print('Bin 5: ',len(stack_bins[stack_bins == 4]))
            # print('Bin 6: ',len(stack_bins[stack_bins == 5]))
            # print('Total: ',len(stack_bins[stack_bins == 0])+len(stack_bins[stack_bins == 1])+len(stack_bins[stack_bins == 2])+len(stack_bins[stack_bins == 3])+len(stack_bins[stack_bins == 4])+len(stack_bins[stack_bins == 5]))

            # ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='gray',marker=8,s=45,edgecolors=None, alpha=1,rasterized=True)
            # ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='gray', marker=1,s=45, alpha=1)
            if fir_field:
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] == 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] == 'S82X'],marker='x',color='k',alpha=0.75,s=25)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
            else:
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
        
        if 'FIR' in Y:
            if fir_field:
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=11,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=2,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] == 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] == 'S82X'],marker='x',color='k',alpha=0.75,s=25)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
            else:
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
        

            plot_fit(x3med,y3med,2,43,45.5,'k',lw=3)

            # ax3.scatter(xF1[self.up_check == 1],y3[self.up_check == 1],color='gray',marker='|',alpha=0.5)

            # print('check: ',len(np.where(xF1[self.up_check == 1] > x3[self.up_check == 1])[0]))

            # for i in range(len(x3[self.up_check == 1])):
                # print([xF1[self.up_check == 1][i], x3[self.up_check == 1][i]])
                # ax3.plot([xF1[self.up_check == 1][i],x3[self.up_check == 1][i]],[y3[self.up_check == 1][i],y3[self.up_check == 1][i]],color='gray',alpha=0.45,zorder=0)
                # ax3.plot(xF1[i],y3[self.up_check == 1][i],'.',color='b')


            # if stack_color:
            #     ax3.scatter(x3[stack_bins == 0],y3[stack_bins == 0],marker='s',color='purple')
            #     ax3.scatter(x3[stack_bins == 1],y3[stack_bins == 1],marker='s',color='yellow')
            #     ax3.scatter(x3[stack_bins == 2],y3[stack_bins == 2],marker='s',color='blue')
            #     ax3.scatter(x3[stack_bins == 3],y3[stack_bins == 3],marker='s',color='skyblue')
            #     ax3.scatter(x3[stack_bins == 4],y3[stack_bins == 4],marker='s',color='green')
            #     ax3.scatter(x3[stack_bins == 5],y3[stack_bins == 5],marker='s',color='orange')

            # plt.scatter(stack_x, stack_y, marker='P', color='r')

            if compare:
                ax3.scatter(np.log10(comp_fir),comp_L,marker='X',c='r',s=100)
            # plot_fit(x3_good,y3_good,1,43,45,'orange')
            # plot_fit(x3[x3 < 44.5],y3[x3 < 44.5],1,42.5,44.5,'r')
            # plot_fit(x3[x3 > 44.5], y3[x3 > 44.5],1,44.5,46.5,'purple')
            # ax3.plot(xrange3,yang,color='orange',lw=2)
        else:
            ax3.scatter(x, y3, color='gray', marker='+', s=30,rasterized=True)
        if error:
            ax3.errorbar(x3,y3,xerr=x3err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax3.plot(x3med, y3med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax3.errorbar(x3med, y3med, xerr=xstd, yerr=y3std,
                         color='k', markeredgecolor='white', linestyle='')
        if X != 'Lx':
            secax3 = ax3.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax3.set_xticks([9,10,11,12,13])
            secax3.set_xlabel(xlabel3+r' [L$_{\odot}$]')
        if Y == 'UV-MIR-FIR':
            secax3 = ax3.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax3.set_yticks([9,10,11,12,13])
            secax3.set_ylabel(ylabel+r' [L$_{\odot}$]')

        if Y == 'UV-MIR-FIR':
            ax1.text(45.22,42.3,'a = 0.25')
            ax2.text(45.22,42.3,'a = 6')
            ax3.text(45.22,42.3,'a = 100')

        # check = (x3[self.up_check == 0] > 45) & (y3[self.up_check == 0] < 43.5)
        # print('Large scatter: ',self.ID[self.up_check == 0][check])

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_scatter_3panels_vert(self, savestring, X, Y, median, F1, uv, mir, fir, shape, L=None, uv_err=None, mir_err=None, fir_err=None, error=False, compare=False, comp_L=None, comp_uv=None, comp_mir=None, comp_fir=None, stack_color=False, stack_bins=None, F100_ratio=None, field=None, fir_field=False):
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
            xticks = [43,43.5,44,44.5,45,45.5,46]
            xlim = [42.5, 46]

            b1x1 = (x1 > 43) & (x1 < 43.5)
            b1x2 = (x1 > 43.5) & (x1 < 44)
            b1x3 = (x1 > 44) & (x1 < 44.5)
            b1x4 = (x1 > 44.5) & (x1 < 45)
            b1x5 = (x1 > 45)
            b1x6 = (x1 > 100)

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
            x1err = uv_err
            x2err = mir_err
            x3err = fir_err
            xF1 = np.log10(F100_ratio)

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
            b1x6 = (x1 > 45) & (x1 <45.5)

            b2x1 = (x2 > 43) & (x2 < 43.5)
            b2x2 = (x2 > 43.5) & (x2 < 44)
            b2x3 = (x2 > 44) & (x2 < 44.5)
            b2x4 = (x2 > 44.5) & (x2 < 45)
            b2x5 = (x2 > 45) & (x2 < 45.5)

            # b2x1 = (x2 > 42.5) & (x2 < 43)
            # b2x2 = (x1 > 43) & (x2 < 43.5)
            # b2x3 = (x2 > 43.5) & (x2 < 44)
            # b2x4 = (x2 > 44) & (x2 < 44.5)
            # b2x5 = (x2 > 44.5) & (x2 < 45)

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
            y1err = uv_err
            y2err = mir_err
            y3err = fir_err

            ylabel1 = (r'log L (0.25$\mu$m)')
            ylabel2 = (r'log L (6$\mu$m)')
            ylabel3 = (r'log L (100$\mu$m)')
            ylabel = (r'log L (a $\mu$m)')
            yunits = ' [erg/s]'
            yvar1 = r'L (0.25$\mu$m)'
            yvar2 = r'L (6$\mu$m)'
            yvar3 = r'L (100$\mu$m)'
            yticks = [42,43,44,45,46,47] 
            ylim = [41, 47]

        # Set median points for X-axis bins
        y1med1, y2med1, y3med1 = np.nanmedian(y1[b1x1]), np.nanmedian(y2[b2x1]), np.nanmedian(y3[b3x1])
        y1med2, y2med2, y3med2 = np.nanmedian(y1[b1x2]), np.nanmedian(y2[b2x2]), np.nanmedian(y3[b3x2])
        y1med3, y2med3, y3med3 = np.nanmedian(y1[b1x3]), np.nanmedian(y2[b2x3]), np.nanmedian(y3[b3x3])
        y1med4, y2med4, y3med4 = np.nanmedian(y1[b1x4]), np.nanmedian(y2[b2x4]), np.nanmedian(y3[b3x4])
        y1med5, y2med5, y3med5 = np.nanmedian(y1[b1x5]), np.nanmedian(y2[b2x5]), np.nanmedian(y3[b3x5])
        y1med6 = np.nanmedian(y1[b1x6])

        # y1std1, y2std1, y3std1 = np.std(y1[b1x1]), np.std(y2[b2x1]), np.std(y3[b3x1]) 
        # y1std2, y2std2, y3std2 = np.std(y1[b1x2]), np.std(y2[b2x2]), np.std(y3[b3x2]) 
        # y1std3, y2std3, y3std3 = np.std(y1[b1x3]), np.std(y2[b2x3]), np.std(y3[b3x3]) 
        # y1std4, y2std4, y3std4 = np.std(y1[b1x4]), np.std(y2[b2x4]), np.std(y3[b3x4]) 
        # y1std5, y2std5, y3std5 = np.std(y1[b1x5]), np.std(y2[b2x5]), np.std(y3[b3x5]) 
        # y1std6 = np.std(y1[b1x6])

        y1per25_1, y2per25_1, y3per25_1 = y1med1 - np.nanpercentile(y1[b1x1],25), y2med1 - np.nanpercentile(y2[b2x1],25), y3med1 - np.nanpercentile(y3[b3x1],25) 
        y1per25_2, y2per25_2, y3per25_2 = y1med2 - np.nanpercentile(y1[b1x2],25), y2med2 - np.nanpercentile(y2[b2x2],25), y3med2 - np.nanpercentile(y3[b3x2],25) 
        y1per25_3, y2per25_3, y3per25_3 = y1med3 - np.nanpercentile(y1[b1x3],25), y2med3 - np.nanpercentile(y2[b2x3],25), y3med3 - np.nanpercentile(y3[b3x3],25) 
        y1per25_4, y2per25_4, y3per25_4 = y1med4 - np.nanpercentile(y1[b1x4],25), y2med4 - np.nanpercentile(y2[b2x4],25), y3med4 - np.nanpercentile(y3[b3x4],25) 
        y1per25_5, y2per25_5, y3per25_5 = y1med5 - np.nanpercentile(y1[b1x5],25), y2med5 - np.nanpercentile(y2[b2x5],25), y3med5 - np.nanpercentile(y3[b3x5],25) 
        y1per25_6 = np.nanpercentile(y1[b1x6],25)

        y1per75_1, y2per75_1, y3per75_1 = np.nanpercentile(y1[b1x1],75) - y1med1, np.nanpercentile(y2[b2x1],75) - y2med1, np.nanpercentile(y3[b3x1],75) - y3med1 
        y1per75_2, y2per75_2, y3per75_2 = np.nanpercentile(y1[b1x2],75) - y1med2, np.nanpercentile(y2[b2x2],75) - y2med2, np.nanpercentile(y3[b3x2],75) - y3med2 
        y1per75_3, y2per75_3, y3per75_3 = np.nanpercentile(y1[b1x3],75) - y1med3, np.nanpercentile(y2[b2x3],75) - y2med3, np.nanpercentile(y3[b3x3],75) - y3med3 
        y1per75_4, y2per75_4, y3per75_4 = np.nanpercentile(y1[b1x4],75) - y1med4, np.nanpercentile(y2[b2x4],75) - y2med4, np.nanpercentile(y3[b3x4],75) - y3med4 
        y1per75_5, y2per75_5, y3per75_5 = np.nanpercentile(y1[b1x5],75) - y1med5, np.nanpercentile(y2[b2x5],75) - y2med5, np.nanpercentile(y3[b3x5],75) - y3med5 
        y1per75_6 = np.nanpercentile(y1[b1x6],75)

        x1_good, y1_good = remove_outliers(x1,y1,[43,46])
        x2_good, y2_good = remove_outliers(x2,y2,[43,46])
        x3_good, y3_good = remove_outliers(x3,y3,[43,46])


        # xmed = np.array([xmed1,xmed2,xmed3,xmed4,xmed5])
        if X == 'Lbol':
            xmed = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
        elif X == 'Lx':
            x1med = np.array([43.25, 43.75, 44.25, 44.75, 45.25, np.nan])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
        elif X == 'UV-MIR-FIR':
            x1med = np.array([42.75, 43.25, 43.75, 44.25, 44.75, 45.25])
            # x2med = np.array([42.75, 43.25, 43.75, 44.25, 44.75])
            x2med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])
            x3med = np.array([43.25, 43.75, 44.25, 44.75, 45.25])

        y1med = np.array([y1med1, y1med2, y1med3, y1med4, y1med5, y1med6])
        y2med = np.array([y2med1, y2med2, y2med3, y2med4, y2med5])
        y3med = np.array([y3med1, y3med2, y3med3, y3med4, y3med5])

        xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        x1std = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        # y1std = np.array([y1std1, y1std2, y1std3, y1std4, y1std5, y1std6])
        # y2std = np.array([y2std1, y2std2, y2std3, y2std4, y2std5])
        # y3std = np.array([y3std1, y3std2, y3std3, y3std4, y3std5])

        y1std = np.array([[y1per25_1,y1per75_1], [y1per25_2,y1per75_2], [y1per25_3,y1per75_3], [y1per25_4,y1per75_4], [y1per25_5,y1per75_5], [y1per25_6,y1per75_6]]).T
        y2std = np.array([[y2per25_1,y2per75_1], [y2per25_2,y2per75_2], [y2per25_3,y2per75_3], [y2per25_4,y2per75_4], [y2per25_5,y2per75_5]]).T
        y3std = np.array([[y3per25_1,y3per75_1], [y3per25_2,y3per75_2], [y3per25_3,y3per75_3], [y3per25_4,y3per75_4], [y3per25_5,y3per75_5]]).T


        stern_Lx = Lit_functions.Stern_MIR(np.arange(42, 48, 0.25))
        just_Lx = Lit_functions.Just_alpha_ox(np.arange(42, 48, 0.25))

        fig = plt.figure(figsize=(10, 21))
        gs = fig.add_gridspec(ncols=1,nrows=3,left=0.1,right=0.85,top=0.95,bottom=0.05,wspace=0.0,hspace=0.1)
        ax1 = plt.subplot(gs[0])#, aspect='equal', adjustable='box')
        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_ylabel(r'log L (0.25 $\mu$m) [erg s$^{-1}$]')
        ax1.set_yticks(yticks)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])
        ax1.grid()
        if X == 'UV-MIR-FIR':
            ax1.plot(np.arange(42,48,0.25),just_Lx,color='b',label='Just 2007')
        # elif Y == 'UV-MIR-FIR':
        ax1.plot(just_Lx,np.arange(42,48,0.25),color='b',label='Just et al. 2007')
        ax1.scatter(x1,y1,color='k',marker='.',s=35,alpha=0.5,rasterized=True)
        if error:
            ax1.errorbar(x1,y1,xerr=x1err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax1.plot(x1med, y1med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax1.errorbar(x1med,y1med,xerr=x1std,yerr=y1std,color='k',markeredgecolor='white',linestyle='',lw=3)
        if X != 'Lx':
            secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax1.set_xticks([9,10,11,12,13])
            secax1.set_xlabel(xlabel1+r' [L$_{\odot}$]')
        if Y == 'UV-MIR-FIR':
            secax1 = ax1.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax1.set_yticks([9,10,11,12,13])
            secax1.set_ylabel(r'log L (0.25 $\mu$m) [L$_{\odot}$]')
        # if Y != 'Nh':
        #     secax1 = ax1.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_yticks([9,10,11,12,13])
        #     secax1.set_ylabel(ylabel+r' [L$_{\odot}$]')
        plot_fit(x1med,y1med,2,43,45.5,'k')
        # plot_fit(x1[y1 < 44],y1[y1 < 44],1,42.5,47,'r')
        # plot_fit(x1[y1 > 44], y1[y1 > 44],1,42.5,47,'purple')
        # plot_fit(x1_good, y1_good, 2, 43, 45, 'orange')
        if compare:
            ax1.scatter(np.log10(comp_uv),comp_L,marker='X',c='r',s=100)
        # plt.scatter(x1[shape == 1],y1[shape == 1],color='purple')
        # plt.scatter(x1[shape == 2],y1[shape == 2],color='purple')
        # print('here')
        # plot_fit(x1[np.logical_or(shape == 1, shape == 2)],
                #  y1[np.logical_or(shape == 1, shape == 2)], 1, 43, 45, 'r')
        # plot_fit(x1[shape == 2],
                #  y1[shape == 2], 1, 43, 45, 'r')


        plt.legend(fontsize=15)

        ax2 = plt.subplot(gs[1])#, aspect='equal', adjustable='box')
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(ylim[0], ylim[1])
        ax2.set_ylabel(r'log L (6 $\mu$m) [erg s$^{-1}$]')
        # ax2.set_yticklabels([])
        ax2.set_yticks(yticks)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([])
        ax2.grid()

        if X == 'UV-MIR-FIR':
            ax2.plot(np.arange(42,48,0.25),stern_Lx,color='r',label='Stern 2015')
        # elif Y == 'UV-MIR-FIR':
        ax2.plot(stern_Lx, np.arange(42, 48, 0.25),color='r', label='Stern 2015')
        ax2.scatter(x2,y2,color='k',marker='.',alpha=0.5,s=35,rasterized=True)
        if error:
            ax2.errorbar(x2,y2,xerr=x2err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax2.plot(x2med, y2med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax2.errorbar(x2med,y2med,xerr=xstd,yerr=y2std,color='k',markeredgecolor='white',linestyle='',lw=3)
        if X != 'Lx':
            secax2 = ax2.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax2.set_xticks([9,10,11,12,13])
            secax2.set_xlabel(xlabel2+r' [L$_{\odot}$]')
        if Y == 'UV-MIR-FIR':
            secax2 = ax2.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax2.set_yticks([9, 10, 11, 12, 13])
            secax2.set_ylabel(r'log L (6 $\mu$m) [L$_{\odot}$]')

        # ax2.scatter(x2_good,y2_good,color='b',marker='.',alpha=0.75,s=35,rasterized=True)
 
        # if Y != 'Nh':
        #     secax2 = ax2.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
        #     secax2.set_yticks([9,10,11,12,13])
        #     secax2.set_ylabel(ylabel+r' [L$_{\odot}$]')

        plot_fit(x2med, y2med, 1, 43, 45.5, 'k')
        # ytest1 = 0.54824163*(np.arange(43, 45.5))+19.62959268
        # ytest2 = 1.0*np.arange(43, 45.5)
        # ax2.plot(np.arange(43,45.5),ytest1,color='b')
        # ax2.plot(np.arange(43,45.5),ytest2,color='orange')
        # plot_fit(x2med, y2med, 1, 42, 47, 'r')
        # plot_fit(x2, y2, 1, 42, 47, 'b')

        if compare:
            ax2.scatter(np.log10(comp_mir),comp_L,marker='X',c='r',s=100)
        # plot_fit(10**x2_good,10**y2_good,1,40,48,'orange')
        ax2.legend(fontsize=15)

        ax3 = plt.subplot(gs[2])#, aspect='equal', adjustable='box')
        ax3.set_xlim(xlim[0], xlim[1])
        ax3.set_ylim(ylim[0], ylim[1])
        ax3.set_ylabel(ylabel+yunits)
        ax3.set_ylabel(r'log L (100 $\mu$m) [erg s$^{-1}$]')
        # ax3.set_yticklabels([]) 
        ax3.set_xlabel(xlabel3+xunits)
        ax3.set_xlabel(r'log L$_{\rm X}$ [erg s$^{-1}$]')
        ax3.set_yticks(yticks)
        ax3.set_xticks(xticks)
        ax3.grid()

        # z3 = np.polyfit(x3[self.up_check == 0],y3[self.up_check == 0], 1)
        # p3 = np.poly1d(z3)
        # xrange3 = np.linspace(43.5,47,5)
        # yang = Lit_functions.Yang_FIR_Lx(xrange3)


        if 'FIR' in X:

            stack_x = np.asarray([43.46,43.72,44.068,44.18,44.545,44.95])- np.log10(3)
            stack_y = np.asarray([43.5,43.71,43.857,44.11,44.31,44.569])
            # print('Bin 1: ',len(stack_bins[stack_bins == 0]))
            # print('Bin 2: ',len(stack_bins[stack_bins == 1]))
            # print('Bin 3: ',len(stack_bins[stack_bins == 2]))
            # print('Bin 4: ',len(stack_bins[stack_bins == 3]))
            # print('Bin 5: ',len(stack_bins[stack_bins == 4]))
            # print('Bin 6: ',len(stack_bins[stack_bins == 5]))
            # print('Total: ',len(stack_bins[stack_bins == 0])+len(stack_bins[stack_bins == 1])+len(stack_bins[stack_bins == 2])+len(stack_bins[stack_bins == 3])+len(stack_bins[stack_bins == 4])+len(stack_bins[stack_bins == 5]))

            # ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='gray',marker=8,s=45,edgecolors=None, alpha=1,rasterized=True)
            # ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1], color='gray', marker=1,s=45, alpha=1)
            if fir_field:
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] == 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] == 'S82X'],marker='x',color='k',alpha=0.75,s=25)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
            else:
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
        
        if 'FIR' in Y:
            if fir_field:
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=11,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] != 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] != 'S82X'],marker=2,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1][field[self.up_check == 1] == 'S82X'], y3[self.up_check == 1][field[self.up_check == 1] == 'S82X'],marker='x',color='k',alpha=0.75,s=25)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
            else:
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=8,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 1], y3[self.up_check == 1],marker=1,color='gray',alpha=0.8,s=45)
                ax3.scatter(x3[self.up_check == 0], y3[self.up_check == 0], color='k', marker='.', alpha=0.65,s=40, rasterized=True)
        

            plot_fit(x3med,y3med,2,43,45.5,'k',lw=3)

            # ax3.scatter(xF1[self.up_check == 1],y3[self.up_check == 1],color='gray',marker='|',alpha=0.5)

            # print('check: ',len(np.where(xF1[self.up_check == 1] > x3[self.up_check == 1])[0]))

            # for i in range(len(x3[self.up_check == 1])):
                # print([xF1[self.up_check == 1][i], x3[self.up_check == 1][i]])
                # ax3.plot([xF1[self.up_check == 1][i],x3[self.up_check == 1][i]],[y3[self.up_check == 1][i],y3[self.up_check == 1][i]],color='gray',alpha=0.45,zorder=0)
                # ax3.plot(xF1[i],y3[self.up_check == 1][i],'.',color='b')


            # if stack_color:
            #     ax3.scatter(x3[stack_bins == 0],y3[stack_bins == 0],marker='s',color='purple')
            #     ax3.scatter(x3[stack_bins == 1],y3[stack_bins == 1],marker='s',color='yellow')
            #     ax3.scatter(x3[stack_bins == 2],y3[stack_bins == 2],marker='s',color='blue')
            #     ax3.scatter(x3[stack_bins == 3],y3[stack_bins == 3],marker='s',color='skyblue')
            #     ax3.scatter(x3[stack_bins == 4],y3[stack_bins == 4],marker='s',color='green')
            #     ax3.scatter(x3[stack_bins == 5],y3[stack_bins == 5],marker='s',color='orange')

            # plt.scatter(stack_x, stack_y, marker='P', color='r')

            if compare:
                ax3.scatter(np.log10(comp_fir),comp_L,marker='X',c='r',s=100)
            # plot_fit(x3_good,y3_good,1,43,45,'orange')
            # plot_fit(x3[x3 < 44.5],y3[x3 < 44.5],1,42.5,44.5,'r')
            # plot_fit(x3[x3 > 44.5], y3[x3 > 44.5],1,44.5,46.5,'purple')
            # ax3.plot(xrange3,yang,color='orange',lw=2)
        else:
            ax3.scatter(x, y3, color='gray', marker='+', s=30,rasterized=True)
        if error:
            ax3.errorbar(x3,y3,xerr=x3err,color='k',lw=1,linestyle='')
        if median == 'X-axis' or median == 'Both':
            ax3.plot(x3med, y3med, marker='s', color='k',ms=10, markeredgecolor='white',linestyle='')
            ax3.errorbar(x3med, y3med, xerr=xstd, yerr=y3std,color='k', markeredgecolor='white', linestyle='',lw=3)
        if X != 'Lx':
            secax3 = ax3.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax3.set_xticks([9,10,11,12,13])
            secax3.set_xlabel(xlabel3+r' [L$_{\odot}$]')
        if Y == 'UV-MIR-FIR':
            secax3 = ax3.secondary_yaxis('right', functions=(self.solar_log, self.ergs_log))
            secax3.set_yticks([9,10,11,12,13])
            secax3.set_ylabel(r'log L (100 $\mu$m) [L$_{\odot}$]')

        # check = (x3[self.up_check == 0] > 45) & (y3[self.up_check == 0] < 43.5)
        # print('Large scatter: ',self.ID[self.up_check == 0][check])

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_scatter_1panel(self,savestring,X,Y,median,F1,uv,mir,fir,Nh,shape,L=None,line=None,Lx_h=None,Lum_range=None):
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

        elif X == 'FIR_lum':
            x = np.log10(Lum_range)
            # x = fir
            xlabel = r'log L$_{\rm FIR}$'
            xunits = ' [erg/s]'
            xvar = r'L$_{\rm FIR}$'
            xticks = [42, 43, 44, 45, 46]
            xlim = [42, 46]

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

        elif Y == 'Lx_h':
            y = Lx_h
            ylabel = r'log L$_{\mathrm{HX}}$'
            yunits = ' [erg/s]'
            yticks = [39, 40, 41, 42, 43, 44, 45, 46]
            yticks = [38, 39, 40, 41, 42]
            ylim = [38, 42]

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
        ranalli = Lit_functions.Ranalli(np.arange(42,48))
        torres = Lit_functions.Torres(np.arange(42,48))

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
        if line == 'Stern':
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

        if line == 'Ranalli':
            ax1.plot(np.arange(42,48),ranalli,color='k',ls='--',label='Ranalli')
            ax1.plot(np.arange(42,48),torres,color='b',ls='--',label='Torres')

        plt.tight_layout()
        plt.legend()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def L_ratio_multi_panel(self,savestring,X,Y,median,F1,uv,mir,fir,Nh,L,shape,up_check,F100_ratio=None,field=None):
        B1 = (shape == 1)
        B2 = (shape == 2)
        B3 = (shape == 3)
        B4 = (shape == 4)
        B5 = (shape == 5)

        if X == 'MIR6':
            x = mir
            xlabel = r'log L (6$\mu$m)'
            xunits1 = r'/(erg s$^{-1}$)'
            xlim1 = 42.5
            xlim2 = 46.5
            xticks = [42,43,44,45,46]

        elif X == 'Lx':
            x = self.L
            xlabel = r'log $L_{\rm X}$'
            xunits1 = r'/(erg s$^{-1}$)'
            xlim1 = 42.5
            xlim2 = 46
            xticks = [43,44,45,46]

        elif X == 'Lbol':
            x = L
            xlabel = r'log $L_{\rm bol}$'
            xunits1 = r'/(erg s$^{-1}$)'
            xlim1 = 43.75
            xlim2 = 46.75
            xticks = [44.5,45.5,46.5]

        y_uv = uv - self.L
        ylabel_uv = r'log L (0.25$\mu$m)/ $L_{\mathrm{X}}$'
        ylim1_uv = -1.5
        ylim2_uv = 2
        yticks_uv = [-1,0,1,2]

        y_mir = mir - self.L
        ylabel_mir = r'log L (6$\mu$m)/ $L_{\mathrm{X}}$'
        ylim1_mir = -1.5
        ylim2_mir = 2
        yticks_mir = [-1,0,1,2]

        y_fir = fir - self.L
        ylabel_fir = r'log L (100$\mu$m)/ $L_{\mathrm{X}}$'
        ylim1_fir = -1.5
        ylim2_fir = 2
        yticks_fir = [-1,0,1,2]

        yF1 = np.log10(F100_ratio) - self.L

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        x1 = x[B1]
        x2 = x[B2]
        x3 = x[B3]
        x4 = x[B4]
        x5 = x[B5]
        y_uv1 = y_uv[B1]
        y_uv2 = y_uv[B2]
        y_uv3 = y_uv[B3]
        y_uv4 = y_uv[B4]
        y_uv5 = y_uv[B5]

        y_mir1 = y_mir[B1]
        y_mir2 = y_mir[B2]
        y_mir3 = y_mir[B3]
        y_mir4 = y_mir[B4]
        y_mir5 = y_mir[B5]

        y_fir1 = y_fir[B1]
        y_fir2 = y_fir[B2]
        y_fir3 = y_fir[B3]
        y_fir4 = y_fir[B4]
        y_fir5 = y_fir[B5]

        yF1_1 = yF1[B1]
        yF1_2 = yF1[B2]
        yF1_3 = yF1[B3]
        yF1_4 = yF1[B4]
        yF1_5 = yF1[B5]

        up_check1 = up_check[B1]
        up_check2 = up_check[B2]
        up_check3 = up_check[B3]
        up_check4 = up_check[B4]
        up_check5 = up_check[B5]

        field1 = field[B1]
        field2 = field[B2]
        field3 = field[B3]
        field4 = field[B4]
        field5 = field[B5]

        # Medians
        y1_uv_med, y1_mir_med, y1_fir_med, x1_med = np.nanmedian(y_uv1), np.nanmedian(y_mir1), np.nanmedian(y_fir1), np.nanmedian(x1)
        y2_uv_med, y2_mir_med, y2_fir_med, x2_med = np.nanmedian(y_uv2), np.nanmedian(y_mir2), np.nanmedian(y_fir2), np.nanmedian(x2)
        y3_uv_med, y3_mir_med, y3_fir_med, x3_med = np.nanmedian(y_uv3), np.nanmedian(y_mir3), np.nanmedian(y_fir3), np.nanmedian(x3)
        y4_uv_med, y4_mir_med, y4_fir_med, x4_med = np.nanmedian(y_uv4), np.nanmedian(y_mir4), np.nanmedian(y_fir4), np.nanmedian(x4)
        y5_uv_med, y5_mir_med, y5_fir_med, x5_med = np.nanmedian(y_uv5), np.nanmedian(y_mir5), np.nanmedian(y_fir5), np.nanmedian(x5)

        # Dispersion
        y1_uv_25per, y1_mir_25per, y1_fir_25per, x1_25per = np.nanpercentile(y_uv1,25), np.nanpercentile(y_mir1, 25), np.nanpercentile(y_fir1, 25), np.nanpercentile(x1,25)
        y2_uv_25per, y2_mir_25per, y2_fir_25per, x2_25per = np.nanpercentile(y_uv2,25), np.nanpercentile(y_mir2, 25), np.nanpercentile(y_fir2, 25), np.nanpercentile(x2,25)
        y3_uv_25per, y3_mir_25per, y3_fir_25per, x3_25per = np.nanpercentile(y_uv3,25), np.nanpercentile(y_mir3, 25), np.nanpercentile(y_fir3, 25), np.nanpercentile(x3,25)
        y4_uv_25per, y4_mir_25per, y4_fir_25per, x4_25per = np.nanpercentile(y_uv4,25), np.nanpercentile(y_mir4, 25), np.nanpercentile(y_fir4, 25), np.nanpercentile(x4,25)
        y5_uv_25per, y5_mir_25per, y5_fir_25per, x5_25per = np.nanpercentile(y_uv5,25), np.nanpercentile(y_mir5, 25), np.nanpercentile(y_fir5, 25), np.nanpercentile(x5,25)
        
        y1_uv_75per, y1_mir_75per, y1_fir_75per, x1_75per = np.nanpercentile(y_uv1,75), np.nanpercentile(y_mir1, 75), np.nanpercentile(y_fir1, 75), np.nanpercentile(x1,75)
        y2_uv_75per, y2_mir_75per, y2_fir_75per, x2_75per = np.nanpercentile(y_uv2,75), np.nanpercentile(y_mir2, 75), np.nanpercentile(y_fir2, 75), np.nanpercentile(x2,75)
        y3_uv_75per, y3_mir_75per, y3_fir_75per, x3_75per = np.nanpercentile(y_uv3,75), np.nanpercentile(y_mir3, 75), np.nanpercentile(y_fir3, 75), np.nanpercentile(x3,75)
        y4_uv_75per, y4_mir_75per, y4_fir_75per, x4_75per = np.nanpercentile(y_uv4,75), np.nanpercentile(y_mir4, 75), np.nanpercentile(y_fir4, 75), np.nanpercentile(x4,75)
        y5_uv_75per, y5_mir_75per, y5_fir_75per, x5_75per = np.nanpercentile(y_uv5,75), np.nanpercentile(y_mir5, 75), np.nanpercentile(y_fir5, 75), np.nanpercentile(x5,75)

        xmed = np.array([x1_med,x2_med,x3_med,x4_med,x5_med])
        xerr1, xerr2 = xmed - np.array([x1_25per,x2_25per,x3_25per,x4_25per,x5_25per]),np.array([x1_75per,x2_75per,x3_75per,x4_75per,x5_75per]) - xmed
        y_uv_med = np.array([y1_uv_med,y2_uv_med,y3_uv_med,y4_uv_med,y5_uv_med])
        y_uv_err1, y_uv_err2 = y_uv_med - np.array([y1_uv_25per,y2_uv_25per,y3_uv_25per, y4_uv_25per,y5_uv_25per]),np.array([y1_uv_75per,y2_uv_75per,y3_uv_75per,y4_uv_75per,y5_uv_75per]) - y_uv_med
        y_mir_med = np.array([y1_mir_med,y2_mir_med,y3_mir_med,y4_mir_med,y5_mir_med])
        y_mir_err1, y_mir_err2 = y_mir_med - np.array([y1_mir_25per,y2_mir_25per,y3_mir_25per, y4_mir_25per,y5_mir_25per]),np.array([y1_mir_75per,y2_mir_75per,y3_mir_75per,y4_mir_75per,y5_mir_75per]) - y_mir_med
        y_fir_med = np.array([y1_fir_med,y2_fir_med,y3_fir_med,y4_fir_med,y5_fir_med]) 
        y_fir_err1, y_fir_err2 = y_fir_med - np.array([y1_fir_25per,y2_fir_25per,y3_fir_25per, y4_fir_25per,y5_fir_25per]),np.array([y1_fir_75per,y2_fir_75per,y3_fir_75per,y4_fir_75per,y5_fir_75per]) - y_fir_med

        fig = plt.figure(figsize=(10, 21))
        gs = fig.add_gridspec(ncols=1,nrows=3)#,left=0.06,right=0.95,top=0.86,bottom=0.14,wspace=0.0)
        ax1 = plt.subplot(gs[0], aspect='equal', adjustable='box')
        ax1.scatter(x1,y_uv1,c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 1')
        ax1.scatter(x2,y_uv2,c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 2')
        ax1.scatter(x3,y_uv3,c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 3')
        ax1.scatter(x4,y_uv4,c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 4')
        ax1.scatter(x5,y_uv5,c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 5')

        ax1.errorbar(xmed, y_uv_med, xerr=[xerr1, xerr2], yerr=[y_uv_err1, y_uv_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
        ax1.scatter(x1_med, y1_uv_med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x2_med, y2_uv_med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x3_med, y3_uv_med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x4_med, y4_uv_med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x5_med, y5_uv_med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)

        ax2 = plt.subplot(gs[1], aspect='equal', adjustable='box')
        ax2.scatter(x1,y_mir1,c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax2.scatter(x2,y_mir2,c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax2.scatter(x3,y_mir3,c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax2.scatter(x4,y_mir4,c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax2.scatter(x5,y_mir5,c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)

        ax2.errorbar(xmed, y_mir_med, xerr=[xerr1, xerr2], yerr=[y_mir_err1, y_mir_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
        ax2.scatter(x1_med, y1_mir_med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x2_med, y2_mir_med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x3_med, y3_mir_med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x4_med, y4_mir_med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x5_med, y5_mir_med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)

        ax3 = plt.subplot(gs[2], aspect='equal', adjustable='box')
        ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] == 'S82X'],y_fir1[up_check1 == 1][field1[up_check1 == 1] == 'S82X'],c=c1,marker='x',rasterized=True,s=55,alpha=0.75)
        ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] == 'S82X'],y_fir2[up_check2 == 1][field2[up_check2 == 1] == 'S82X'],c=c2,marker='x',rasterized=True,s=55,alpha=0.75)
        ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] == 'S82X'],y_fir3[up_check3 == 1][field3[up_check3 == 1] == 'S82X'],c=c3,marker='x',rasterized=True,s=55,alpha=0.75)
        ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] == 'S82X'],y_fir4[up_check4 == 1][field4[up_check4 == 1] == 'S82X'],c=c4,marker='x',rasterized=True,s=55,alpha=0.75)
        ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] == 'S82X'],y_fir5[up_check5 == 1][field5[up_check5 == 1] == 'S82X'],c=c5,marker='x',rasterized=True,s=55,alpha=0.75)

        ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],y_fir1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],c='gray',marker=11,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],y_fir2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],c='gray',marker=11,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],y_fir3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],c='gray',marker=11,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],y_fir4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],c='gray',marker=11,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],y_fir5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],c='gray',marker=11,rasterized=True,s=55,alpha=0.5)

        ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],y_fir1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],c='gray',marker=2,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],y_fir2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],c='gray',marker=2,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],y_fir3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],c='gray',marker=2,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],y_fir4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],c='gray',marker=2,rasterized=True,s=55,alpha=0.5)
        ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],y_fir5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],c='gray',marker=2,rasterized=True,s=55,alpha=0.5)


        # ax3.scatter(x1[up_check1 == 1],yF1_1[up_check1 == 1],c='gray',marker='_',alpha=0.75)
        # ax3.scatter(x2[up_check2 == 1],yF1_2[up_check2 == 1],c='gray',marker='_',alpha=0.75)
        # ax3.scatter(x3[up_check3 == 1],yF1_3[up_check3 == 1],c='gray',marker='_',alpha=0.75)
        # ax3.scatter(x4[up_check4 == 1],yF1_4[up_check4 == 1],c='gray',marker='_',alpha=0.75)
        # ax3.scatter(x5[up_check5 == 1],yF1_5[up_check5 == 1],c='gray',marker='_',alpha=0.75)

        # for i in range(len(x1[up_check1 == 1])):
        #     ax3.plot([x1[up_check1 == 1][i],x1[up_check1 == 1][i]],[y_fir1[up_check1 == 1][i],yF1_1[up_check1 == 1][i]],c='gray',alpha=0.65,zorder=0,rasterized=True)
        # for i in range(len(x2[up_check2 == 1])):
        #     ax3.plot([x2[up_check2 == 1][i],x2[up_check2 == 1][i]],[y_fir2[up_check2 == 1][i],yF1_2[up_check2 == 1][i]],c='gray',alpha=0.65,zorder=0,rasterized=True)
        # for i in range(len(x3[up_check3 == 1])):
        #     ax3.plot([x3[up_check3 == 1][i],x3[up_check3 == 1][i]],[y_fir3[up_check3 == 1][i],yF1_3[up_check3 == 1][i]],c='gray',alpha=0.65,zorder=0,rasterized=True)
        # for i in range(len(x4[up_check4 == 1])):
        #     ax3.plot([x4[up_check4 == 1][i],x4[up_check4 == 1][i]],[y_fir4[up_check4 == 1][i],yF1_4[up_check4 == 1][i]],c='gray',alpha=0.65,zorder=0,rasterized=True)
        # for i in range(len(x5[up_check5 == 1])):
        #     ax3.plot([x5[up_check5 == 1][i],x5[up_check5 == 1][i]],[y_fir5[up_check5 == 1][i],yF1_5[up_check5 == 1][i]],c='gray',alpha=0.65,zorder=0,rasterized=True)

        ax3.scatter(x1[up_check1 == 0],y_fir1[up_check1 == 0],c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax3.scatter(x2[up_check2 == 0],y_fir2[up_check2 == 0],c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax3.scatter(x3[up_check3 == 0],y_fir3[up_check3 == 0],c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax3.scatter(x4[up_check4 == 0],y_fir4[up_check4 == 0],c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)
        ax3.scatter(x5[up_check5 == 0],y_fir5[up_check5 == 0],c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8)

        ax3.errorbar(xmed, y_fir_med, xerr=[xerr1, xerr2], yerr=[y_fir_err1, y_fir_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
        ax3.scatter(x1_med, y1_fir_med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax3.scatter(x2_med, y2_fir_med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax3.scatter(x3_med, y3_fir_med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax3.scatter(x4_med, y4_fir_med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax3.scatter(x5_med, y5_fir_med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        plot_fit(x,y_fir,1,43,45.5,'k')
        # plot_fit(x[up_check == 0],y_fir[up_check == 0],1,43,45.5,'r')
        # ax3.plot([42,43,44,45,46],[2,1,0,-1,-2],color='k',alpha=0.8,zorder=0)

        ax1.set_ylabel(ylabel_uv)
        ax2.set_ylabel(ylabel_mir)
        ax3.set_ylabel(ylabel_fir)
        ax3.set_xlabel(xlabel+xunits1)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.set_xlim(xlim1,xlim2)
        ax2.set_xlim(xlim1,xlim2)
        ax3.set_xlim(xlim1,xlim2)
        ax1.set_ylim(ylim1_uv,ylim2_uv)
        ax2.set_ylim(ylim1_mir,ylim2_mir)
        ax3.set_ylim(ylim1_fir,ylim2_fir)
        ax1.set_yticks(yticks_uv)
        ax2.set_yticks(yticks_mir)
        ax3.set_yticks(yticks_fir)
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax3.set_xticks(xticks)
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax1.legend()
        plt.tight_layout()

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        print('done')
        plt.show()

    def L_hist(self,savestring,x,xlabel=None,xlim=[np.nan,np.nan],bins=[np.nan,np.nan,np.nan],median=True,std=False,split=False,split_param=None,top_label=False,xlabel2=None):
        hist1 = (split_param != 0)
        hist2 = (split_param == 0)
        
        # c = 'gray'
        c = '#75bbfb'
        plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111)
        n = plt.hist(x, bins=np.arange(bins[0], bins[1], bins[2]), color=c, zorder=0)
        if split:
            # plt.hist(x[hist2], bins=np.arange(bins[0], bins[1], bins[2]), color='gray', alpha=0.5)
            plt.hist(x[hist2], bins=np.arange(bins[0], bins[1], bins[2]), histtype='step',color=c,edgecolor='k',lw=2,hatch='/',fill=True)
        if median:
            plt.axvline(np.nanmedian(x),color='k',ls='--',lw=3)
            if std:
                plt.axvline(np.std(x)+np.nanmean(x),color='k',lw=1)
                plt.axvline(np.nanmean(x)-np.std(x),color='k',lw=1)
                plt.fill_between([np.nanmean(x)-np.std(x), np.std(x)+np.nanmean(x)],[0,0],[max(n[0])+max(n[0])*0.1,max(n[0])+max(n[0])*0.1],color='gray',alpha=0.4)
        if top_label:
            secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax1.set_xticks([9,10,11,12,13])
            secax1.set_xlabel(xlabel2)

        plt.xlabel(xlabel)
        plt.grid()
        plt.ylim(0,max(n[0])+max(n[0])*0.1)

        print('mean: ', np.nanmean(x))
        print('std: ', np.std(x))
        print('min: ', min(x))
        print('max: ', max(x))
        
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_hist_zbins(self,savestring,x,xlabel=None,xlim=[np.nan,np.nan],bins=[np.nan,np.nan,np.nan],median=True,std=False,bin_type='z'):
        if bin_type == 'z':
            b1 = self.z < 0.6
            b2 = (self.z > 0.6) & (self.z < 0.9)
            b3 = self.z > 0.9
            l1 = 'z < 0.6'
            l2 = '0.6 < z < 0.9'
            l3 = '0.9 < z < 1.2'
            c1 = '#1E62E5'
            c2 = '#04BF0C'
            c3 = '#D23737'
        elif bin_type == 'Lx':
            b1 = self.L < 43.75
            b2 = (self.L > 43.75) & (self.L < 44.5)
            b3 = self.L > 44.5
            l1 = r'43 < log L$_{\rm X}$ < 43.75'
            l2 = r'43.75 < L$_{\rm X}$ < 44.5'
            l3 = r'44.5 < L$_{\rm X}$ < 45.5'
            c1 = 'blue'
            c2 = 'green'
            c3 = 'red'
            
        
        plt.figure(figsize=(12,12))
        n1 = plt.hist(x[b1], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color=c3,lw=4,alpha=0.8,label=l1)
        n2 = plt.hist(x[b2], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color=c2,lw=4,alpha=0.8,label=l2)
        n3 = plt.hist(x[b3], bins=np.arange(bins[0],bins[1],bins[2]),histtype='step',color=c1,lw=4,alpha=0.8,label=l3)
        n = np.append(n1[0],n2[0])
        n = np.append(n,n3[0])
        if median:
            plt.axvline(np.nanmean(x[b1]), color=c3, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b2]), color=c2, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b3]), color=c1, ls='--', lw=3)
        plt.xlabel(xlabel)
        plt.grid()
        plt.ylim(0,max(n)+max(n)*0.1)
        plt.legend(fontsize=18)

        print('bin 1: ',np.nanmean(x[b1]),np.std(x[b1]))
        print('bin 2: ',np.nanmean(x[b2]),np.std(x[b2]))
        print('bin 3: ',np.nanmean(x[b3]),np.std(x[b3]))

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_scatter_comp(self,savestring,L1,L2,color_array,cbar=True,one2one=True,xlabel=None,ylabel=None,colorbar_label=None):
        L1 = np.asarray(L1)
        L2 = np.asarray(L2)

        if xlabel is None:
            xlab = ''
        else:
            xlab = xlabel

        if ylabel is None:
            ylab = ''
        else:
            ylab = ylabel

        x1 = min(L1) - 0.5
        x2 = max(L1) + 1

        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot(111, aspect='equal', adjustable='box')
        if cbar:
            pts = plt.scatter(L1, L2, c=color_array, edgecolor='k', s=100)
            axcb = fig.colorbar(pts)  # make colorbar
            # axcb.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits Lir
            axcb.mappable.set_clim(21, 25)  # initialize colorbar limits Nh
            axcb.set_label(label=colorbar_label)
        else:
            plt.plot(L1,L2,'.',ms=15,color='gray')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        if one2one:
            plt.plot(np.arange(x1,x2),np.arange(x1,x2),color='k')
        plt.xlim(40.5,45)
        plt.ylim(40.5,45)
        ax1.set_xticks([41,42,43,44,45])
        ax1.set_yticks([41,42,43,44,45])
        plt.grid()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def IR_colors(self,savestring,x,y,L=[np.nan],gal_x=[np.nan],gal_y=[np.nan],donley=True,Lacy=False,colorbar=False,colorbar_label='',agn=[np.nan],select_sources=False):

        x1d = np.linspace(0.08, 1.5)
        x2d = np.linspace(0.35, 2.0)

        x1L = np.linspace(-0.3, 1.5)

        arp220 = (self.ID == 'UGC 09913')
        mrk231 = (self.ID == 'UGC 08058')
        print('here')
        print(L)

        fig = plt.figure(figsize=(15,13))
        ax = plt.subplot(111, aspect='equal', adjustable='box')
        if colorbar:
            # pts = plt.scatter(x[~agn],y[~agn],c=L[~agn],edgecolor='k',s=175)
            agn_pts = plt.scatter(x[agn],y[agn],c=L[agn],marker='*',edgecolor='k',s=175)
            # plt.colorbar(label=colorbar_label)
            axcb = fig.colorbar(agn_pts)  # make colorbar
            # axcb.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits
            axcb.mappable.set_clim(43, 46)  # initialize colorbar limits
            # axcb.remove()
            # axcb2 = fig.colorbar(pts)  # make colorbar
            # axcb2.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits
            # axcb2.mappable.set_clim(43, 46)
            # axcb2.set_label(label=colorbar_label)
        else:
            plt.plot(x,y,'.',ms=15,color='r')
        if donley:
            plt.plot(x1d, 1.21*x1d + 0.27,color='k',lw=3,label='Donley et al. 2012')
            plt.plot(x2d, 1.21*x2d - 0.27,color='k',lw=3)
            plt.vlines(0.08,ymin=0.15,ymax=0.37,color='k',lw=3)
            plt.hlines(0.15,xmin=0.08,xmax=0.35,color='k',lw=3)
        if Lacy:
            plt.plot(x1L, 0.8*x1L + 0.5,color='gray',ls='--',lw=3,label='Lacy et al. 2013')
            plt.vlines(-0.3,ymin=-0.3,ymax=0.2575,color='gray',ls='--',lw=3)
            plt.hlines(-0.3,xmin=-0.3,xmax=2,color='gray',ls='--',lw=3)
        if select_sources:
            select1 = plt.scatter(x[arp220],y[arp220],c=L[arp220],edgecolor='k',marker='X',label='Arp 229',s=350)
            axcb3 = fig.colorbar(select1)  # make colorbar
            axcb3.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits
            axcb3.remove()
            select2 = plt.scatter(x[mrk231],y[mrk231],c=L[mrk231],edgecolor='k',marker='P',label='Mrk 231',s=400)
            axcb4 = fig.colorbar(select2)  # make colorbar
            axcb4.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits
            axcb4.remove()
        plt.xlim(-0.5,1.5)
        plt.ylim(-0.5,1.5)
        plt.xlabel(r'log $\frac{f_{5.8}}{f_{3.4}}$',fontsize=26)
        plt.ylabel(r'log $\frac{f_{8.0}}{f_{4.5}}$',fontsize=26)
        plt.grid()
        plt.legend(fontsize=13)
        # plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_ratio_1panel_GOALS(self,savestring,X,Y,median,F1,uv,mir,fir,shape,L=None,goals_Lx=[np.nan],goals_Lbol=[np.nan]):
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
            # y = L - (self.L+np.log10(0.611))
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
        # ax1.set_xlim(42.5,46.75)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel+xunits)
        ax1.set_yticks(yticks)
        ax1.set_xticks(xticks)
        ax1.grid()
        ax1.plot(np.arange(42,48,0.25),np.log10(durras_K),color='r',label='Duras+2020')
        ax1.plot(np.arange(42,48,0.25),np.log10(hopkins_K),color='b',label='Hopkins+2007')
        # ax1.plot(L,np.log10(check),'.',color='b')
        ax1.scatter(x,y,color='gray',marker='o',edgecolor=None,s=100,alpha=0.25,rasterized=True)
        # if median == 'X-axis' or median == 'Both':
            # ax1.plot(xmed, ymed, marker='s', color='k',ms=10)
            # ax1.errorbar(xmed,ymed,xerr=xstd,yerr=y1std,color='k')
        plt.plot(goals_Lbol,goals_Lbol-goals_Lx,'o',ms=10,color='r',label='GOALS AGN')
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')
        ax1.legend(fontsize=15)

        # plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
        plt.show()

    def violin_plot(self,savestring,var,x,shape,bins='shape'):
        plt.rcParams['font.size'] = 30
        plt.rcParams['axes.linewidth'] = 3.5
        plt.rcParams['xtick.major.size'] = 5.5
        plt.rcParams['xtick.major.width'] = 4.5
        plt.rcParams['ytick.major.size'] = 5.5
        plt.rcParams['ytick.major.width'] = 4.5

        shape = shape[np.isfinite(x)]
        x = x[np.isfinite(x)]

        b1 = shape == 1
        b2 = shape == 2
        b3 = shape == 3
        b4 = shape == 4
        b5 = shape == 5

        if var == 'Nh':
            ylabel = r'log N$_{\mathrm{H}}$'
            units = r' [cm$^{-2}$]'
            ylim1 = 19.5
            ylim2 = 25

        elif var == 'Lone':
            x -= np.log10(3.8E33)
            ylabel = r'log L (1$\mu$m)'
            units = r' [L$_{\odot}$]'
            ylim1 = 8
            ylim2 = 13

        elif var == 'Lx':
            x -= np.log10(3.8E33)

            ylabel = r'log L$_{\mathrm{X}}$'
            units = r' [L$_{\odot}$]'
            ylim1 = 8
            ylim2 = 13

        elif var == 'Lbol':
            x -= np.log10(3.8E33)
            # if any(ulirg_x) != None:
                # ulirg_x -= np.log10(3.8E33)
            ylabel = r'log L$_{\mathrm{bol}}$'
            units = r' [L$_{\odot}$]'  
            ylim1 = 9.5
            ylim2 = 14.5

        elif var == 'Lbol/Lx':
            ylabel = r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$'
            units = ''
            ylim1 = -1
            ylim2 = 4

        xticklabels = ['0','1','2','3','4','5']

        x1 = x[b1]
        x2 = x[b2]
        x3 = x[b3]
        x4 = x[b4]
        x5 = x[b5]

        x1_25 = np.nanpercentile(x1,25)
        x2_25 = np.nanpercentile(x2,25)
        x3_25 = np.nanpercentile(x3,25)
        x4_25 = np.nanpercentile(x4,25)
        x5_25 = np.nanpercentile(x5,25)

        x1_75 = np.nanpercentile(x1,75)
        x2_75 = np.nanpercentile(x2,75)
        x3_75 = np.nanpercentile(x3,75)
        x4_75 = np.nanpercentile(x4,75)
        x5_75 = np.nanpercentile(x5,75)

        x_median = np.array([np.nanmedian(x1),np.nanmedian(x2),np.nanmedian(x3),np.nanmedian(x4),np.nanmedian(x5)])
        x_25 = np.array([x1_25,x2_25,x3_25,x4_25,x5_25])
        x_75 = np.array([x1_75,x2_75,x3_75,x4_75,x5_75])

        xerr_lo = x_median - x_25
        xerr_hi = x_75 - x_median

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        def solar(x):
            return x - np.log10(3.8E33)

        def ergs(x):
            return x + np.log10(3.8E33)

        fig = plt.figure(figsize=(11,11))
        ax1 = plt.subplot(111, aspect='equal', adjustable='box')

        # parts = ax1.violinplot([x1,x2,x3,x4,x5], positions=[1,2,3,4,5])
        parts1 = ax1.violinplot(x1, positions=[2], widths=0.7)
        parts2 = ax1.violinplot(x2, positions=[3], widths=0.7)
        parts3 = ax1.violinplot(x3, positions=[4], widths=0.7)
        parts4 = ax1.violinplot(x4, positions=[5], widths=0.7)
        parts5 = ax1.violinplot(x5, positions=[6], widths=0.7)
        ax1.plot([2,3,4,5,6],x_median,color='k')
        ax1.plot([2,3,4,5,6],x_median,'o',ms=12,color='k')

        ax1.errorbar([2,3,4,5,6],x_median,yerr=[xerr_lo,xerr_hi],color='k',elinewidth=3.5,lw=0)

        for partname in ('cbars','cmins','cmaxes'):
            vp = parts1[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

            vp = parts2[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

            vp = parts3[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

            vp = parts4[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

            vp = parts5[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

        for pc in parts1['bodies']:
            pc.set_facecolor(c1)
            pc.set_edgecolor(c1)
            pc.set_alpha(1)

        for pc in parts2['bodies']:
            pc.set_facecolor(c2)
            pc.set_edgecolor(c2)
            pc.set_alpha(1)

        for pc in parts3['bodies']:
            pc.set_facecolor(c3)
            pc.set_edgecolor(c3)
            pc.set_alpha(1)

        for pc in parts4['bodies']:
            pc.set_facecolor(c4)
            pc.set_edgecolor(c4)
            pc.set_alpha(1)

        for pc in parts5['bodies']:
            pc.set_facecolor(c5)
            pc.set_edgecolor(c5)
            pc.set_alpha(1)

        ax1.set_ylim(ylim1, ylim2)
        plt.gca().invert_xaxis()
        ax1.set_ylabel(ylabel+units)
        ax1.set_xlabel('Panel Number')
        ax1.set_xticklabels(xticklabels)
        
        if var == 'Lbol':
            secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
            secax1.set_ylabel(ylabel+' [erg/s]')
        elif var == 'Lone':
            secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
            secax1.set_ylabel(ylabel+' [erg/s]')
        elif var == 'Lx':
            secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
            secax1.set_ylabel(ylabel+' [erg/s]')

        ax1.grid()
        plt.savefig('/users/connor_auge/Desktop/Final_Plots/'+savestring+'.pdf')
        plt.show()

    def mix_plot(self,savestring,x,y,shape,bins='shape'):

        temp_opt = [0.950,0.876,0.794,0.702,0.598,0.480,0.342,0.178,-0.024,-0.286,-0.656]
        temp_nir = [-0.719,-0.621,-0.513,-0.394,-0.262,-0.112,0.060,0.263,0.509,0.823,1.264]

        temp_opt2 = [0.950, 0.857, 0.752, 0.634, 0.496,0.334,0.138,-0.109,-0.439,-0.934,-1.972]
        temp_nir2 = [-0.719, -0.618, -0.506, -0.382, -0.244, -0.086, 0.097, 0.316, 0.586, 0.941, 1.469]

        temp_opt3 = [0.950,0.864,0.766,0.655,0.525,0.372,0.184,-0.053,-0.373,-0.857,-1.876]
        temp_nir3 = [-0.719,-0.620,-0.511,-0.392,-0.260,-0.111,0.059,0.257,0.497,0.800,1.218]

        red_opt = [0.950,0.553,0.156,-0.241,-0.639,-1.036,-1.433,-1.830,-2.227]#,-2.624,-3.021]
        red_nir = [-0.719,-0.890,-1.060,-1.231,-1.402,-1.572,-1.743,-1.914,-2.084]#,-2.255,-2.426]

        if bins == 'shape':
            b1 = shape == 1
            b2 = shape == 2
            b3 = shape == 3
            b4 = shape == 4
            b5 = shape == 5

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        x1, y1 = x[b1], y[b1]
        x2, y2 = x[b2], y[b2]
        x3, y3 = x[b3], y[b3]
        x4, y4 = x[b4], y[b4]
        x5, y5 = x[b5], y[b5]

        xmed1, ymed1 = np.nanmean(x[b1]), np.nanmean(y[b1])
        xmed2, ymed2 = np.nanmean(x[b2]), np.nanmean(y[b2])
        xmed3, ymed3 = np.nanmean(x[b3]), np.nanmean(y[b3])
        xmed4, ymed4 = np.nanmean(x[b4]), np.nanmean(y[b4])
        xmed5, ymed5 = np.nanmean(x[b5]), np.nanmean(y[b5])

        ystd1 = np.std(y[b1])
        ystd2 = np.std(y[b2])
        ystd3 = np.std(y[b3])
        ystd4 = np.std(y[b4])
        ystd5 = np.std(y[b5])

        xstd1 = np.std(x[b1])
        xstd2 = np.std(x[b2])
        xstd3 = np.std(x[b3])
        xstd4 = np.std(x[b4])
        xstd5 = np.std(x[b5])

        # ymed = np.array([ymed1, ymed2, ymed3, ymed4, ymed5])
        # xmed = np.array([xmed1, xmed2, xmed3, xmed4, xmed5])
        # xstd = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        # ystd = np.array([ystd1, ystd2, ystd3, ystd4, ystd5])

        plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        ax.scatter(x1, y1, marker = 'P', color=c1, rasterized=True)
        ax.scatter(x2, y2, marker = 'P', color=c2, rasterized=True)
        ax.scatter(x3, y3, marker = 'P', color=c3, rasterized=True)
        ax.scatter(x4, y4, marker = 'P', color=c4, rasterized=True)
        ax.scatter(x5, y5, marker = 'P', color=c5, rasterized=True)

        ax.errorbar(xmed1,ymed1,xerr=xstd1, yerr=ystd1, color='k',zorder=0)
        ax.errorbar(xmed2,ymed2,xerr=xstd2, yerr=ystd2, color='k',zorder=0)
        ax.errorbar(xmed3,ymed3,xerr=xstd3, yerr=ystd3, color='k',zorder=0)
        ax.errorbar(xmed4,ymed4,xerr=xstd4, yerr=ystd4, color='k',zorder=0)
        ax.errorbar(xmed5,ymed5,xerr=xstd5, yerr=ystd5, color='k',zorder=0)

        ax.scatter(xmed1,ymed1,color=c1,marker='o',s=150,edgecolor='k',linewidth=2,rasterized=True)
        ax.scatter(xmed2,ymed2,color=c2,marker='o',s=150,edgecolor='k',linewidth=2,rasterized=True)
        ax.scatter(xmed3,ymed3,color=c3,marker='o',s=150,edgecolor='k',linewidth=2,rasterized=True)    
        ax.scatter(xmed4,ymed4,color=c4,marker='o',s=150,edgecolor='k',linewidth=2,rasterized=True)
        ax.scatter(xmed5,ymed5,color=c5,marker='o',s=150,edgecolor='k',linewidth=2,rasterized=True)

        ax.scatter(0.95,-0.72,marker='x',s=100,color='r',label='Elvis+94 Quasar')
        ax.plot(temp_opt3,temp_nir3,color='k',label='NGC 6090 mixing curve')
        ax.plot(red_opt,red_nir,color='r',label='Calzetti reddening curve')
        ax.arrow(0.95, -0.72, (-2.227-0.95), (-2.084+0.72), color='r',head_width=0.075, head_length=0.1)

        ax.set_xlim(-2.7,1.5)
        ax.set_ylim(-2.3,1.9)

        ax.set_xlabel(r'$\alpha_{\rm OPT}$')
        ax.set_ylabel(r'$\alpha_{\rm NIR}$')

        plt.legend()
        plt.grid()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def Upanels_ratio(self,savestring,X,Y,Median,x,uv,mir,fir,norm,shape,up_check,field):
        b1 = (shape == 1)
        b2 = (shape == 2)
        b3 = (shape == 3)
        b4 = (shape == 4)
        b5 = (shape == 5)

        if X == 'Lbol':
            xlabel = r'log $L_{\rm bol-gal,e}$'
            xunits = r'/(erg s$^{-1}$)'

        if Y == 'UV-MIR-FIR':
            ylabel1 = r'log L (UV)/$L_{\rm bol-gal,e}$'
            ylabel2 = r'log L (MIR)/$L_{\rm bol-gal,e}$'
            ylabel3 = r'log L (FIR)/$L_{\rm bol-gal,e}$'

        if X == 'Lbol' and Y == 'UV-MIR-FIR':
            y1 = uv - x 
            y2 = mir - x
            y3 = fir - x

        x1 = x[b1]
        x2 = x[b2]
        x3 = x[b3]
        x4 = x[b4]
        x5 = x[b5]

        y11 = y1[b1]
        y12 = y1[b2]
        y13 = y1[b3]
        y14 = y1[b4]
        y15 = y1[b5]

        y21 = y2[b1]
        y22 = y2[b2]
        y23 = y2[b3]
        y24 = y2[b4]
        y25 = y2[b5]

        y3[y3 > 0.05] = np.nan

        y31 = y3[b1]
        y32 = y3[b2]
        y33 = y3[b3]
        y34 = y3[b4]
        y35 = y3[b5]

        up_check1 = up_check[b1]
        up_check2 = up_check[b2]
        up_check3 = up_check[b3]
        up_check4 = up_check[b4]
        up_check5 = up_check[b5]

        field1 = field[b1]
        field2 = field[b2]
        field3 = field[b3]
        field4 = field[b4]
        field5 = field[b5]

        x1med = np.nanmedian(x1)
        x2med = np.nanmedian(x2)
        x3med = np.nanmedian(x3)
        x4med = np.nanmedian(x4)
        x5med = np.nanmedian(x5)

        y11med = np.nanmedian(y11)
        y12med = np.nanmedian(y12)
        y13med = np.nanmedian(y13)
        y14med = np.nanmedian(y14)
        y15med = np.nanmedian(y15)

        y21med = np.nanmedian(y21)
        y22med = np.nanmedian(y22)
        y23med = np.nanmedian(y23)
        y24med = np.nanmedian(y24)
        y25med = np.nanmedian(y25)

        y31med = np.nanmedian(y31)
        y32med = np.nanmedian(y32)
        y33med = np.nanmedian(y33)
        y34med = np.nanmedian(y34)
        y35med = np.nanmedian(y35)

        # Dispersion
        y11_25per, y21_25per, y31_25per, x1_25per = np.nanpercentile(y11,25), np.nanpercentile(y21, 25), np.nanpercentile(y31, 25), np.nanpercentile(x1,25)
        y12_25per, y22_25per, y32_25per, x2_25per = np.nanpercentile(y12,25), np.nanpercentile(y22, 25), np.nanpercentile(y32, 25), np.nanpercentile(x2,25)
        y13_25per, y23_25per, y33_25per, x3_25per = np.nanpercentile(y13,25), np.nanpercentile(y23, 25), np.nanpercentile(y33, 25), np.nanpercentile(x3,25)
        y14_25per, y24_25per, y34_25per, x4_25per = np.nanpercentile(y14,25), np.nanpercentile(y24, 25), np.nanpercentile(y34, 25), np.nanpercentile(x4,25)
        y15_25per, y25_25per, y35_25per, x5_25per = np.nanpercentile(y15,25), np.nanpercentile(y25, 25), np.nanpercentile(y35, 25), np.nanpercentile(x5,25)
        
        y11_75per, y21_75per, y31_75per, x1_75per = np.nanpercentile(y11,75), np.nanpercentile(y21, 75), np.nanpercentile(y31, 75), np.nanpercentile(x1,75)
        y12_75per, y22_75per, y32_75per, x2_75per = np.nanpercentile(y12,75), np.nanpercentile(y22, 75), np.nanpercentile(y32, 75), np.nanpercentile(x2,75)
        y13_75per, y23_75per, y33_75per, x3_75per = np.nanpercentile(y13,75), np.nanpercentile(y23, 75), np.nanpercentile(y33, 75), np.nanpercentile(x3,75)
        y14_75per, y24_75per, y34_75per, x4_75per = np.nanpercentile(y14,75), np.nanpercentile(y24, 75), np.nanpercentile(y34, 75), np.nanpercentile(x4,75)
        y15_75per, y25_75per, y35_75per, x5_75per = np.nanpercentile(y15,75), np.nanpercentile(y25, 75), np.nanpercentile(y35, 75), np.nanpercentile(x5,75)

        xmed = np.array([x1med,x2med,x3med,x4med,x5med])
        xerr1, xerr2 = xmed - np.array([x1_25per,x2_25per,x3_25per,x4_25per,x5_25per]),np.array([x1_75per,x2_75per,x3_75per,x4_75per,x5_75per]) - xmed
        y_uv_med = np.array([y11med,y12med,y13med,y14med,y15med])
        y_uv_err1, y_uv_err2 = y_uv_med - np.array([y11_25per,y12_25per,y13_25per, y14_25per,y15_25per]),np.array([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y_uv_med
        y_mir_med = np.array([y21med,y22med,y23med,y24med,y25med])
        y_mir_err1, y_mir_err2 = y_mir_med - np.array([y21_25per,y22_25per,y23_25per, y24_25per,y25_25per]),np.array([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y_mir_med
        y_fir_med = np.array([y31med,y32med,y33med,y34med,y35med]) 
        y_fir_err1, y_fir_err2 = y_fir_med - np.array([y31_25per,y32_25per,y33_25per, y34_25per,y35_25per]),np.array([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y_fir_med


        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        xlim = [43.5,47]
        ylim = [-3,0.25]

        fig = plt.figure(figsize=(10, 21))
        gs = fig.add_gridspec(ncols=1,nrows=3)#,left=0.06,right=0.95,top=0.86,bottom=0.14,wspace=0.0)
        ax1 = plt.subplot(gs[0], aspect='equal', adjustable='box')
        ax1.scatter(x1,y11,c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 1')
        ax1.scatter(x2,y12,c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 2')
        ax1.scatter(x3,y13,c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 3')
        ax1.scatter(x4,y14,c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 4')
        ax1.scatter(x5,y15,c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 5')

        ax1.errorbar(xmed, y_uv_med, xerr=[xerr1, xerr2], yerr=[y_uv_err1, y_uv_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
        ax1.scatter(x1med, y11med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x2med, y12med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x3med, y13med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x4med, y14med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax1.scatter(x5med, y15med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)

        ax2 = plt.subplot(gs[1], aspect='equal', adjustable='box')
        ax2.scatter(x1,y21,c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 1')
        ax2.scatter(x2,y22,c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 2')
        ax2.scatter(x3,y23,c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 3')
        ax2.scatter(x4,y24,c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 4')
        ax2.scatter(x5,y25,c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 5')

        ax2.errorbar(xmed, y_mir_med, xerr=[xerr1, xerr2], yerr=[y_mir_err1, y_mir_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
        ax2.scatter(x1med, y21med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x2med, y22med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x3med, y23med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x4med, y24med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
        ax2.scatter(x5med, y25med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)

        ax3 = plt.subplot(gs[2], aspect='equal', adjustable='box')
        if 'FIR' in Y:
            ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] == 'S82X'],y31[up_check1 == 1][field1[up_check1 == 1] == 'S82X'],c=c1,marker='x',rasterized=True,s=55,alpha=0.5)
            ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] == 'S82X'],y32[up_check2 == 1][field2[up_check2 == 1] == 'S82X'],c=c2,marker='x',rasterized=True,s=55,alpha=0.5)
            ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] == 'S82X'],y33[up_check3 == 1][field3[up_check3 == 1] == 'S82X'],c=c3,marker='x',rasterized=True,s=55,alpha=0.5)
            ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] == 'S82X'],y34[up_check4 == 1][field4[up_check4 == 1] == 'S82X'],c=c4,marker='x',rasterized=True,s=55,alpha=0.5)
            ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] == 'S82X'],y35[up_check5 == 1][field5[up_check5 == 1] == 'S82X'],c=c5,marker='x',rasterized=True,s=55,alpha=0.5)

            ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],y31[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],c=c1,marker=2,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],y32[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],c=c2,marker=2,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],y33[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],c=c3,marker=2,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],y34[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],c=c4,marker=2,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],y35[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],c=c5,marker=2,rasterized=True,s=55,alpha=0.65)

            ax3.scatter(x1[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],y31[up_check1 == 1][field1[up_check1 == 1] != 'S82X'],c=c1,marker=11,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x2[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],y32[up_check2 == 1][field2[up_check2 == 1] != 'S82X'],c=c2,marker=11,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x3[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],y33[up_check3 == 1][field3[up_check3 == 1] != 'S82X'],c=c3,marker=11,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x4[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],y34[up_check4 == 1][field4[up_check4 == 1] != 'S82X'],c=c4,marker=11,rasterized=True,s=55,alpha=0.65)
            ax3.scatter(x5[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],y35[up_check5 == 1][field5[up_check5 == 1] != 'S82X'],c=c5,marker=11,rasterized=True,s=55,alpha=0.65)

            ax3.scatter(x1[up_check1 == 0],y31[up_check1 == 0],c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 1')
            ax3.scatter(x2[up_check2 == 0],y32[up_check2 == 0],c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 2')
            ax3.scatter(x3[up_check3 == 0],y33[up_check3 == 0],c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 3')
            ax3.scatter(x4[up_check4 == 0],y34[up_check4 == 0],c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 4')
            ax3.scatter(x5[up_check5 == 0],y35[up_check5 == 0],c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 5')
            
        else:    
            ax3.scatter(x1,y31,c=c1,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 1')
            ax3.scatter(x2,y32,c=c2,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 2')
            ax3.scatter(x3,y33,c=c3,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 3')
            ax3.scatter(x4,y34,c=c4,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 4')
            ax3.scatter(x5,y35,c=c5,marker='P',lw=0,rasterized=True,s=55,alpha=0.8,label='Panel 5')

        if Median == 'Bins':
            ax3.errorbar(xmed, y_fir_med, xerr=[xerr1, xerr2], yerr=[y_fir_err1, y_fir_err2], mfc=c1, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
            ax3.scatter(x1med, y31med, color=c1, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
            ax3.scatter(x2med, y32med, color=c2, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
            ax3.scatter(x3med, y33med, color=c3, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
            ax3.scatter(x4med, y34med, color=c4, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)
            ax3.scatter(x5med, y35med, color=c5, marker='o', s = 150, edgecolor='k', linewidth=2, rasterized=True)

        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9,10,11,12,13])
        secax1.set_xlabel(xlabel+r'/L$_{\odot}$')

        ax1.set_xlim(xlim[0],xlim[1])
        ax2.set_xlim(xlim[0],xlim[1])
        ax3.set_xlim(xlim[0],xlim[1])

        ax1.set_ylim(ylim[0],ylim[1])
        ax2.set_ylim(ylim[0],ylim[1])
        ax3.set_ylim(ylim[0],ylim[1])

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

        ax3.set_xlabel(xlabel+xunits)
        ax1.set_ylabel(ylabel1)
        ax2.set_ylabel(ylabel2)
        ax3.set_ylabel(ylabel3)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax2.legend()
        
        plt.tight_layout()
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