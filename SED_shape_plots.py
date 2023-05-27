import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

from SED_plots_v2 import Plotter

class SED_shape_Plotter(Plotter):
    def __init__(self, ID, z, wavelength, Lum, L, norm, up_check, shape):
        super().__init__(ID, z, wavelength, Lum, L, norm, up_check)
        self.shape = shape

        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 3
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 4
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 4
        plt.rcParams['hatch.linewidth'] = 2.0

    def solar_log(self, x):
        return x - np.log10(3.8E33)

    def ergs_log(self, x):
        return x + np.log10(3.8E33)

    def shape_1bin_h(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], uv_slope=None, mir_slope1=None, mir_slope2=None, Median_line=True, FIR_med=True, FIR_upper='upper lims', bins='shape'):
        '''Function to plot the 5 sed shapes in a single horizonal plot. This creates a 1 row x 5 column plot'''
        
        b1 = self.shape == 1
        b2 = self.shape == 2
        b3 = self.shape == 3
        b4 = self.shape == 4
        b5 = self.shape == 5

        # b1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
        # b2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
        # b3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
        # b4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
        # b5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

        if len(self.norm) == len(ffir):
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

        wfir_seg = np.delete(wfir, 0, 1)
        ffir_seg = np.delete(ffir_norm, 0, 1)

        x1, x2, x3, x4, x5 = x[b1], x[b2], x[b3], x[b4], x[b5]
        y1, y2, y3, y4, y5 = y[b1], y[b2], y[b3], y[b4], y[b5]
        L1, L2, L3, L4, L5 = L[b1], L[b2], L[b3], L[b4], L[b5]
        wfir1, wfir2, wfir3, wfir4, wfir5 = wfir[b1], wfir[b2], wfir[b3], wfir[b4], wfir[b5]
        ffir1, ffir2, ffir3, ffir4, ffir5 = ffir_norm[b1], ffir_norm[b2], ffir_norm[b3], ffir_norm[b4], ffir_norm[b5]
        wfir1_seg, wfir2_seg, wfir3_seg, wfir4_seg, wfir5_seg = wfir_seg[b1], wfir_seg[b2], wfir_seg[b3], wfir_seg[b4], wfir_seg[b5]
        ffir1_seg, ffir2_seg, ffir3_seg, ffir4_seg, ffir5_seg = ffir_seg[b1], ffir_seg[b2], ffir_seg[b3], ffir_seg[b4], ffir_seg[b5]
        median_x1, median_x2, median_x3, median_x4, median_x5 = median_x[b1], median_x[b2], median_x[b3], median_x[b4], median_x[b5]
        median_y1, median_y2, median_y3, median_y4, median_y5 = median_y[b1], median_y[b2], median_y[b3], median_y[b4], median_y[b5]
        # xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
        # yticks = [0.001,0.01,0.1,1,10,100]
        # xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

        xticks = [1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100]
        # xticks = [1E-4, 1E-2, 1, 100]
        yticks = [1E-2, 0.1, 1, 10]

        xticks_labels = ['-4', ' ', '-2', ' ', '0', ' ', '2']
        # xticks_labels = ['-4', '-2', '0', '2']
        yticklabels = ['-2', '-1', '0',' 1']

        fig = plt.figure(figsize=(30,12))
        gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1,top=0.7,bottom=0.3,right=0.9,wspace=0.05,hspace=0.05,height_ratios=[0.2,3])
        # gs = fig.add_gridspec(nrows=2, ncols=5,wspace=0.05,top=0.7,bottom=0.2,hspace=-0.5,height_ratios=[0.2,3])

        ax1 = fig.add_subplot(gs[1, 4])#, aspect='equal', adjustable='box')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([])
        ax1.set_xticklabels(xticks_labels)
        ax1.text(0.05, 0.8, f'n = {len(x1)}', transform=ax1.transAxes)
        ax1.text(0.73, 0.08, str((len(x1)/len(x))*100)[0:4]+'%', transform=ax1.transAxes, weight='bold')
        ax1.grid()

        # Plot data
        upper_seg1 = np.stack((wfir1_seg,ffir1_seg), axis=2)
        upper_all1 = LineCollection(upper_seg1, color='gray', alpha=0.3)
        ax1.add_collection(upper_all1)
        lc1 = self.multilines(x1,y1,L1,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb1 = fig.colorbar(lc1, orientation='horizontal', pad=-0.1)
        axcb1.mappable.set_clim(clim1, clim2)
        axcb1.remove()
        # Plot median line
        if Median_line:
            ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=True, connect_point=True, Bin=True, bin_in=b1, lw=3)
                self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b1, lw=3,ls='--')
            else:
                self.median_sed(median_x1, median_y1,Bin=True, bin_in = b1, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax2 = fig.add_subplot(gs[1, 3])#, aspect='equal', adjustable='box')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([])
        ax2.set_xticklabels(xticks_labels)
        ax2.text(0.05, 0.8, f'n = {len(x2)}', transform=ax2.transAxes)
        ax2.text(0.73, 0.08, str((len(x2)/len(x))*100)[0:4]+'%', transform=ax2.transAxes, weight='bold')
        ax2.grid()

        # Plot data
        upper_seg2 = np.stack((wfir2_seg,ffir2_seg), axis=2)
        upper_all2 = LineCollection(upper_seg2, color='gray', alpha=0.3)
        ax2.add_collection(upper_all2)
        lc2 = self.multilines(x2,y2,L2,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb2 = fig.colorbar(lc2,orientation='horizontal',pad=-0.1)
        axcb2.mappable.set_clim(clim1, clim2)
        axcb2.remove()
        # Plot median line
        if Median_line:
            ax2.plot(np.nanmedian(x2[:, :2], axis=0),np.nanmedian(y2[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x2, median_y2, Norm=True, connect_point=True, Bin=True, bin_in=b2, lw=3)
                self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b2, lw=3,ls='--')
            else:
                self.median_sed(median_x2, median_y2,Bin=True, bin_in = b2, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax3 = fig.add_subplot(gs[1, 2])#, aspect='equal', adjustable='box')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xticks(xticks)
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([])
        ax3.set_xticklabels(xticks_labels)
        ax3.set_xlabel(r'log ($\lambda_{\rm rest}/\mu$m)')
        # ax3.set_ylabel(r'Log $\lambda L_{\lambda}$ Normalized at 1$\mu$m')
        ax3.text(0.05, 0.8, f'n = {len(x3)}', transform=ax3.transAxes)
        ax3.text(0.73, 0.08, str((len(x3)/len(x))*100)[0:4]+'%', transform=ax3.transAxes, weight='bold')
        ax3.grid()

        # Plot data
        upper_seg3 = np.stack((wfir3_seg,ffir3_seg), axis=2)
        upper_all3 = LineCollection(upper_seg3, color='gray', alpha=0.3)
        ax3.add_collection(upper_all3)
        lc3 = self.multilines(x3,y3,L3,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb3 = fig.colorbar(lc3,orientation='horizontal',pad=-0.1)
        axcb3.mappable.set_clim(clim1, clim2)
        axcb3.remove()
        # Plot median line
        if Median_line:
            ax3.plot(np.nanmedian(x3[:, :2], axis=0),np.nanmedian(y3[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x3, median_y3, Norm=True, connect_point=True, Bin=True, bin_in=b3, lw=3)
                self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b3, lw=3,ls='--')
            else:
                self.median_sed(median_x3, median_y3, Bin=True, bin_in = b3, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax4 = fig.add_subplot(gs[1, 1])#, aspect='equal', adjustable='box')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xticks(xticks)
        ax4.set_yticks(yticks)
        ax4.set_yticklabels([])
        ax4.set_xticklabels(xticks_labels)
        ax4.text(0.05, 0.8, f'n = {len(x4)}', transform=ax4.transAxes)
        ax4.text(0.73, 0.08, str((len(x4)/len(x))*100)[0:4]+'%', transform=ax4.transAxes, weight='bold')
        ax4.grid()

        # Plot data
        upper_seg4 = np.stack((wfir4_seg,ffir4_seg), axis=2)
        upper_all4 = LineCollection(upper_seg4, color='gray', alpha=0.3)
        ax4.add_collection(upper_all4)
        lc4 = self.multilines(x4,y4,L4,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb4 = fig.colorbar(lc4,orientation='horizontal',pad=-0.1)
        axcb4.mappable.set_clim(clim1, clim2)
        axcb4.remove()
        # Plot median line
        if Median_line:
            ax4.plot(np.nanmedian(x4[:, :2], axis=0),np.nanmedian(y4[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x4, median_y4, Norm=True, connect_point=True, Bin=True, bin_in=b4, lw=3)
                self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b4, lw=3,ls='--')
            else:
                self.median_sed(median_x4, median_y4, Bin=True, bin_in = b4, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax5 = fig.add_subplot(gs[1, 0])#, aspect='equal', adjustable='box')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.set_xticks(xticks)
        ax5.set_yticks(yticks)
        ax5.set_xticklabels(xticks_labels)
        ax5.text(0.05, 0.8, f'n = {len(x5)}', transform=ax5.transAxes)
        ax5.text(0.73, 0.08, str((len(x5)/len(x))*100)[0:4]+'%', transform=ax5.transAxes, weight='bold')
        ax5.set_ylabel(r'Normalized log ($\lambda$ L$_\lambda$)')
        ax5.grid()

        # Plot data
        upper_seg5 = np.stack((wfir5_seg,ffir5_seg), axis=2)
        upper_all5 = LineCollection(upper_seg5, color='gray', alpha=0.3)
        ax5.add_collection(upper_all5)
        lc5 = self.multilines(x5,y5,L5,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb5 = fig.colorbar(lc5,orientation='horizontal',pad=-0.1)
        axcb5.mappable.set_clim(clim1, clim2)
        axcb5.remove()
        # Plot median line
        if Median_line:
            ax5.plot(np.nanmedian(x5[:, :2], axis=0),np.nanmedian(y5[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x5, median_y5, Norm=True, connect_point=True, Bin=True, bin_in=b5, lw=3)
                self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b5, lw=3,ls='--')
            else:
                self.median_sed(median_x5, median_y5, Bin=True, bin_in = b5, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        cbar_ax = fig.add_subplot(gs[:-1, :])
        cb = fig.colorbar(lc1, cax=cbar_ax, orientation='horizontal')
        cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')
        cb.mappable.set_clim(clim1, clim2)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def shape_1bin_v(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], uv_slope=None, mir_slope1=None, mir_slope2=None, Median_line=True, FIR_med=True, FIR_upper='upper lims', bins='shape'):
        '''Function to plot the 5 sed shapes in a single horizonal plot. This creates a 1 row x 5 column plot'''
        # plt.rcParams['font.size'] = 40
        # plt.rcParams['axes.linewidth'] = 4
        # plt.rcParams['xtick.major.size'] = 6
        # plt.rcParams['xtick.major.width'] = 5
        # plt.rcParams['ytick.major.size'] = 6
        # plt.rcParams['ytick.major.width'] = 5
        
        if bins == 'shape':
            b1 = self.shape == 1
            b2 = self.shape == 2
            b3 = self.shape == 3
            b4 = self.shape == 4
            b5 = self.shape == 5

        elif bins == 'Lx':
            b5 = self.L < 43.5
            b4 = (self.L > 43.5) & (self.L < 44)
            b3 = (self.L > 44) & (self.L < 44.5)
            b2 = (self.L > 44.5) & (self.L < 45)
            b1 = self.L > 45

        else:
            print('Invalid bins option. Options are:   shape,    Lx')
            return

        # b1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
        # b2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
        # b3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
        # b4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
        # b5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

        if len(self.norm) == len(ffir):
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

        wfir_seg = np.delete(wfir, 0, 1)
        ffir_seg = np.delete(ffir_norm, 0, 1)

        x1, x2, x3, x4, x5 = x[b1], x[b2], x[b3], x[b4], x[b5]
        y1, y2, y3, y4, y5 = y[b1], y[b2], y[b3], y[b4], y[b5]
        L1, L2, L3, L4, L5 = L[b1], L[b2], L[b3], L[b4], L[b5]
        wfir1, wfir2, wfir3, wfir4, wfir5 = wfir[b1], wfir[b2], wfir[b3], wfir[b4], wfir[b5]
        ffir1, ffir2, ffir3, ffir4, ffir5 = ffir_norm[b1], ffir_norm[b2], ffir_norm[b3], ffir_norm[b4], ffir_norm[b5]
        wfir1_seg, wfir2_seg, wfir3_seg, wfir4_seg, wfir5_seg = wfir_seg[b1], wfir_seg[b2], wfir_seg[b3], wfir_seg[b4], wfir_seg[b5]
        ffir1_seg, ffir2_seg, ffir3_seg, ffir4_seg, ffir5_seg = ffir_seg[b1], ffir_seg[b2], ffir_seg[b3], ffir_seg[b4], ffir_seg[b5]
        median_x1, median_x2, median_x3, median_x4, median_x5 = median_x[b1], median_x[b2], median_x[b3], median_x[b4], median_x[b5]
        median_y1, median_y2, median_y3, median_y4, median_y5 = median_y[b1], median_y[b2], median_y[b3], median_y[b4], median_y[b5]
        # xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
        # yticks = [0.001,0.01,0.1,1,10,100]
        # xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

        xticks = [1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100]
        yticks = [1E-2, 0.1, 1, 10]

        xticks_labels = ['-4', ' ', '-2', ' ', '0', ' ', '2']
        yticklabels = [-2, -1, 0, 1]

        fig = plt.figure(figsize=(16,35))
        gs = fig.add_gridspec(nrows=5, ncols=2, left=0.2, right=0.75, hspace=0.05,wspace=-0.05,width_ratios=[3,0.25])

        ax1 = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='box')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticklabels)
        ax1.set_xticklabels([])
        ax1.text(0.05, 0.72, f'n = {len(x1)}', transform=ax1.transAxes)
        ax1.text(0.05, 0.835, '1', transform=ax1.transAxes, fontsize=40, weight='bold')
        ax1.text(0.73, 0.08, str((len(x1)/len(x))*100)[0:4]+'%', transform=ax1.transAxes, weight='bold')
        ax1.grid()

        # Plot data
        upper_seg1 = np.stack((wfir1_seg,ffir1_seg), axis=2)
        upper_all1 = LineCollection(upper_seg1, color='gray', alpha=0.3)
        ax1.add_collection(upper_all1)
        lc1 = self.multilines(x1,y1,L1,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb1 = fig.colorbar(lc1, orientation='vertical', pad=-0.1)
        axcb1.mappable.set_clim(clim1, clim2)
        axcb1.remove()
        # Plot median line
        if Median_line:
            ax1.plot(np.nanmedian(x1[:, :2], axis=0),np.nanmedian(y1[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=True, connect_point=True, Bin=True, bin_in=b1, lw=3)
                self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b1, lw=3,ls='--')
                
                # x_connect, y_connect = self.median_sed(median_x1, median_y1, Norm=True,connect_point=True,lw=4)
                # self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False,lw=4,ls='--')
            else:
                self.median_sed(median_x1, median_y1,Bin=True, bin_in = b1, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax2 = fig.add_subplot(gs[1, 0], aspect='equal', adjustable='box')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xticks(xticks)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticklabels)
        ax2.set_xticklabels([])
        ax2.text(0.05, 0.72, f'n = {len(x2)}', transform=ax2.transAxes)
        ax2.text(0.05, 0.835, '2', transform=ax2.transAxes, fontsize=40, weight='bold')
        ax2.text(0.73, 0.08, str((len(x2)/len(x))*100)[0:4]+'%', transform=ax2.transAxes, weight='bold')
        ax2.grid()

        # Plot data
        upper_seg2 = np.stack((wfir2_seg,ffir2_seg), axis=2)
        upper_all2 = LineCollection(upper_seg2, color='gray', alpha=0.3)
        ax2.add_collection(upper_all2)
        lc2 = self.multilines(x2,y2,L2,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb2 = fig.colorbar(lc2,orientation='vertical',pad=-0.1)
        axcb2.mappable.set_clim(clim1, clim2)
        axcb2.remove()
        # Plot median line
        if Median_line:
            ax2.plot(np.nanmedian(x2[:, :2], axis=0),np.nanmedian(y2[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x2, median_y2, Norm=True, connect_point=True, Bin=True, bin_in=b2, lw=3)
                self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b2, lw=3,ls='--')
            else:
                self.median_sed(median_x2, median_y2,Bin=True, bin_in = b2, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax3 = fig.add_subplot(gs[2, 0], aspect='equal', adjustable='box')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xticks(xticks)
        ax3.set_yticks(yticks)
        ax3.set_yticklabels(yticklabels)
        ax3.set_xticklabels([])
        ax3.set_ylabel(r'Normalized log ($\lambda$ L$_\lambda$)')
        ax3.text(0.05, 0.72, f'n = {len(x3)}', transform=ax3.transAxes)
        ax3.text(0.05, 0.835, '3', transform=ax3.transAxes, fontsize=40, weight='bold')
        ax3.text(0.73, 0.08, str((len(x3)/len(x))*100)[0:4]+'%', transform=ax3.transAxes, weight='bold')
        ax3.grid()

        # Plot data
        upper_seg3 = np.stack((wfir3_seg,ffir3_seg), axis=2)
        upper_all3 = LineCollection(upper_seg3, color='gray', alpha=0.3)
        ax3.add_collection(upper_all3)
        lc3 = self.multilines(x3,y3,L3,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb3 = fig.colorbar(lc3,orientation='vertical',pad=-0.1)
        axcb3.mappable.set_clim(clim1, clim2)
        axcb3.remove()
        # Plot median line
        if Median_line:
            ax3.plot(np.nanmedian(x3[:, :2], axis=0),np.nanmedian(y3[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x3, median_y3, Norm=True, connect_point=True, Bin=True, bin_in=b3, lw=3)
                self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b3, lw=3,ls='--')
            else:
                self.median_sed(median_x3, median_y3, Bin=True, bin_in = b3, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax4 = fig.add_subplot(gs[3, 0], aspect='equal', adjustable='box')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xticks(xticks)
        ax4.set_yticks(yticks)
        ax4.set_yticklabels(yticklabels)
        ax4.set_xticklabels([])
        ax4.text(0.05, 0.72, f'n = {len(x4)}', transform=ax4.transAxes)
        ax4.text(0.05, 0.835, '4', transform=ax4.transAxes, fontsize=40, weight='bold')
        ax4.text(0.73, 0.08, str((len(x4)/len(x))*100)[0:4]+'%', transform=ax4.transAxes, weight='bold')
        ax4.grid()

        # Plot data
        upper_seg4 = np.stack((wfir4_seg,ffir4_seg), axis=2)
        upper_all4 = LineCollection(upper_seg4, color='gray', alpha=0.3)
        ax4.add_collection(upper_all4)
        lc4 = self.multilines(x4,y4,L4,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb4 = fig.colorbar(lc4,orientation='vertical',pad=-0.1)
        axcb4.mappable.set_clim(clim1, clim2)
        axcb4.remove()
        # Plot median line
        if Median_line:
            ax4.plot(np.nanmedian(x4[:, :2], axis=0),np.nanmedian(y4[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x4, median_y4, Norm=True, connect_point=True, Bin=True, bin_in=b4, lw=3)
                self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b4, lw=3,ls='--')
            else:
                self.median_sed(median_x4, median_y4, Bin=True, bin_in = b4, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        ax5 = fig.add_subplot(gs[4, 0], aspect='equal', adjustable='box')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.set_xticks(xticks)
        ax5.set_yticks(yticks)
        ax5.set_yticklabels(yticklabels)
        ax5.set_xticklabels(xticks_labels)
        ax5.set_xlabel(r'log ($\lambda_{\rm rest}/\mu$m)')
        ax5.text(0.05, 0.72, f'n = {len(x5)}', transform=ax5.transAxes)
        ax5.text(0.05, 0.835, '5', transform=ax5.transAxes, fontsize=40, weight='bold')
        ax5.text(0.73, 0.08, str((len(x5)/len(x))*100)[0:4]+'%', transform=ax5.transAxes, weight='bold')
        ax5.grid()

        # Plot data
        upper_seg5 = np.stack((wfir5_seg,ffir5_seg), axis=2)
        upper_all5 = LineCollection(upper_seg5, color='gray', alpha=0.3)
        ax5.add_collection(upper_all5)
        lc5 = self.multilines(x5,y5,L5,cmap=cmap,lw=1.5,alpha=0.7,rasterized=True)
        axcb5 = fig.colorbar(lc5,orientation='vertical',pad=-0.1)
        axcb5.mappable.set_clim(clim1, clim2)
        axcb5.remove()
        # Plot median line
        if Median_line:
            ax5.plot(np.nanmedian(x5[:, :2], axis=0),np.nanmedian(y5[:, :2], axis=0), c='k', lw=3)
            if FIR_med:
                x_connect, y_connect = self.median_sed(median_x5, median_y5, Norm=True, connect_point=True, Bin=True, bin_in=b5, lw=3)
                self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b5, lw=3,ls='--')
            else:
                self.median_sed(median_x5, median_y5, Bin=True, bin_in = b5, lw=3)
        plt.ylim(1E-3, 75)
        plt.xlim(8E-5, 7E2)

        cbar_ax = fig.add_subplot(gs[:,-1:])
        cb = fig.colorbar(lc1, cax=cbar_ax, orientation='vertical')
        cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')
        cb.mappable.set_clim(clim1, clim2)
        # cb.ax.xaxis.set_ticks_position('right')
        # cb.ax.xaxis.set_label_position('right')

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()
    
    def L_hist_bins(self,savestring,x,xlabel=None,xlim=[np.nan,np.nan],hist_bins=[np.nan,np.nan,np.nan],median=True,std=False,bins='shape'):
        if bins == 'shape':
            b1 = self.shape == 1
            b2 = self.shape == 2
            b3 = self.shape == 3
            b4 = self.shape == 4
            b5 = self.shape == 5

            bin1_name = 'Panel 1'
            bin2_name = 'Panel 2'
            bin3_name = 'Panel 3'
            bin4_name = 'Panel 4'
            bin5_name = 'Panel 5'

        elif bins == 'Lx':
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

        else:
            print('Invalid bins option. Options are:   shape,    Lx')
            return

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        plt.figure(figsize=(9,9))
        n5 = plt.hist(x[b5], bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),histtype='step',color=c5,lw=4,alpha=0.8,label=bin5_name)
        n4 = plt.hist(x[b4], bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),histtype='step',color=c4,lw=4,alpha=0.8,label=bin4_name)
        n3 = plt.hist(x[b3], bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),histtype='step',color=c3,lw=4,alpha=0.8,label=bin3_name)        
        n2 = plt.hist(x[b2], bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),histtype='step',color=c2,lw=4,alpha=0.8,label=bin2_name)        
        n1 = plt.hist(x[b1], bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),histtype='step',color=c1,lw=4,alpha=0.8,label=bin1_name)

        n = np.append(n1[0],n2[0])
        n = np.append(n,n3[0])
        n = np.append(n,n4[0])
        n = np.append(n,n5[0])
        if median:
            plt.axvline(np.nanmean(x[b1]), color=c1, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b2]), color=c2, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b3]), color=c3, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b4]), color=c4, ls='--', lw=3)
            plt.axvline(np.nanmean(x[b5]), color=c5, ls='--', lw=3)
        plt.xlabel(xlabel)
        plt.grid()
        plt.ylim(0,max(n)+max(n)*0.1)
        plt.legend(fontsize=15)

        print('bin 1: ',np.nanmean(x[b1]))
        print('bin 2: ',np.nanmean(x[b2]))
        print('bin 3: ',np.nanmean(x[b3]))
        print('bin 4: ',np.nanmean(x[b4]))
        print('bin 5: ',np.nanmean(x[b5]))

        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_hist_panels(self, savestring, x, xlabel=None, xlim=[np.nan, np.nan],hist_bins=[np.nan, np.nan], median=True,std=False,bins='shape',split=False,split_param=None,z_label=False,top_label=False,xlabel2=None):

        if bins == 'shape':
            b1 = self.shape == 1
            b2 = self.shape == 2
            b3 = self.shape == 3
            b4 = self.shape == 4
            b5 = self.shape == 5

            bin1_name = 'Panel 1'
            bin2_name = 'Panel 2'
            bin3_name = 'Panel 3'
            bin4_name = 'Panel 4'
            bin5_name = 'Panel 5'

        elif bins == 'Lx':
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

        else:
            print('Invalid bins option. Options are:   shape,    Lx')
            return

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        x1 = x[b1]
        x2 = x[b2]
        x3 = x[b3]
        x4 = x[b4]
        x5 = x[b5]

        hist1 = np.ones(np.shape(x),dtype=bool)
        hist2 = np.zeros(np.shape(x),dtype=bool)


        if split:
            hist1 = (split_param != 0)
            hist2 = (split_param == 0)

        hist11 = hist1[b1]
        hist12 = hist1[b2]
        hist13 = hist1[b3]
        hist14 = hist1[b4]
        hist15 = hist1[b5]

        hist21 = hist2[b1]
        hist22 = hist2[b2]
        hist23 = hist2[b3]
        hist24 = hist2[b4]
        hist25 = hist2[b5]

        if z_label:
            z1 = self.z[b1]
            z2 = self.z[b2]
            z3 = self.z[b3]
            z4 = self.z[b4]
            z5 = self.z[b5]

            zlabel1 = r'z$_{\rm med}$ = '+str(np.nanmedian(z1).round(decimals=2))
            zlabel2 = r'z$_{\rm med}$ = '+str(np.nanmedian(z2).round(decimals=2))
            zlabel3 = r'z$_{\rm med}$ = '+str(np.nanmedian(z3).round(decimals=2))
            zlabel4 = r'z$_{\rm med}$ = '+str(np.nanmedian(z4).round(decimals=2))
            zlabel5 = r'z$_{\rm med}$ = '+str(np.nanmedian(z5).round(decimals=2))

        else:
            zlabel1 = ''
            zlabel2 = ''
            zlabel3 = ''
            zlabel4 = ''
            zlabel5 = ''

        fig = plt.figure(figsize=(15,15))
        gs = fig.add_gridspec(nrows=5, ncols=1, left=0.2, right=0.75, hspace=0.1,wspace=-0.05)

        ax1 = fig.add_subplot(gs[0, 0])
        n1 = ax1.hist(x1, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c1,lw=4,alpha=0.8,label=bin1_name)
        ax1.hist(x1, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin1_name)
        if len(x1[hist21]) > 0:
            ax1.hist(x1[hist21],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax2 = fig.add_subplot(gs[1, 0])
        n2 = ax2.hist(x2, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c2,lw=4,alpha=0.8,label=bin2_name)
        ax2.hist(x2, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin2_name)        
        if len(x2[hist22]) > 0:
            ax2.hist(x2[hist22],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax3 = fig.add_subplot(gs[2, 0])
        n3 = ax3.hist(x3, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c3,lw=4,alpha=0.8,label=bin3_name)
        ax3.hist(x3, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin3_name)
        if len(x3[hist23]) > 0:
            ax3.hist(x3[hist23],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax4 = fig.add_subplot(gs[3, 0])
        n4 = ax4.hist(x4, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c4,lw=4,alpha=0.8,label=bin4_name)
        ax4.hist(x4, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin4_name)
        if len(x4[hist24]) > 0:
            ax4.hist(x4[hist24],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax5 = fig.add_subplot(gs[4, 0])
        n5 = ax5.hist(x5, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c5,lw=4,alpha=0.8,label=bin5_name)
        ax5.hist(x5, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin5_name)
        if len(x5[hist25]) > 0:
            ax5.hist(x5[hist25],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        n = np.append(n1[0], n2[0])
        n = np.append(n, n3[0])
        n = np.append(n, n4[0])
        n = np.append(n, n5[0])
        
        y_lim = np.linspace(0,max(n)+8,4)
        y_lim = np.asarray(y_lim, dtype=int)
        if xlabel == r'log N$_{\rm H}$':
            x_lim = np.arange(xlim[0],xlim[1],1)
        else:
            x_lim = np.arange(xlim[0],xlim[1],0.5)

        ax1.set_yticks(y_lim)
        ax2.set_yticks(y_lim)
        ax3.set_yticks(y_lim)
        ax4.set_yticks(y_lim)
        ax5.set_yticks(y_lim)

        ax1.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 1')
        ax1.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel1)
        ax2.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 2')
        ax2.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel2)
        ax3.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 3')
        ax3.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel3)
        ax4.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 4')  
        ax4.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel4)
        ax5.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 5')
        ax5.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel5)

        # if xlabel == r'log L$_{\rm X}$' or xlabel == r'log L$_{\rm bol}$':
        #     secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_xticks([9,10,11,12])
        #     secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')

        # elif xlabel == r'log L (1$\mu \rm m)$':
        #     secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_xticks([9, 10, 11, 12])
        #     secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')

        if top_label:
            secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
            secax1.set_xticks([9, 10, 11, 12])
            secax1.set_xlabel(xlabel2)

        
        if median:
            ax1.axvline(np.nanmedian(x1[np.isfinite(x1)]), color='k', ls='--', lw=3.5)
            ax2.axvline(np.nanmedian(x2[np.isfinite(x2)]), color='k', ls='--', lw=3.5)
            ax3.axvline(np.nanmedian(x3[np.isfinite(x3)]), color='k', ls='--', lw=3.5)
            ax4.axvline(np.nanmedian(x4[np.isfinite(x4)]), color='k', ls='--', lw=3.5)
            ax5.axvline(np.nanmedian(x5[np.isfinite(x5)]), color='k', ls='--', lw=3.5)

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax1.set_xticks(x_lim)
        ax2.set_xticks(x_lim)
        ax3.set_xticks(x_lim)
        ax4.set_xticks(x_lim)
        ax5.set_xticks(x_lim)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()

        # if xlabel != r'log N$_{\rm H}$':
        #     ax5.set_xlabel(xlabel+' [erg/s]')
        # elif xlabel == r'log N$_{\rm H}$':
        #     ax5.set_xlabel(xlabel+r' [cm$^{-2}$]')
        # else:
        ax5.set_xlabel(xlabel)

        ax1.set_ylim(0, max(n)+max(n)*0.1)
        ax2.set_ylim(0, max(n)+max(n)*0.1)
        ax3.set_ylim(0, max(n)+max(n)*0.1)
        ax4.set_ylim(0, max(n)+max(n)*0.1)
        ax5.set_ylim(0, max(n)+max(n)*0.1)

        s = x1.sort()
        print('        Median,     max,     min,    mean')
        print('panel 1: ',np.nanmedian(x1[np.isfinite(x1)]),max(x1[np.isfinite(x1)]),min(x1[np.isfinite(x1)]),np.nanmean(x1[np.isfinite(x1)]))
        print('panel 2: ',np.nanmedian(x2[np.isfinite(x2)]),max(x2[np.isfinite(x2)]),min(x2[np.isfinite(x2)]),np.nanmean(x2[np.isfinite(x2)]))
        print('panel 3: ',np.nanmedian(x3[np.isfinite(x3)]),max(x3[np.isfinite(x3)]),min(x3[np.isfinite(x3)]),np.nanmean(x3[np.isfinite(x3)]))
        print('panel 4: ',np.nanmedian(x4[np.isfinite(x4)]),max(x4[np.isfinite(x4)]),min(x4[np.isfinite(x4)]),np.nanmean(x4[np.isfinite(x4)]))
        print('panel 5: ',np.nanmedian(x5[np.isfinite(x5)]),max(x5[np.isfinite(x5)]),min(x5[np.isfinite(x5)]),np.nanmean(x5[np.isfinite(x5)]))

        plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()

    def L_hist_panels2(self, savestring, x, y, xlabel=None, xlim=[np.nan, np.nan],hist_bins=[np.nan, np.nan],xlim2=[np.nan,np.nan],hist_bins2=[np.nan,np.nan], median=True,std=False,bins='shape',split=False,split_param=None,z_label=False,top_label=False,xlabel2=None):

        if bins == 'shape':
            b1 = self.shape == 1
            b2 = self.shape == 2
            b3 = self.shape == 3
            b4 = self.shape == 4
            b5 = self.shape == 5

            bin1_name = 'Panel 1'
            bin2_name = 'Panel 2'
            bin3_name = 'Panel 3'
            bin4_name = 'Panel 4'
            bin5_name = 'Panel 5'

        elif bins == 'Lx':
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

        else:
            print('Invalid bins option. Options are:   shape,    Lx')
            return

        c1 = '#377eb8'
        c2 = '#984ea3'
        c3 = '#4daf4a'
        c4 = '#ff7f00'
        c5 = '#e41a1c'

        x1 = x[b1]
        x2 = x[b2]
        x3 = x[b3]
        x4 = x[b4]
        x5 = x[b5]

        x6 = y[b1]
        x7 = y[b2]
        x8 = y[b3]
        x9 = y[b4]
        x10 = y[b5]

        print(y)

        hist1 = np.ones(np.shape(x),dtype=bool)
        hist2 = np.zeros(np.shape(x),dtype=bool)


        if split:
            hist1 = (split_param != 1)
            hist2 = (split_param == 1)

        hist11 = hist1[b1]
        hist12 = hist1[b2]
        hist13 = hist1[b3]
        hist14 = hist1[b4]
        hist15 = hist1[b5]

        hist21 = hist2[b1]
        hist22 = hist2[b2]
        hist23 = hist2[b3]
        hist24 = hist2[b4]
        hist25 = hist2[b5]

        if z_label:
            z1 = self.z[b1]
            z2 = self.z[b2]
            z3 = self.z[b3]
            z4 = self.z[b4]
            z5 = self.z[b5]

            zlabel1 = r'z$_{\rm med}$ = '+str(np.nanmedian(z1).round(decimals=2))
            zlabel2 = r'z$_{\rm med}$ = '+str(np.nanmedian(z2).round(decimals=2))
            zlabel3 = r'z$_{\rm med}$ = '+str(np.nanmedian(z3).round(decimals=2))
            zlabel4 = r'z$_{\rm med}$ = '+str(np.nanmedian(z4).round(decimals=2))
            zlabel5 = r'z$_{\rm med}$ = '+str(np.nanmedian(z5).round(decimals=2))

        else:
            zlabel1 = ''
            zlabel2 = ''
            zlabel3 = ''
            zlabel4 = ''
            zlabel5 = ''

        fig = plt.figure(figsize=(20,15))
        gs = fig.add_gridspec(nrows=5, ncols=2)# , left=0.2, right=0.75, hspace=0.1,wspace=-0.05)

        ax1 = fig.add_subplot(gs[0, 0])
        n1 = ax1.hist(x1, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c1,lw=4,alpha=0.8,label=bin1_name)
        ax1.hist(x1, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin1_name)
        if len(x1[hist21]) > 0:
            ax1.hist(x1[hist21],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax2 = fig.add_subplot(gs[1, 0])
        n2 = ax2.hist(x2, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c2,lw=4,alpha=0.8,label=bin2_name)
        ax2.hist(x2, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin2_name)        
        if len(x2[hist22]) > 0:
            ax2.hist(x2[hist22],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax3 = fig.add_subplot(gs[2, 0])
        n3 = ax3.hist(x3, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c3,lw=4,alpha=0.8,label=bin3_name)
        ax3.hist(x3, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin3_name)
        if len(x3[hist23]) > 0:
            ax3.hist(x3[hist23],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax4 = fig.add_subplot(gs[3, 0])
        n4 = ax4.hist(x4, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c4,lw=4,alpha=0.8,label=bin4_name)
        ax4.hist(x4, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin4_name)
        if len(x4[hist24]) > 0:
            ax4.hist(x4[hist24],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax5 = fig.add_subplot(gs[4, 0])
        n5 = ax5.hist(x5, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color=c5,lw=4,alpha=0.8,label=bin5_name)
        ax5.hist(x5, bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin5_name)
        if len(x5[hist25]) > 0:
            ax5.hist(x5[hist25],bins=np.arange(hist_bins[0],hist_bins[1],hist_bins[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        
        ax6 = fig.add_subplot(gs[0, 1])
        n6 = ax6.hist(x6, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color=c1,lw=4,alpha=0.8,label=bin1_name)
        ax6.hist(x6, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin1_name)
        if len(x6[hist21]) > 0:
            ax6.hist(x6[hist21],bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax7 = fig.add_subplot(gs[1, 1])
        n7 = ax7.hist(x7, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color=c2,lw=4,alpha=0.8,label=bin2_name)
        ax7.hist(x7, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin2_name)        
        if len(x7[hist22]) > 0:
            ax7.hist(x7[hist22],bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax8 = fig.add_subplot(gs[2, 1])
        n8 = ax8.hist(x8, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color=c3,lw=4,alpha=0.8,label=bin3_name)
        ax8.hist(x8, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin3_name)
        if len(x8[hist23]) > 0:
            ax8.hist(x8[hist23],bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax9 = fig.add_subplot(gs[3, 1])
        n9 = ax9.hist(x9, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color=c4,lw=4,alpha=0.8,label=bin4_name)
        ax9.hist(x9, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin4_name)
        if len(x9[hist24]) > 0:
            ax9.hist(x9[hist24],bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        ax10 = fig.add_subplot(gs[4, 1])
        n10 = ax10.hist(x10, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color=c5,lw=4,alpha=0.8,label=bin5_name)
        ax10.hist(x10, bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),color='k',histtype='step',lw=1.5,alpha=1.0,label=bin5_name)
        if len(x10[hist25]) > 0:
            ax10.hist(x10[hist25],bins=np.arange(hist_bins2[0],hist_bins2[1],hist_bins2[2]),facecolor='None',edgecolor='k',histtype='step',hatch='/',lw=2.5,fill=True)
        
        
        
        
        n = np.append(n1[0], n2[0])
        n = np.append(n, n3[0])
        n = np.append(n, n4[0])
        n = np.append(n, n5[0])
        n = np.append(n, n6[0])
        n = np.append(n, n7[0])
        n = np.append(n, n8[0])
        n = np.append(n, n9[0])
        n = np.append(n, n10[0])
        
        y_lim = np.linspace(0,max(n)+8,4)
        y_lim = np.asarray(y_lim, dtype=int)
        if xlabel == r'log N$_{\rm H}$':
            x_lim = np.arange(xlim[0],xlim[1],1)
        else:
            x_lim = np.arange(xlim[0],xlim[1],0.5)

        x_lim2 = np.arange(xlim2[0],xlim2[1],0.5)

        ax1.set_yticks(y_lim)
        ax2.set_yticks(y_lim)
        ax3.set_yticks(y_lim)
        ax4.set_yticks(y_lim)
        ax5.set_yticks(y_lim)

        ax6.set_yticks(y_lim)
        ax7.set_yticks(y_lim)
        ax8.set_yticks(y_lim)
        ax9.set_yticks(y_lim)
        ax10.set_yticks(y_lim)



        ax1.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 1')
        ax1.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel1)
        ax2.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 2')
        ax2.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel2)
        ax3.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 3')
        ax3.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel3)
        ax4.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 4')  
        ax4.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel4)
        ax5.text(hist_bins[1]-1, max(n)-max(n)*0.1, 'Panel 5')
        ax5.text(hist_bins[1]-1, max(n)-max(n)*0.28, zlabel5)

        # if xlabel == r'log L$_{\rm X}$' or xlabel == r'log L$_{\rm bol}$':
        secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        secax1.set_xticks([9.5,10.5,11.5,12.5])
        secax1.set_xlabel(xlabel+r'/L$_{\odot}$')

        # elif xlabel == r'log L (1$\mu \rm m)$':
        #     secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_xticks([9, 10, 11, 12])
        #     secax1.set_xlabel(xlabel+r' [L$_{\odot}$]')

        # if top_label:
        #     secax1 = ax1.secondary_xaxis('top', functions=(self.solar_log, self.ergs_log))
        #     secax1.set_xticks([9, 10, 11, 12])
        #     secax1.set_xlabel(xlabel2)

        
        if median:
            ax1.axvline(np.nanmedian(x1[np.isfinite(x1)]), color='k', ls='--', lw=3.5)
            ax2.axvline(np.nanmedian(x2[np.isfinite(x2)]), color='k', ls='--', lw=3.5)
            ax3.axvline(np.nanmedian(x3[np.isfinite(x3)]), color='k', ls='--', lw=3.5)
            ax4.axvline(np.nanmedian(x4[np.isfinite(x4)]), color='k', ls='--', lw=3.5)
            ax5.axvline(np.nanmedian(x5[np.isfinite(x5)]), color='k', ls='--', lw=3.5)

            ax6.axvline(np.nanmedian(x6[np.isfinite(x6)]), color='k', ls='--', lw=3.5)
            ax7.axvline(np.nanmedian(x7[np.isfinite(x7)]), color='k', ls='--', lw=3.5)
            ax8.axvline(np.nanmedian(x8[np.isfinite(x8)]), color='k', ls='--', lw=3.5)
            ax9.axvline(np.nanmedian(x9[np.isfinite(x9)]), color='k', ls='--', lw=3.5)
            ax10.axvline(np.nanmedian(x10[np.isfinite(x10)]), color='k', ls='--', lw=3.5)

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax6.set_xticklabels([])
        ax7.set_xticklabels([])
        ax8.set_xticklabels([])
        ax9.set_xticklabels([])
        ax6.set_yticklabels([])
        ax7.set_yticklabels([])
        ax8.set_yticklabels([])
        ax9.set_yticklabels([])
        ax10.set_yticklabels([])
        ax1.set_xticks(x_lim)
        ax2.set_xticks(x_lim)
        ax3.set_xticks(x_lim)
        ax4.set_xticks(x_lim)
        ax5.set_xticks(x_lim)
        ax6.set_xticks(x_lim2)
        ax7.set_xticks(x_lim2)
        ax8.set_xticks(x_lim2)
        ax9.set_xticks(x_lim2)
        ax10.set_xticks(x_lim2)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        ax6.grid()
        ax7.grid()
        ax8.grid()
        ax9.grid()
        ax10.grid()

        # if xlabel != r'log N$_{\rm H}$':
        #     ax5.set_xlabel(xlabel+' [erg/s]')
        # elif xlabel == r'log N$_{\rm H}$':
        #     ax5.set_xlabel(xlabel+r' [cm$^{-2}$]')
        # else:
        ax5.set_xlabel(xlabel+r'/(erg s$^ {-1}$)')
        ax10.set_xlabel(r'log $L_{\rm bol-gal,e}$/$L_{\rm X}$')

        ax1.set_ylim(0, max(n)+max(n)*0.1)
        ax2.set_ylim(0, max(n)+max(n)*0.1)
        ax3.set_ylim(0, max(n)+max(n)*0.1)
        ax4.set_ylim(0, max(n)+max(n)*0.1)
        ax5.set_ylim(0, max(n)+max(n)*0.1)

        s = x1.sort()
        print('Hist 1')
        print('        Median,     max,     min,    mean')
        print('panel 1: ',np.nanmedian(x1[np.isfinite(x1)]),max(x1[np.isfinite(x1)]),min(x1[np.isfinite(x1)]),np.nanmean(x1[np.isfinite(x1)]))
        print('panel 2: ',np.nanmedian(x2[np.isfinite(x2)]),max(x2[np.isfinite(x2)]),min(x2[np.isfinite(x2)]),np.nanmean(x2[np.isfinite(x2)]))
        print('panel 3: ',np.nanmedian(x3[np.isfinite(x3)]),max(x3[np.isfinite(x3)]),min(x3[np.isfinite(x3)]),np.nanmean(x3[np.isfinite(x3)]))
        print('panel 4: ',np.nanmedian(x4[np.isfinite(x4)]),max(x4[np.isfinite(x4)]),min(x4[np.isfinite(x4)]),np.nanmean(x4[np.isfinite(x4)]))
        print('panel 5: ',np.nanmedian(x5[np.isfinite(x5)]),max(x5[np.isfinite(x5)]),min(x5[np.isfinite(x5)]),np.nanmean(x5[np.isfinite(x5)]))
        print('Hist 2')
        print('        Median,     max,     min,    mean')
        print('panel 1: ',10**np.nanmedian(x6[np.isfinite(x6)]),max(x6[np.isfinite(x6)]),min(x6[np.isfinite(x6)]),np.nanmean(x6[np.isfinite(x6)]))
        print('panel 2: ',10**np.nanmedian(x7[np.isfinite(x7)]),max(x7[np.isfinite(x7)]),min(x7[np.isfinite(x7)]),np.nanmean(x7[np.isfinite(x7)]))
        print('panel 3: ',10**np.nanmedian(x8[np.isfinite(x8)]),max(x8[np.isfinite(x8)]),min(x8[np.isfinite(x8)]),np.nanmean(x8[np.isfinite(x8)]))
        print('panel 4: ',10**np.nanmedian(x9[np.isfinite(x9)]),max(x9[np.isfinite(x9)]),min(x9[np.isfinite(x9)]),np.nanmean(x9[np.isfinite(x9)]))
        print('panel 5: ',10**np.nanmedian(x10[np.isfinite(x10)]),max(x10[np.isfinite(x10)]),min(x10[np.isfinite(x10)]),np.nanmean(x10[np.isfinite(x10)]))

        plt.tight_layout()
        plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/{savestring}.pdf')
        plt.show()