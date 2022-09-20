import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

from SED_plots_v2 import Plotter

class SED_shape_Plotter(Plotter):
    def __init__(self, ID, z, wavelength, Lum, L, norm, up_check, shape):
        super().__init__(ID, z, wavelength, Lum, L, norm, up_check)
        self.shape = shape

    def shape_1bin_h(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], uv_slope=None, mir_slope1=None, mir_slope2=None, Median_line=True, FIR_med=True, FIR_upper='upper lims'):
        '''Function to plot the 5 sed shapes in a single horizonal plot. This creates a 1 row x 5 column plot'''
        plt.rcParams['font.size'] = 40
        plt.rcParams['axes.linewidth'] = 4
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 5
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 5
        
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
        print('CHECK: ',len(b3[b3]))

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

        fig = plt.figure(figsize=(50,11))
        gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1,right=0.9,wspace=0.05,hspace=0.01,height_ratios=[0.2,3])

        ax1 = fig.add_subplot(gs[1, 4])#, aspect='equal', adjustable='box')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([])
        ax1.set_xticklabels(xticks_labels)
        ax1.text(0.05, 0.8, f'n = {len(x1)}', transform=ax1.transAxes)
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
                self.median_FIR_sed(wfir1, ffir1, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b1, lw=3)
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
                self.median_FIR_sed(wfir2, ffir2, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b2, lw=3)
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
        ax3.text(0.05, 0.8, f'n = {len(x3)}', transform=ax3.transAxes)
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
                self.median_FIR_sed(wfir3, ffir3, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b3, lw=3)
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
                self.median_FIR_sed(wfir4, ffir4, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b4, lw=3)
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
        ax5.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
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
                self.median_FIR_sed(wfir5, ffir5, connect=[x_connect, y_connect], upper=FIR_upper, Norm=False, Bin=True, bin_in=b5, lw=3)
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

    def shape_1bin_v(self, savestring, median_x=[np.nan], median_y=[np.nan], wfir=[[np.nan]], ffir=[[np.nan]], uv_slope=None, mir_slope1=None, mir_slope2=None, Median_line=True, FIR_med=True, FIR_upper='upper lims'):
        '''Function to plot the 5 sed shapes in a single horizonal plot. This creates a 1 row x 5 column plot'''
        # plt.rcParams['font.size'] = 40
        # plt.rcParams['axes.linewidth'] = 4
        # plt.rcParams['xtick.major.size'] = 6
        # plt.rcParams['xtick.major.width'] = 5
        # plt.rcParams['ytick.major.size'] = 6
        # plt.rcParams['ytick.major.width'] = 5
        
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
        ax1.text(0.05, 0.8, f'n = {len(x1)}', transform=ax1.transAxes)
        ax1.text(-0.25, 0.5, '1', transform=ax1.transAxes, fontsize=40, weight='bold')
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
        ax2.text(0.05, 0.8, f'n = {len(x2)}', transform=ax2.transAxes)
        ax2.text(-0.25, 0.5, '2', transform=ax2.transAxes, fontsize=40, weight='bold')
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
        ax3.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
        ax3.text(0.05, 0.8, f'n = {len(x3)}', transform=ax3.transAxes)
        ax3.text(-0.25, 0.5, '3', transform=ax3.transAxes, fontsize=40, weight='bold')
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
        ax4.text(0.05, 0.8, f'n = {len(x4)}', transform=ax4.transAxes)
        ax4.text(-0.25, 0.5, '4', transform=ax4.transAxes, fontsize=40, weight='bold')
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
        ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
        ax5.text(0.05, 0.8, f'n = {len(x5)}', transform=ax5.transAxes)
        ax5.text(-0.25, 0.5, '5', transform=ax5.transAxes, fontsize=40, weight='bold')
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
    

