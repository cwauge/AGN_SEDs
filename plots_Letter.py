# import binascii
# from re import I
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from astropy.cosmology import FlatLambdaCDM
from numpy.core.defchararray import lower
# from RunSEDv3 import F100_2
from filters import Filters
from astropy.io import ascii
from astropy.io import fits
from match import match
from SED_v7 import Flux_to_Lum
from bootstrap import bootstrap2D

class Plotter_Letter():
	'''A class to plot the properties of each source from the AGN class for the published letter'''

	def __init__(self,ID,z,wavelength,flux,Lx=None,Lbol=None,spec_type=None):
		self.ID = np.asarray(ID)
		self.z = np.asarray(z)
		self.wavelength_array = np.asarray(wavelength)
		self.flux_array = np.asarray(flux)
		self.Lx = np.asarray(Lx)
		self.Lbol = np.asarray(Lbol)
		self.spec_type = np.asarray(spec_type)


	def multilines(self,xs,ys,cs,ax=None,**kwargs):
		ax = plt.gca() if ax is None else ax # find axes
		segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
		lc = LineCollection(segments, **kwargs)
		lc.set_array(np.asarray(cs)) # set coloring of line segments
		ax.add_collection(lc) # add lines to axes and rescale
		ax.autoscale()
		return lc


	def Lx_z(self,Lx1,z1,label1=None,Lx2=None,z2=None,label2=None):

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3
		
		plt.figure(figsize=(10,8))
		plt.plot(z1,Lx1,'.',color='b',label=label1,rasterized=True)
		plt.xlabel('z')
		plt.ylabel(r'L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')
		plt.grid()

		plt.hist([-10,-10,-9,-10,-8,-7,-11],histtype='step',color='orange',lw=3,label='Sample')

		if any(Lx2) != None:
			plt.plot(z2,Lx2,'x',color='r',label=label2,rasterized=True)
			lgnd = plt.legend(markerscale=1.5)

		plt.axvline(0.89,ymin=0.357,ymax=0.873,color='orange',lw=3)
		plt.axvline(1.11,ymin=0.357,ymax=0.873,color='orange',lw=3)
		plt.axhline(42.49,xmin=0.1571,xmax=0.19,color='orange',lw=3)
		plt.axhline(46.13,xmin=0.1571,xmax=0.19,color='orange',lw=3)


		plt.xlim(-0.05,6)
		plt.ylim(40,47)

		
		plt.savefig('/Users/connor_auge/Desktop/Lx_z.png')
		plt.show()

	def Lx_z_hist(self,L1,z1,L2,z2,L3,z3,L4,z4):
		plt.rcParams['font.size']=24
		plt.rcParams['axes.linewidth']=3
		plt.rcParams['xtick.major.size']=3
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=3
		plt.rcParams['ytick.major.width'] = 3



		fig = plt.figure(figsize=(12,12))

		ax1 = plt.subplot(111)
		ax1.plot(z3,L3,'.',ms=10,color='gray',label='GOODS-N/S',rasterized=True)
		ax1.plot(z4,L4,'.',ms=10,color='gray',rasterized=True)
		ax1.plot(z1,L1,'+',ms=10,color='blue',label='COSMOS',rasterized=True,alpha=0.8)
		ax1.plot(z2,L2,'x',ms=10,color='red',label='Stripe82X',rasterized=True,alpha=0.8)
		
		ax1.set_xlim(-0.1,6)
		ax1.set_ylim(39,46.5)
		ax1.set_xlabel('z')
		ax1.set_ylabel(r'log$_{10}$ L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]')
		ax1.grid()

		lgnd = plt.legend(loc='lower right')
		lgnd.legendHandles[0]._legmarker.set_markersize(12)
		lgnd.legendHandles[1]._legmarker.set_markersize(12)

		plt.savefig('/Users/connor_auge/Desktop/Paper/cosmos2020/Lx_z_scatter.pdf')

		print(len(z1)+len(z2)+len(z3)+len(z4))
		plt.show()


		fig = plt.figure(figsize=(20,10))
		gs = fig.add_gridspec(nrows=1, ncols=2)
		# gs.update(wspace=0.08) # set the spacing between axes
		gs.update(left=0.08,right=0.98,top=0.9,bottom=0.15)

		bins = np.arange(0,7,0.25)

		ax2 = plt.subplot(gs[0])
		ax2.hist(np.append(z3,z4), bins=bins, histtype='step',color='gray',lw=3,label='GOODS-N/S')
		ax2.hist(z1, bins=bins, histtype='step',color='blue',lw=3,label='COSMOS')
		ax2.hist(z2, bins=bins, histtype='step',color='red',lw=3,label='Stripe82X')

		# ax2.set_xlim(0,6.5)
		ax2.set_ylabel('N')
		ax2.set_xlabel('z')
		ax2.grid()

		lgnd = plt.legend(loc='upper right')
		# lgnd.legendHandles[0]._legmarker.set_markersize(12)
		# lgnd.legendHandles[1]._legmarker.set_markersize(12)

		bins = np.arange(39,46.5,0.25)

		ax2 = plt.subplot(gs[1])
		ax2.hist(np.append(L3,L4), bins=bins, histtype='step',color='gray',lw=3,label='GOODS-N/S')
		ax2.hist(L1, bins=bins, histtype='step',color='blue',lw=3,label='COSMOS')
		ax2.hist(L2, bins=bins, histtype='step',color='red',lw=3,label='Stripe82X')

		ax2.set_xlim(39,46.5)
		ax2.set_ylabel('N')
		ax2.set_xlabel(r'log$_{10}$ L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]')
		ax2.grid()

		plt.savefig('/Users/connor_auge/Desktop/Paper/cosmos2020/Lx_z_hist.pdf')
		plt.show()



	def Lx_bin_hist(self,Lx,uv_slope=None,mir_slope1=None,mir_slope2=None):

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= 0.2, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= 0.2, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		L1 = Lx[B1]
		L2 = Lx[B2]
		L3 = Lx[B3]
		L4 = Lx[B4]
		L5 = Lx[B5]

		fig = plt.figure(figsize=(8,6))
		ax = plt.subplot(111)
		plt.hist(L1,bins=np.arange(42.5,46.5,0.25),lw=2,alpha=0.6,color=c1,label='Panel 1')
		plt.hist(L2,bins=np.arange(42.5,46.5,0.25),lw=2,alpha=0.6,color=c2,label='Panel 2')
		plt.hist(L3,bins=np.arange(42.5,46.5,0.25),lw=2,alpha=0.6,color=c3,label='Panel 3')
		plt.hist(L4,bins=np.arange(42.5,46.5,0.25),lw=2,alpha=0.6,color=c4,label='Panel 4')
		plt.hist(L5,bins=np.arange(42.5,46.5,0.25),lw=2,alpha=0.6,color=c5,label='Panel 5')
		plt.axvline(np.nanmedian(L1),lw=3,color=c1)
		plt.axvline(np.nanmedian(L2),lw=3,color=c2)
		plt.axvline(np.nanmedian(L3),lw=3,color=c3)
		plt.axvline(np.nanmedian(L4),lw=3,color=c4)
		plt.axvline(np.nanmedian(L5),lw=3,color=c5)
		plt.xlabel(r'log L$_{0.5-10\mathrm{keV}}$ [erg/s]')
		plt.legend()
		plt.grid()
		plt.show()


	def SEDs_param(self,x,y,Lx,param,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,uv_slope=None,mir_slope1=None,mir_slope2=None):
		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.5
		clim2 = 46

		param = np.asarray(param)
		# print(param)
		# param = np.log10(param)
		# param = np.log10(1/param)
		param = 1/param
		# param = param-1

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(Lx)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		cosmos_norm1 = norm1[mark1 == 0]
		s82X_norm1 = norm1[mark1 == 1]

		cosmos_norm2 = norm2[mark2 == 0]
		s82X_norm2 = norm2[mark2 == 1]

		cosmos_norm3 = norm3[mark3 == 0]
		s82X_norm3 = norm3[mark3 == 1]

		cosmos_norm4 = norm4[mark4 == 0]
		s82X_norm4 = norm4[mark4 == 1]

		cosmos_norm5 = norm5[mark5 == 0]
		s82X_norm5 = norm5[mark5 == 1]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]

		fig = plt.figure(figsize=(18,15),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=5, ncols=2, left=0.15,right=0.45,wspace=-0.25,hspace=0.1,width_ratios=[3,0.25])
		gs2 = fig.add_gridspec(nrows=5, ncols=1, left=0.6,right=0.85,wspace=0.1,hspace=0.1)

		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax2 = fig.add_subplot(gs1[1,0])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		spec_type2 = spec_type[B2]


		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax3 = fig.add_subplot(gs1[2,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		spec_type3 = spec_type[B3]

		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')


		ax4 = fig.add_subplot(gs1[3,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		spec_type4 = spec_type[B4]

		lc4 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5)
		ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		axcb4.remove()

		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(5E-3,50)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax5 = fig.add_subplot(gs1[4,0])
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		spec_type5 = spec_type[B5]

		x6 = x5
		y6 = y5

		lc5 = self.multilines(x6,y6,L5,cmap='rainbow',lw=1.5)
		ax5.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0),color='k',lw=3.5)
		axcb5 = fig.colorbar(lc5)
		axcb5.mappable.set_clim(clim1,clim2)
		axcb5.remove()

		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(5E-3,50)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)

		ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		yticks = [0,10,20,30,40]
		xticks = [0,1,2,3,4,5]
		
		cbar_ax = fig.add_subplot(gs1[:,-1:])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')

		ax6 = fig.add_subplot(gs2[0,0])
		ax6.hist(param[B1],np.arange(0,6,0.25),color='gray')
		ax6.axvline(np.median(param[B1]),ls='--',color='k',lw=3)
		# ax6.set_xlim(42,46)
		ax6.set_ylim(0,45)
		ax6.set_yticks(yticks)
		ax6.set_xticks(xticks)
		ax6.set_xticklabels([])
		ax6.grid()

		ax7 = fig.add_subplot(gs2[1,0])
		ax7.hist(param[B2],np.arange(0,6,0.25),color='gray')
		ax7.axvline(np.median(param[B2]),ls='--',color='k',lw=3)
		# ax7.set_xlim(42,46)
		ax7.set_ylim(0,45)
		ax7.set_yticks(yticks)
		ax7.set_xticks(xticks)
		ax7.set_xticklabels([])
		ax7.grid()

		ax8 = fig.add_subplot(gs2[2,0])
		ax8.hist(param[B3],np.arange(0,6,0.25),color='gray')
		ax8.axvline(np.median(param[B3]),ls='--',color='k',lw=3)
		# ax8.set_xlim(42,46)
		ax8.set_ylim(0,45)
		ax8.set_yticks(yticks)
		ax8.set_xticks(xticks)
		ax8.set_xticklabels([])
		ax8.grid()

		ax9 = fig.add_subplot(gs2[3,0])
		ax9.hist(param[B4],np.arange(0,6,0.25),color='gray')
		ax9.axvline(np.median(param[B4]),ls='--',color='k',lw=3)
		# ax9.set_xlim(42,46)
		ax9.set_ylim(0,45)
		ax9.set_yticks(yticks)
		ax9.set_xticks(xticks)
		ax9.set_xticklabels([])
		ax9.grid()

		ax10 = fig.add_subplot(gs2[4,0])
		ax10.hist(param[B5],np.arange(0,6,0.25),color='gray')
		ax10.axvline(np.median(param[B5]),ls='--',color='k',lw=3)
		# ax10.set_xlim(42,46)
		ax10.set_ylim(0,45)
		ax10.set_yticks(yticks)
		ax10.set_xticks(xticks)
		# ax10.set_xlabel(r'log $\frac{\mathrm{F}_{\mathrm{int}}}{\mathrm{F}_{\mathrm{obs}}}$',fontsize=26)
		ax10.set_xlabel(r'$\frac{\mathrm{L}_{\mathrm{X,int}}}{\mathrm{L}_{\mathrm{X,obs}}}$',fontsize=26)

		ax10.grid()

		plt.savefig('/Users/connor_auge/Desktop/NEWSED_abs_corr.png')
		plt.show()
	
	def multi_SED(self,savestring,x,y,L,median_x,median_y,median_x2=None,median_y2=None,median_x_ext=None,median_y_ext=None,flux_point=None,suptitle=None,norm=None,upper_w=None,upper_lim=None,mark=None,spec_z=None,wfir=None,ffir=None,up_check=None,med_x_fir=None,med_y_fir=None):
		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		median_x = np.asarray(median_x)
		median_y = np.asarray(median_y)

		x_data, y_data, L_data = [], [], []
		x_upper, y_upper, L_upper = [], [], []

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq



			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]



			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		median_upper_wave = np.nanmedian(cosmos_s82x_wave,axis=0)
		median_upper = np.nanmedian(cosmos_s82x_list,axis=0)

		median_upper_wave2 = np.nanmedian(cosmos_s82x_wave2,axis=0)
		median_upper2 = np.nanmedian(cosmos_s82x_list2,axis=0)

		upper_25_2 = np.nanpercentile(cosmos_s82x_list2,25,axis=0)
		upper_75_2 = np.nanpercentile(cosmos_s82x_list2,75,axis=0)




		# mean_upper = np.nanmedian(np.asarray([cosmos_nuLnu_upper/np.nanmedian(cosmos_norm),s82X_nuLnu_upper/np.nanmedian(s82x_norm)]),axis=0)
		# median_rest_upper_w_microns = np.nanmedian(rest_upper_w_microns,axis=0)
		# upper_25 = np.nanmedian(np.asarray([cosmos_nuLnu_upper/np.nanpercentile(cosmos_norm,25),s82X_nuLnu_upper/np.nanpercentile(s82x_norm,25)]),axis=0)
		# upper_75 = np.nanmedian(np.asarray([cosmos_nuLnu_upper/np.nanpercentile(cosmos_norm,75),s82X_nuLnu_upper/np.nanpercentile(s82x_norm,75)]),axis=0)

		# for i in range(len(x)):
		# 	ind = np.where(upper_lim[i] == 0.)[0]
		# 	x_data.append(x[i][ind])
		# 	y_data.append(y[i][ind])
		# 	# L_data.append(L[i][ind])

		# 	ind2 = np.where(upper_lim[i] == 1.)[0]
		# 	x_upper.append(x[i][ind2])
		# 	y_upper.append(y[i][ind2])
		# 	# L_upper.append(L[i][ind2])

		# x_data, x_upper, y_data, y_upper = np.asarray(x_data),np.asarray(x_upper),np.asarray(y_data),np.asarray(y_upper)
		x_data = x
		y_data = y
		# x_data = 10**median_x
		# y_data = 10**median_y


		# upper_w = np.vstack(upper_w)
		# upper_lim = np.vstack(upper_lim)

		# print(np.shape(upper_w))
		# print(np.shape(upper_lim))

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		clim1 = 43
		clim2 = 45.5
		cmap = 'rainbow_r'
		# cmap = mpl.colors.ListedColormap(['purple', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red'])
		# cmap = mpl.colors.ListedColormap([c2, c1, 'cyan', c3, 'yellow', c4, c5])
		# bounds = np.arange(clim1,clim2,0.5)
		# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

		# x[y > 5E2] = np.nan
		# y[y > 5E2] = np.nan
		# x[y < 1E-4] = np.nan
		# y[y < 1E-4] = np.nan

		loLx = min(L[~np.isnan(L)])-0.25
		hiLx = max(L[~np.isnan(L)])+0.25

		try:
			ffir = np.asarray([ffir[i]/norm[i] for i in range(len(ffir))])
		except TypeError:
			ffir = np.asarray([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])
			wfir = np.asarray([[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan]])


		plt.rcParams['font.size'] = 40
		plt.rcParams['axes.linewidth'] = 3.5
		plt.rcParams['xtick.major.size'] = 4.5
		plt.rcParams['xtick.major.width'] = 4.5
		plt.rcParams['ytick.major.size'] = 5.5
		plt.rcParams['ytick.major.width'] = 4.5


		fig, ax = plt.subplots(figsize=(20,15))
		ax.set_aspect(1)
		ax.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		# ax.set_xlim(8E-5,7E2)
		# ax.set_xlim(8E-1,7E2)
		ax.set_ylim(3E-3,120)
		ax.set_xticks([1E-3,1E-2,1E-1,1E0,1E1,1E2,1E3])
		# ax.set_title(suptitle)

		plt.grid()
		ax.text(0.15,0.85,f'n = {len(L)}',transform=ax.transAxes)

	
		# upper_seg = np.stack((cosmos_s82x_wave,cosmos_s82x_list),axis=2)
		upper_seg = np.stack((wfir,ffir),axis=2)
		upper_all = LineCollection(upper_seg,color='gray',alpha=0.3)
		ax.add_collection(upper_all)

		lc = self.multilines(x_data[L >= clim1-0.1],y_data[L >= clim1-0.1],L[L >= clim1-0.1],lw=2.5,cmap=cmap,alpha=0.75,rasterized=True)
		# plt.plot(x_data[L >= clim1-0.1], y_data[L >= clim1-0.1],'.',color='k')
		# upper_lim_points = plt.plot(upper_w,upper_lim,color='k')
		# lc_upper = self.multilines(upper_w[L >= clim1-0.1],upper_lim[L >= clim1-0.1],L[L >= clim1-0.1],cmap='rainbow',ls='--')

		# upper_lim = plt.plot(upper_wave,mean_upper/norm,'.',color='k')
		# lc2 = self.multilines(x[L < clim1],y[L < clim1],L[L < clim1],cmap='gray')
		# points = plt.plot(x[L >= clim1-0.1],y[L >= clim1-0.1],'.',color='k')
		axcb1 = fig.colorbar(lc)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()
		# axcb2 = fig.colorbar(lc2)
		# axcb2.mappable.set_clim(20,100)
		# axcb2.remove()
		test = plt.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap=cmap)
		axcb = fig.colorbar(test)
		# axcb = fig.colorbar(test,cmap=cmap, norm=norm)
		# axcb.mappable.set_clim(clim1,clim2)
		axcb.set_label( r'log L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]')
		median_line = ax.plot(10**np.nanmedian(median_x,axis=0),10**np.nanmean(median_y,axis=0),color='k',lw=5.5)
		# median_line2 = ax.plot(np.nanmedian(10**median_x2,axis=0),np.nanmedian(10**median_y2,axis=0),color='k',ls='--',lw=5.5)
		# percentile_25 = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,25,axis=0),color='k',ls='--',lw=3.5)
		# percentile_75 = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,75,axis=0),color='k',ls='--',lw=3.5)
		xray = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data[L >= clim1-0.1][:,:2],axis=0),color='k',lw=5.5)
		# xray_percentile_25 = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data[L >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.5)
		# xray_percentile_75 = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data[L >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.5)

		# ax.fill_between(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,75,axis=0),np.nanpercentile(10**median_y,25,axis=0),color='k',alpha=0.8,zorder=-1)

		FIR_med_x = np.nanmedian(x, axis=0)[np.nanmedian(x, axis=0) > 50]
		FIR_med_y = np.nanmedian(y, axis=0)[np.nanmedian(x, axis=0) > 50]


		# FIR_data_line = ax.plot(FIR_med_x, FIR_med_y, color='k',lw=3.0)
		# lc2 = self.multilines(wfir[up_check==0],ffir[up_check==0],L[up_check==0],cmap='viridis',alpha=0.75,rasterized=True)	

		# upper_lim_points = ax.plot(median_upper_wave2,median_upper2,'v',ms=12,color='k')
		# upper_lim_line = ax.plot(median_upper_wave2,median_upper2,color='k',lw=5.0)
		# ax.plot(np.nanmean(wfir,axis=0),np.nanmean(ffir,axis=0)/np.nanmean(norm,axis=0),color='green',lw=7.0)
		ax.plot(np.nanmedian(wfir,axis=0)[-3:],np.nanmedian(ffir,axis=0)[-3:],color='k',lw=5.0)
		ax.plot(np.nanmedian(wfir,axis=0)[-3:],np.nanmedian(ffir,axis=0)[-3:],'v',color='k',ms=15.0)
		try:
			MIR_x = np.append(np.nanmedian(10**median_x[up_check==0],axis=0)[-1],np.nanmedian(wfir[up_check==0],axis=0)[:2])
			MIR_y = np.append(np.nanmedian(10**median_y[up_check==0],axis=0)[-1],np.nanmedian(ffir[up_check==0],axis=0)[:2])
			MIR_x = np.append(MIR_x, np.nanmedian(wfir, axis=0)[-3])
			MIR_y = np.append(MIR_y,np.nanmedian(ffir,axis=0)[-3])
		except IndexError:
			MIR_x = np.asarray([np.nan,np.nan,np.nan])
			MIR_y = np.asarray([np.nan,np.nan,np.nan])

		ax.plot(MIR_x, MIR_y, '--',color='k',lw=5.0)
		# ax.plot(np.nanmedian(10**med_x_fir[up_check == 0],axis=0),np.nanmedian(10**med_y_fir[up_check == 0],axis=0), '--',color='b',lw=5.0)
		# ax.plot(np.append(np.nanmedian(10**median_x,axis=0)[-1],np.nanmedian(wfir,axis=0)[:3]),np.append(np.nanmedian(10**median_y,axis=0)[-1],np.nanmedian(ffir,axis=0)[:3]),'--',color='r',lw=5.0)
		# upper_lim_line_25 = ax.plot(median_upper_wave2,upper_25_2,'--',color='k',lw=3.5)
		# upper_lim_line_75 = ax.plot(median_upper_wave2,upper_75_2,'--',color='k',lw=3.5)

		plt.xlim(5E-5,1E3)
		plt.ylim(1E-4,200)
		# flux_point_wave = np.zeros(np.shape(flux_point))
		# flux_point_wave[flux_point_wave == 0] = 100
		# ax.plot(flux_point_wave,flux_point,'x',color='red')
		plt.tight_layout()

		plt.savefig('/Users/connor_auge/Desktop/Final_plots/Multi_SEDs'+savestring+'.pdf')
		plt.show()

	def multi_SED_zbins(self, savestring, x, y, Lx, z, median_wavelength, median_flux, norm, mark, spec_z=None,wfir=None,ffir=None,up_check=None,med_x_fir=None,med_y_fir=None):

		L = Lx
		x_data = x
		y_data = y

		clim1 = 43
		clim2 = 45.5

		# upper_wave1 = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
		# rest_upper1_w = upper_wave1/(1+0.4)
		# rest_upper1_w_cgs = rest_upper1_w*1E-8
		# rest_upper1_w_microns = rest_upper1_w*1E-4
		# rest_upper1_w_freq = 3E10/rest_upper1_w_cgs

		# upper_wave2 = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
		# rest_upper2_w = upper_wave2/(1+0.7)
		# rest_upper2_w_cgs = rest_upper2_w*1E-8
		# rest_upper2_w_microns = rest_upper2_w*1E-4
		# rest_upper2_w_freq = 3E10/rest_upper2_w_cgs

		# upper_wave3 = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
		# rest_upper3_w = upper_wave3/(1+1.0)
		# rest_upper3_w_cgs = rest_upper3_w*1E-8
		# rest_upper3_w_microns = rest_upper3_w*1E-4
		# rest_upper3_w_freq = 3E10/rest_upper3_w_cgs

		# cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
		# s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6

		# cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
		# s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

		# cosmos_nuFnu_upper1 = cosmos_upper_lim_cgs*rest_upper1_w_freq
		# s82X_nuFnu_upper1 = s82X_upper_lim_cgs*rest_upper1_w_freq

		# cosmos_nuFnu_upper2 = cosmos_upper_lim_cgs*rest_upper2_w_freq
		# s82X_nuFnu_upper2 = s82X_upper_lim_cgs*rest_upper2_w_freq

		# cosmos_nuFnu_upper3 = cosmos_upper_lim_cgs*rest_upper3_w_freq
		# s82X_nuFnu_upper3 = s82X_upper_lim_cgs*rest_upper3_w_freq

		# cosmos_nuLnu_upper1 = Flux_to_Lum(cosmos_nuFnu_upper1,0.4)
		# s82X_nuLnu_upper1 = Flux_to_Lum(s82X_nuFnu_upper1,0.4)

		# cosmos_nuLnu_upper2 = Flux_to_Lum(cosmos_nuFnu_upper2,0.7)
		# s82X_nuLnu_upper2 = Flux_to_Lum(s82X_nuFnu_upper2,0.7)

		# cosmos_nuLnu_upper3 = Flux_to_Lum(cosmos_nuFnu_upper3,1.0)
		# s82X_nuLnu_upper3 = Flux_to_Lum(s82X_nuFnu_upper3,1.0)

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq



			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]



			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		cosmos_norm = norm[mark == 0]
		s82x_norm = norm[mark == 1]


		plt.rcParams['font.size']=19
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# B1 = np.logical_and(z >= 0.3,z <= 0.5)
		# B2 = np.logical_and(z >= 0.6,z <= 0.8)
		# B3 = np.logical_and(z >= 0.9,z <= 1.1)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		B1 = np.logical_and(z >= zlim_1,z <= zlim_2)
		B2 = np.logical_and(z > zlim_2,z <= zlim_3)
		B3 = np.logical_and(z > zlim_3,z <= zlim_4)



		# scale_s82x = np.nanmedian(norm[B3],axis=0)/np.nanmedian(norm[B2],axis=0)
		# scale_goods = np.nanmedian(norm[B1],axis=0)/np.nanmedian(norm[B2],axis=0)
		scale_s82x = 1.0
		scale_goods = 1.0

		# median_upper_wave1 = np.nanmedian(cosmos_s82x_wave2[B1],axis=0)
		# median_upper1 = np.nanmedian(cosmos_s82x_list2[B1],axis=0)
		# upper_25_1 = np.nanpercentile(cosmos_s82x_list2[B1],25,axis=0)
		# upper_75_1 = np.nanpercentile(cosmos_s82x_list2[B1],75,axis=0)

		# median_upper_wave2 = np.nanmedian(cosmos_s82x_wave2[B2],axis=0)
		# median_upper2 = np.nanmedian(cosmos_s82x_list2[B2],axis=0)
		# upper_25_2 = np.nanpercentile(cosmos_s82x_list2[B2],25,axis=0)
		# upper_75_2 = np.nanpercentile(cosmos_s82x_list2[B2],75,axis=0)

		# median_upper_wave3 = np.nanmedian(cosmos_s82x_wave2[B3],axis=0)
		# median_upper3 = np.nanmedian(cosmos_s82x_list2[B3],axis=0)
		# upper_25_3 = np.nanpercentile(cosmos_s82x_list2[B3],25,axis=0)
		# upper_75_3 = np.nanpercentile(cosmos_s82x_list2[B3],75,axis=0)


		median_upper_wave1 = np.nanmedian(wfir[B1],axis=0)[-3:]
		median_upper1 = np.nanmedian(ffir[B1],axis=0)[-3:]/np.nanmedian(norm[B1],axis=0)

		median_upper_wave2 = np.nanmedian(wfir[B2],axis=0)[-3:]
		median_upper2 = np.nanmedian(ffir[B2],axis=0)[-3:]/np.nanmedian(norm[B2],axis=0)

		median_upper_wave3 = np.nanmedian(wfir[B3],axis=0)[-3:]
		median_upper3 = np.nanmedian(ffir[B3],axis=0)[-3:]/np.nanmedian(norm[B3],axis=0)

		median_upper_wave12 = np.nanmedian(wfir[B1][up_check[B1] == 0],axis=0)[:2]
		median_upper12 = np.nanmedian(ffir[B1][up_check[B1] == 0],axis=0)[:2]/np.nanmedian(norm[B1][up_check[B1] == 0],axis=0)

		median_upper_wave22 = np.nanmedian(wfir[B2],axis=0)[:2]
		median_upper22 = np.nanmedian(ffir[B2],axis=0)[:2]/np.nanmedian(norm[B2],axis=0)

		median_upper_wave32 = np.nanmedian(wfir[B3],axis=0)[:2]
		median_upper32 = np.nanmedian(ffir[B3],axis=0)[:2]/np.nanmedian(norm[B3],axis=0)

		ffir = np.asarray([ffir[i]/norm[i] for i in range(len(ffir))])

		xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
		yticks = [0.001,0.01,0.1,1,10,100]
		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

		fig = plt.figure(figsize=(18,6),constrained_layout=False)
		# gs1 = fig.add_gridspec(nrows=1, ncols=4, bottom=0.15, top=0.9, left=0.08,right=0.9,wspace=-0.2,hspace=0.2,width_ratios=[3.5,3.5,3.5,0.15])
		gs1 = fig.add_gridspec(nrows=1, ncols=3, bottom=0.1, top=0.95, left=0.08,right=1.05,wspace=-0.15)

		ax1 = fig.add_subplot(gs1[0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		x_data1 = x1
		y_data1 = y1

		# upper_seg1 = np.stack((cosmos_s82x_wave[B1],cosmos_s82x_list[B1]),axis=2)
		upper_seg1 = np.stack((wfir[B1][up_check[B1]==0],ffir[B1][up_check[B1]==0]),axis=2)
		upper_all1 = LineCollection(upper_seg1,color='gray',alpha=0.3)
		ax1.add_collection(upper_all1)


		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		
		lc1 = self.multilines(x1,y1*scale_goods,L1,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0)*scale_goods,color='k',lw=4)
		# percentile1_25 = ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanpercentile(10**median_flux[B1],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile1_75 = ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanpercentile(10**median_flux[B1],75,axis=0),color='k',ls='--',lw=3.0)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		xray1 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data1[L1 >= clim1-0.1][:,:2],axis=0)*scale_goods,color='k',lw=4)
		# xray1_percentile_25 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data1[L1 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray1_percentile_75 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data1[L1 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax1.set_aspect(1)
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(6E-5,7E2)
		ax1.set_ylim(1E-4,120)
		# ax1.set_xticklabels([])
		# ax1.set_yticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_xticklabels(xticks_labels)
		ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2),fontsize=21)
		ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$',fontsize=22)
		# ax1.set_title(r'L$_{\mathrm{X}}$ > 10$^{44}$')

		# upper_lim_points1 = ax1.plot(median_upper_wave1,median_upper1,'v',ms=10,color='k')
		# upper_lim_line1 = ax1.plot(median_upper_wave1,median_upper1,color='k',lw=4.0)
		ax1.plot(median_upper_wave1,median_upper1*scale_goods,'v',ms=10,color='k')
		ax1.plot(median_upper_wave1,median_upper1*scale_goods,color='k',lw=4.0)

		MIR_x = np.append(np.nanmedian(10**median_wavelength[B1],axis=0)[-1],median_upper_wave12)
		MIR_y = np.append(np.nanmedian(10**median_flux[B1],axis=0)[-1],median_upper12)
		MIR_x = np.append(MIR_x, median_upper_wave1[0])
		MIR_y = np.append(MIR_y,median_upper1[0])
		# lc12 = self.multilines(wfir[B1][up_check[B1]==0],ffir[B1][up_check[B1]==0],L[B1][up_check[B1]==0],cmap='viridis',alpha=0.75,rasterized=True)	

		# ax1.plot(np.nanmedian(10**med_x_fir[B1][up_check[B1] == 0],axis=0),np.nanmedian(10**med_y_fir[B1][up_check[B1] == 0],axis=0), '--',color='b',lw=5.0)
		ax1.plot(MIR_x, MIR_y, '--',color='k',lw=5.0)
		# ax1.plot(np.append(np.nanmedian(10**median_wavelength[B1],axis=0)[-1],median_upper_wave12),np.append(np.nanmedian(10**median_flux[B1],axis=0)[-1],median_upper12)*scale_goods,'--',color='k',lw=4.0)

		# upper_lim_line1_25 = ax1.plot(median_upper_wave1,upper_25_1,'--',color='k',lw=3.0)
		# upper_lim_line1_75 = ax1.plot(median_upper_wave1,upper_75_1,'--',color='k',lw=3.0)

		ax2 = fig.add_subplot(gs1[1])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		x_data2 = x2
		y_data2 = y2

		# upper_seg2 = np.stack((cosmos_s82x_wave[B2],cosmos_s82x_list[B2]),axis=2)
		upper_seg2 = np.stack((wfir[B2],ffir[B2]),axis=2)
		upper_all2 = LineCollection(upper_seg2,color='gray',alpha=0.3)
		ax2.add_collection(upper_all2)

		lc2 = self.multilines(x2,y2,L2,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=4)
		# percentile2_25 = ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanpercentile(10**median_flux[B2],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile2_75 = ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanpercentile(10**median_flux[B2],75,axis=0),color='k',ls='--',lw=3.0)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		xray2 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data2[L2 >= clim1-0.1][:,:2],axis=0),color='k',lw=4)
		# xray2_percentile_25 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data2[L2 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray2_percentile_75 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data2[L2 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax2.set_aspect(1)
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(6E-5,7E2)
		ax2.set_ylim(1E-4,120)
		ax2.set_yticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_xticklabels(xticks_labels)
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3),fontsize=21)		
		ax2.text(0.05,0.8,f'n = {len(x2)}',transform=ax2.transAxes)
		# ax2.text(-0.2,0.625,r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',transform=ax2.transAxes,rotation=90,fontsize=27)

		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$ normalized at 1$\mu$m')
		ax2.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)

		# upper_lim_points2 = ax2.plot(median_upper_wave2,median_upper2,'v',ms=10,color='k')
		# upper_lim_line2 = ax2.plot(median_upper_wave2,median_upper2,color='k',lw=4.0)
		# upper_lim_line2_25 = ax2.plot(median_upper_wave2,upper_25_2,'--',color='k',lw=3.0)
		# upper_lim_line2_75 = ax2.plot(median_upper_wave2,upper_75_2,'--',color='k',lw=3.0)
		ax2.plot(median_upper_wave2,median_upper2,'v',ms=10,color='k')
		ax2.plot(median_upper_wave2,median_upper2,color='k',lw=4.0)
		ax2.plot(np.append(np.nanmedian(10**median_wavelength[B2],axis=0)[-1],median_upper_wave22),np.append(np.nanmedian(10**median_flux[B2],axis=0)[-1],median_upper22),'--',color='k',lw=4.0)
		

		ax3 = fig.add_subplot(gs1[2])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		x_data3 = x3
		y_data3 = y3

		# upper_seg3 = np.stack((cosmos_s82x_wave[B3],cosmos_s82x_list[B3]),axis=2)
		upper_seg3 = np.stack((wfir[B3],ffir[B3]),axis=2)
		upper_all3 = LineCollection(upper_seg3,color='gray',alpha=0.3)
		ax3.add_collection(upper_all3)

		lc3 = self.multilines(x3,y3*scale_s82x,L3,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanmedian(10**median_flux[B3],axis=0)*scale_s82x,color='k',lw=4)
		# percentile3_25 = ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanpercentile(10**median_flux[B3],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile3_75 = ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanpercentile(10**median_flux[B3],75,axis=0),color='k',ls='--',lw=3.0)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		xray3 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data3[L3 >= clim1-0.1][:,:2],axis=0)*scale_s82x,color='k',lw=4)
		# xray3_percentile_25 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data3[L3 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray3_percentile_75 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data3[L3 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax3.set_aspect(1)
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(6E-5,7E2)
		ax3.set_ylim(1E-4,120)
		ax3.set_yticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.set_xticklabels(xticks_labels)
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4), fontsize=21)
		ax3.text(0.05,0.8,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax3.text(-0.2,0.625,r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',transform=ax2.transAxes,rotation=90,fontsize=27)

		# upper_lim_points3 = ax3.plot(median_upper_wave3,median_upper3,'v',ms=10,color='k')
		# upper_lim_line3 = ax3.plot(median_upper_wave3,median_upper3,color='k',lw=4.0)
		# upper_lim_line3_25 = ax3.plot(median_upper_wave3,upper_25_3,'--',color='k',lw=3.0)
		# upper_lim_line3_75 = ax3.plot(median_upper_wave3, upper_75_3, '--', color='k', lw=3.0)
		ax3.plot(median_upper_wave3,median_upper3*scale_s82x,'v',ms=10,color='k')
		ax3.plot(median_upper_wave3,median_upper3*scale_s82x,color='k',lw=4.0)
		ax3.plot(np.append(np.nanmedian(10**median_wavelength[B3], axis=0)[-1], median_upper_wave32), np.append(
			np.nanmedian(10**median_flux[B3], axis=0)[-1], median_upper32)*scale_s82x, '--', color='k', lw=4.0)

		ax1.grid()
		ax2.grid()
		ax3.grid()
		# cbar_ax = fig.add_subplot(gs1[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		# cb = fig.colorbar(test,cax=cbar_ax)
		# cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]',fontsize=21)

		plt.savefig('/Users/connor_auge/Desktop/New_plots3/Multi_SEDs_zbins'+savestring+'.pdf')
		plt.show()


	def multi_SED_field(self, savestring, x, y, Lx, z, median_wavelength, median_flux, norm, mark, spec_z=None, wfir=None, ffir=None):
		L = Lx
		x_data = x
		y_data = y

		clim1 = 43
		clim2 = 45.5

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq

			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]


			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		cosmos_norm = norm[mark == 0]
		s82x_norm = norm[mark == 1]


		plt.rcParams['font.size']=19
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# B1 = np.logical_and(z >= 0.3,z <= 0.5)
		# B2 = np.logical_and(z >= 0.6,z <= 0.8)
		# B3 = np.logical_and(z >= 0.9,z <= 1.1)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		B1 = np.logical_or(mark == 2, mark == 3)
		B2 = (mark == 0)
		B3 = (mark == 1)

		Ffir = np.array([ffir[i]/norm[i] for i in range(len(ffir))])



		# scale_s82x = np.nanmedian(norm[B3],axis=0)/np.nanmedian(norm[B2],axis=0)
		# scale_goods = np.nanmedian(norm[B1],axis=0)/np.nanmedian(norm[B2],axis=0)
		scale_s82x = 1.0
		scale_goods = 1.0

		# median_upper_wave1 = np.nanmedian(cosmos_s82x_wave2[B1],axis=0)
		# median_upper1 = np.nanmedian(cosmos_s82x_list2[B1],axis=0)
		# upper_25_1 = np.nanpercentile(cosmos_s82x_list2[B1],25,axis=0)
		# upper_75_1 = np.nanpercentile(cosmos_s82x_list2[B1],75,axis=0)
		# wfir_1 = np.nanmedian(wfir[B1], axis=0)
		# ffir_1 = np.nanmedian(Ffir[B1], axis=0)
		# ffir_25_1 = np.nanpercentile(Ffir[B1],25,axis=0)
		# ffir_75_1 = np.nanpercentile(Ffir[B1],75,axis=0)

		# median_upper_wave2 = np.nanmedian(cosmos_s82x_wave2[B2],axis=0)
		# median_upper2 = np.nanmedian(cosmos_s82x_list2[B2],axis=0)
		# upper_25_2 = np.nanpercentile(cosmos_s82x_list2[B2],25,axis=0)
		# upper_75_2 = np.nanpercentile(cosmos_s82x_list2[B2],75,axis=0)
		# wfir_2 = np.nanmedian(wfir[B2], axis=0)
		# ffir_2 = np.nanmedian(Ffir[B2], axis=0)
		# ffir_25_2 = np.nanpercentile(Ffir[B2],25,axis=0)
		# ffir_75_2 = np.nanpercentile(Ffir[B2],75,axis=0)

		# median_upper_wave3 = np.nanmedian(cosmos_s82x_wave2[B3],axis=0)
		# median_upper3 = np.nanmedian(cosmos_s82x_list2[B3],axis=0)
		# upper_25_3 = np.nanpercentile(cosmos_s82x_list2[B3],25,axis=0)
		# upper_75_3 = np.nanpercentile(cosmos_s82x_list2[B3],75,axis=0)
		# wfir_3 = np.nanmedian(wfir[B3], axis=0)
		# ffir_3 = np.nanmedian(Ffir[B3], axis=0)
		# ffir_25_3 = np.nanpercentile(Ffir[B3],25,axis=0)
		# ffir_75_3 = np.nanpercentile(Ffir[B3],75,axis=0)

		median_upper_wave1 = np.nanmedian(wfir[B1],axis=0)[-3:]
		median_upper1 = np.nanmedian(ffir[B1],axis=0)[-3:]/np.nanmedian(norm[B1],axis=0)

		median_upper_wave2 = np.nanmedian(wfir[B2],axis=0)[-3:]
		median_upper2 = np.nanmedian(ffir[B2],axis=0)[-3:]/np.nanmedian(norm[B2],axis=0)

		median_upper_wave3 = np.nanmedian(wfir[B3],axis=0)[-3:]
		median_upper3 = np.nanmedian(ffir[B3],axis=0)[-3:]/np.nanmedian(norm[B3],axis=0)

		median_upper_wave12 = np.nanmedian(wfir[B1],axis=0)[:3]
		median_upper12 = np.nanmedian(ffir[B1],axis=0)[:3]/np.nanmedian(norm[B1],axis=0)

		median_upper_wave22 = np.nanmedian(wfir[B2],axis=0)[:3]
		median_upper22 = np.nanmedian(ffir[B2],axis=0)[:3]/np.nanmedian(norm[B2],axis=0)

		median_upper_wave32 = np.nanmedian(wfir[B3],axis=0)[:3]
		median_upper32 = np.nanmedian(ffir[B3],axis=0)[:3]/np.nanmedian(norm[B3],axis=0)

		ffir = np.asarray([ffir[i]/norm[i] for i in range(len(ffir))])


		xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]
		yticks = [0.001,0.01,0.1,1,10,100]
		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']

		fig = plt.figure(figsize=(18,6),constrained_layout=False)
		# gs1 = fig.add_gridspec(nrows=1, ncols=4, bottom=0.15, top=0.9, left=0.08,right=0.9,wspace=-0.2,hspace=0.2,width_ratios=[3.5,3.5,3.5,0.15])
		gs1 = fig.add_gridspec(nrows=1, ncols=3, bottom=0.1, top=0.95, left=0.08,right=1.05,wspace=-0.15)

		ax1 = fig.add_subplot(gs1[0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		x_data1 = x1
		y_data1 = y1

		# upper_seg1 = np.stack((cosmos_s82x_wave[B1],cosmos_s82x_list[B1]),axis=2)
		upper_seg1 = np.stack((wfir[B1],ffir[B1]),axis=2)
		upper_all1 = LineCollection(upper_seg1,color='gray',alpha=0.3)
		ax1.add_collection(upper_all1)


		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		
		lc1 = self.multilines(x1,y1*scale_goods,L1,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0)*scale_goods,color='k',lw=4)
		# percentile1_25 = ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanpercentile(10**median_flux[B1],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile1_75 = ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanpercentile(10**median_flux[B1],75,axis=0),color='k',ls='--',lw=3.0)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		xray1 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data1[L1 >= clim1-0.1][:,:2],axis=0)*scale_goods,color='k',lw=4)
		# xray1_percentile_25 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data1[L1 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray1_percentile_75 = ax1.plot(np.nanmedian(x_data1[L1 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data1[L1 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax1.set_aspect(1)
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(6E-5,7E2)
		ax1.set_ylim(1E-4,120)
		# ax1.set_xticklabels([])
		# ax1.set_yticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_xticklabels(xticks_labels)
		ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_title('GOODS-N/S',fontsize=21)
		ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$',fontsize=22)
		# ax1.set_title(r'L$_{\mathrm{X}}$ > 10$^{44}$')

		# upper_lim_points1 = ax1.plot(median_upper_wave1,median_upper1,'v',ms=10,color='k')
		# upper_lim_line1 = ax1.plot(median_upper_wave1,median_upper1,color='k',lw=4.0)
		# upper_lim_line1_25 = ax1.plot(median_upper_wave1,upper_25_1,'--',color='k',lw=3.0)
		# upper_lim_line1_75 = ax1.plot(median_upper_wave1,upper_75_1,'--',color='k',lw=3.0)
		ax1.plot(median_upper_wave1,median_upper1*scale_goods,'v',ms=10,color='k')
		ax1.plot(median_upper_wave1,median_upper1*scale_goods,color='k',lw=4.0)
		ax1.plot(np.append(np.nanmedian(10**median_wavelength[B1],axis=0)[-1],median_upper_wave12),np.append(np.nanmedian(10**median_flux[B1],axis=0)[-1],median_upper12)*scale_goods,'--',color='k',lw=4.0)

		# ax1.plot(wfir_1,ffir_1,'v',ms=10,color='k')
		# ax1.plot(wfir_1,ffir_1,color='k',lw=4.0)
		# ax1.plot(wfir_1,ffir_1,'--',color='k',lw=3.0)
		# ax1.plot(wfir_1, ffir_1, '--', color='k', lw=3.0)
		# ax1.plot(wfir_1,ffir_25_1,'--',color='k',lw=3.0)
		# ax1.plot(wfir_1,ffir_75_1,'--',color='k',lw=3.0)

		ax2 = fig.add_subplot(gs1[1])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		x_data2 = x2
		y_data2 = y2

		test = ax2.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')

		# upper_seg2 = np.stack((cosmos_s82x_wave[B2],cosmos_s82x_list[B2]),axis=2)
		upper_seg2 = np.stack((wfir[B2],ffir[B2]),axis=2)
		upper_all2 = LineCollection(upper_seg2,color='gray',alpha=0.3)
		ax2.add_collection(upper_all2)

		lc2 = self.multilines(x2,y2,L2,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=4)
		# percentile2_25 = ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanpercentile(10**median_flux[B2],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile2_75 = ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanpercentile(10**median_flux[B2],75,axis=0),color='k',ls='--',lw=3.0)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		xray2 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data2[L2 >= clim1-0.1][:,:2],axis=0),color='k',lw=4)
		# xray2_percentile_25 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data2[L2 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray2_percentile_75 = ax2.plot(np.nanmedian(x_data2[L2 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data2[L2 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax2.set_aspect(1)
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(6E-5,7E2)
		ax2.set_ylim(1E-4,120)
		ax2.set_yticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_xticklabels(xticks_labels)
		ax2.set_title('COSMOS',fontsize=21)		
		ax2.text(0.05,0.8,f'n = {len(x2)}',transform=ax2.transAxes)
		# ax2.text(-0.2,0.625,r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',transform=ax2.transAxes,rotation=90,fontsize=27)

		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$ normalized at 1$\mu$m')
		ax2.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)

		# upper_lim_points2 = ax2.plot(median_upper_wave2,median_upper2,'v',ms=10,color='k')
		# upper_lim_line2 = ax2.plot(median_upper_wave2,median_upper2,color='k',lw=4.0)
		# upper_lim_line2_25 = ax2.plot(median_upper_wave2,upper_25_2,'--',color='k',lw=3.0)
		# upper_lim_line2_75 = ax2.plot(median_upper_wave2,upper_75_2,'--',color='k',lw=3.0)
		ax2.plot(median_upper_wave2,median_upper2,'v',ms=10,color='k')
		ax2.plot(median_upper_wave2,median_upper2,color='k',lw=4.0)
		ax2.plot(np.append(np.nanmedian(10**median_wavelength[B2],axis=0)[-1],median_upper_wave22),np.append(np.nanmedian(10**median_flux[B2],axis=0)[-1],median_upper22),'--',color='k',lw=4.0)

		# ax2.plot(wfir_2,ffir_2,'v',ms=10,color='k')
		# ax2.plot(wfir_2,ffir_2,color='k',lw=4.0)
		# ax2.plot(wfir_2,ffir_2,'--',color='k',lw=3.0)
		# ax2.plot(wfir_2, ffir_2, '--', color='k', lw=3.0)
		# ax2.plot(wfir_2,ffir_25_2,'--',color='k',lw=3.0)
		# ax2.plot(wfir_2,ffir_75_2,'--',color='k',lw=3.0)
		

		ax3 = fig.add_subplot(gs1[2])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		x_data3 = x3
		y_data3 = y3

		test = ax3.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')




		# upper_seg3 = np.stack((cosmos_s82x_wave[B3],cosmos_s82x_list[B3]),axis=2)
		upper_seg3 = np.stack((wfir[B3],ffir[B3]),axis=2)
		upper_all3 = LineCollection(upper_seg3,color='gray',alpha=0.3)
		ax3.add_collection(upper_all3)

		lc3 = self.multilines(x3,y3*scale_s82x,L3,cmap='rainbow_r',lw=1.5,alpha=0.7,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanmedian(10**median_flux[B3],axis=0)*scale_s82x,color='k',lw=4)
		# percentile3_25 = ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanpercentile(10**median_flux[B3],25,axis=0),color='k',ls='--',lw=3.0)
		# percentile3_75 = ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanpercentile(10**median_flux[B3],75,axis=0),color='k',ls='--',lw=3.0)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		xray3 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data3[L3 >= clim1-0.1][:,:2],axis=0)*scale_s82x,color='k',lw=4)
		# xray3_percentile_25 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data3[L3 >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.0)
		# xray3_percentile_75 = ax3.plot(np.nanmedian(x_data3[L3 >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data3[L3 >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.0)

		ax3.set_aspect(1)
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(6E-5,7E2)
		ax3.set_ylim(1E-4,120)
		ax3.set_yticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.set_xticklabels(xticks_labels)
		ax3.set_title('Stripe 82X', fontsize=21)
		ax3.text(0.05,0.8,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax3.text(-0.2,0.625,r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',transform=ax2.transAxes,rotation=90,fontsize=27)

		# upper_lim_points3 = ax3.plot(median_upper_wave3,median_upper3,'v',ms=10,color='k')
		# upper_lim_line3 = ax3.plot(median_upper_wave3,median_upper3,color='k',lw=4.0)
		# upper_lim_line3_25 = ax3.plot(median_upper_wave3,upper_25_3,'--',color='k',lw=3.0)
		# upper_lim_line3_75 = ax3.plot(median_upper_wave3, upper_75_3, '--', color='k', lw=3.0)
		ax3.plot(median_upper_wave3,median_upper3*scale_s82x,'v',ms=10,color='k')
		ax3.plot(median_upper_wave3,median_upper3*scale_s82x,color='k',lw=4.0)
		ax3.plot(np.append(np.nanmedian(10**median_wavelength[B3],axis=0)[-1],median_upper_wave32),np.append(np.nanmedian(10**median_flux[B3],axis=0)[-1],median_upper32)*scale_s82x,'--',color='k',lw=4.0)

		# ax3.plot(wfir_3,ffir_3,'v',ms=10,color='k')
		# ax3.plot(wfir_3,ffir_3,color='k',lw=4.0)
		# ax3.plot(wfir_3,ffir_3,'--',color='k',lw=3.0)
		# ax3.plot(wfir_3, ffir_3, '--', color='k', lw=3.0)
		# ax3.plot(wfir_3,ffir_25_3,'--',color='k',lw=3.0)
		# ax3.plot(wfir_3,ffir_75_3,'--',color='k',lw=3.0)

		ax1.grid()
		ax2.grid()
		ax3.grid()
		# cbar_ax = fig.add_subplot(gs1[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		# cb = fig.colorbar(test,cax=cbar_ax)
		# cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]',fontsize=21)

		plt.savefig('/Users/connor_auge/Desktop/New_plots3/Multi_SEDs_field'+savestring+'.pdf')
		plt.show()

	def plot_1panel(self,savestring,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,suptitle=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None,wfir=None,ffir=None):

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-3] = np.nan
		y[y < 1E-3] = np.nan

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		z_med = np.nanmedian(spec_z)

		clim1 = 43
		clim2 = 45.5

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq



			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]



			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		# B1_check = (uv_slope < -0.3) & (mir_slope1 >= -0.2)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		alpha = 0.7

		
		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]


		plt.rcParams['font.size']=24
		plt.rcParams['axes.linewidth']=2.5
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10,100]
		z = spec_z

	
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		z1 = spec_z[B1]
		cosmos_s82x_list_1 = cosmos_s82x_list[B1]
		median_upper_wave1 = cosmos_s82x_wave[B1]
		cosmos_s82x_list_12 = cosmos_s82x_list2[B1]
		median_upper_wave12 = cosmos_s82x_wave2[B1]
		median_wave1 = 10**median_wavelength[B1]
		median_flux1 = 10**median_flux[B1]
		# wfir1 = np.asarray([wfir[B1][i]/norm1[i] for i in range(len(wfir[B1]))])
		wfir1 = wfir[B1]
		ffir1 = np.asarray([ffir[B1][i]/norm1[i] for i in range(len(ffir[B1]))])

		# print(uv_slope[(z >= zlim_2) & (z <= zlim_3)])
		# print(mir_slope1[(z >= zlim_2) & (z <= zlim_3)])
		
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		z2 = spec_z[B2]
		cosmos_s82x_list_2 = cosmos_s82x_list[B2]
		median_upper_wave2 = cosmos_s82x_wave[B2]
		cosmos_s82x_list_22 = cosmos_s82x_list2[B2]
		median_upper_wave22 = cosmos_s82x_wave2[B2]
		median_wave2 = 10**median_wavelength[B2]
		median_flux2 = 10**median_flux[B2]
		# wfir2 = np.asarray([wfir[B2][i]/norm2[i] for i in range(len(wfir[B2]))])
		wfir2 = wfir[B2]
		ffir2 = np.asarray([ffir[B2][i]/norm2[i] for i in range(len(ffir[B2]))])
	
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		z3 = spec_z[B3]
		cosmos_s82x_list_3 = cosmos_s82x_list[B3]
		median_upper_wave3 = cosmos_s82x_wave[B3]
		cosmos_s82x_list_32 = cosmos_s82x_list2[B3]
		median_upper_wave32 = cosmos_s82x_wave2[B3]
		median_wave3 = 10**median_wavelength[B3]
		median_flux3 = 10**median_flux[B3]
		# wfir3 = np.asarray([wfir[B3][i]/norm3[i] for i in range(len(wfir[B3]))])
		wfir3 = wfir[B3]
		ffir3 = np.asarray([ffir[B3][i]/norm3[i] for i in range(len(ffir[B3]))])
		
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		z4 = spec_z[B4]
		cosmos_s82x_list_4 = cosmos_s82x_list[B4]
		median_upper_wave4 = cosmos_s82x_wave[B4]
		cosmos_s82x_list_42 = cosmos_s82x_list2[B4]
		median_upper_wave42 = cosmos_s82x_wave2[B4]
		median_wave4 = 10**median_wavelength[B4]
		median_flux4 = 10**median_flux[B4]
		# wfir4 = np.asarray([wfir[B4][i]/norm4[i] for i in range(len(wfir[B4]))])
		wfir4 = wfir[B4]
		ffir4 = np.asarray([ffir[B4][i]/norm4[i] for i in range(len(ffir[B4]))])

		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		z5 = spec_z[B5]
		cosmos_s82x_list_5 = cosmos_s82x_list[B5]
		median_upper_wave5 = cosmos_s82x_wave[B5]
		cosmos_s82x_list_52 = cosmos_s82x_list2[B5]
		median_upper_wave52 = cosmos_s82x_wave2[B5]
		median_wave5 = 10**median_wavelength[B5]
		median_flux5 = 10**median_flux[B5]
		# wfir5 = np.asarray([wfir[B5][i]/norm5[i] for i in range(len(wfir[B5]))])
		wfir5 = wfir[B5]
		ffir5 = np.asarray([ffir[B5][i]/norm5[i] for i in range(len(ffir[B5]))])

		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']
		# xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]

		


		fig = plt.figure(figsize=(14,10))
		gs = fig.add_gridspec(nrows=1, ncols=2,width_ratios=[3.25,0.1])
		gs.update(left=0.1, right=0.85, top=0.9, bottom=0.1)
		gs.update(hspace=0.07,wspace=-0.3) # set the spacing between axes
		


		# ax1 = plt.subplot(gs[0,0])

		# upper_seg1 = np.stack((median_upper_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)], cosmos_s82x_list_1[(z1 >= zlim_1) & (z1 <= zlim_2)]), axis=2)
		# upper_all1 = LineCollection(upper_seg1,color='gray',alpha=0.3)
		# ax1.add_collection(upper_all1)

		# test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		# lc1 = self.multilines(x1[(z1 >= zlim_1) & (z1 <= zlim_2)], y1[(z1 >= zlim_1) & (z1 <= zlim_2)], L1[(z1 >= zlim_1) & (z1 <= zlim_2)], cmap='rainbow_r', lw=1.5, alpha = alpha, rasterized=True)
		# ax1.plot(np.nanmedian(median_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmedian(median_flux1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),color='k',lw=3.5)
		# axcb1 = fig.colorbar(lc1)
		# axcb1.mappable.set_clim(clim1,clim2)
		# ax1.plot(np.nanmedian(median_upper_wave12[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax1.plot(np.nanmedian(median_upper_wave12[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmedian(cosmos_s82x_list_12[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), color='k', lw=2.0)
		# # ax1.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# # ax1.plot(np.nanmedian(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmedian(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# # ax1.plot(np.nanmean(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmean(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# axcb1.remove()
	

		# ax1.set_aspect(1)
		# ax1.set_xscale('log')
		# ax1.set_yscale('log')
		# ax1.set_xlim(8E-5,7E2)
		# ax1.set_ylim(1E-4,120)
		# ax1.set_xticklabels([])
		# ax1.set_xticks(xticks)
		# ax1.set_yticks(yticks)
		# # ax1.set_xticklabels(xticks_labels)
		# ax1.text(0.05,0.7,f'n = {len(x1[(z1 >= zlim_1) & (z1 <= zlim_2)])}',transform=ax1.transAxes)
		# ax1.text(0.75,0.08,str((len(x1[(z1 >= zlim_1) & (z1 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		# # ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax1.text(0.0,1.03,r'A',transform=ax1.transAxes,fontsize=27,weight='bold')
		# ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		# ax1.text(-0.45,0.5,'1',transform=ax1.transAxes,fontsize=38,weight='bold')

		# ax2 = plt.subplot(gs[1,0])

		# upper_seg2 = np.stack((median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)], cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)]), axis=2)
		# upper_all2 = LineCollection(upper_seg2,color='gray',alpha=0.3)
		# ax2.add_collection(upper_all2)

		# lc2 = self.multilines(x2[(z2 >= zlim_1) & (z2 <= zlim_2)],y2[(z2 >= zlim_1) & (z2 <= zlim_2)],L2[(z2 >= zlim_1) & (z2 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax2.plot(np.nanmedian(median_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(median_flux2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), color='k', lw=3.5)
		# axcb2 = fig.colorbar(lc2)
		# axcb2.mappable.set_clim(clim1,clim2)
		# ax2.plot(np.nanmedian(median_upper_wave22[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax2.plot(np.nanmedian(median_upper_wave22[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(cosmos_s82x_list_22[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), color='k', lw=2.0)
		# # ax2.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# # ax2.plot(np.nanmedian(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# # ax2.plot(np.nanmean(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmean(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)		
		# # axcb2.remove()

		# ax2.set_aspect(1)
		# ax2.set_xscale('log')
		# ax2.set_yscale('log')
		# ax2.set_xlim(8E-5,7E2)
		# ax2.set_ylim(1E-4,120)
		# ax2.set_xticklabels([])
		# ax2.set_xticks(xticks)
		# ax2.set_yticks(yticks)
		# # ax2.set_xticklabels(xticks_labels)
		# ax2.text(0.05,0.7,f'n = {len(x2[(z2 >= zlim_1) & (z2 <= zlim_2)])}',transform=ax2.transAxes)
		# ax2.text(0.75, 0.08, str((len(x2[(z2 >= zlim_1) & (z2 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%', transform=ax2.transAxes, weight='bold')
		# # ax2.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax2.text(-0.45,0.5,'2',transform=ax2.transAxes,fontsize=38,weight='bold')

		# ax3 = plt.subplot(gs[2,0])

		# upper_seg3 = np.stack((median_upper_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)], cosmos_s82x_list_3[(z3 >= zlim_1) & (z3 <= zlim_2)]), axis=2)
		# upper_all3 = LineCollection(upper_seg3,color='gray',alpha=0.3)
		# ax3.add_collection(upper_all3)

		# lc3 = self.multilines(x3[(z3 >= zlim_1) & (z3 <= zlim_2)],y3[(z3 >= zlim_1) & (z3 <= zlim_2)],L3[(z3 >= zlim_1) & (z3 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax3.plot(np.nanmedian(median_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(median_flux3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),color='k',lw=3.5)
		# axcb3 = fig.colorbar(lc3)
		# axcb3.mappable.set_clim(clim1,clim2)
		# ax3.plot(np.nanmedian(median_upper_wave32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax3.plot(np.nanmedian(median_upper_wave32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),color='k',lw=2.0)
		# # ax3.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# # ax3.plot(np.nanmedian(wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0), np.nanmedian(ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# # ax3.plot(np.nanmean(wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0), np.nanmean(ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# axcb3.remove()

		# ax3.set_aspect(1)
		# ax3.set_xscale('log')
		# ax3.set_yscale('log')
		# ax3.set_xlim(8E-5,7E2)
		# ax3.set_ylim(1E-4,120)
		# ax3.set_xticklabels([])
		# ax3.set_xticks(xticks)
		# ax3.set_yticks(yticks)
		# # ax3.set_xticklabels(xticks_labels)
		# ax3.text(0.05,0.7,f'n = {len(x3[(z3 >= zlim_1) & (z3 <= zlim_2)])}',transform=ax3.transAxes)
		# ax3.text(0.75,0.08,str((len(x3[(z3 >= zlim_1) & (z3 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		# ax3.text(-0.45,0.5,'3',transform=ax3.transAxes,fontsize=38,weight='bold')

		# ax4 = plt.subplot(gs[3,0])

		# upper_seg4 = np.stack((median_upper_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)], cosmos_s82x_list_4[(z4 >= zlim_1) & (z4 <= zlim_2)]), axis=2)
		# upper_all4 = LineCollection(upper_seg4,color='gray',alpha=0.3)
		# ax4.add_collection(upper_all4)

		# lc4 = self.multilines(x4[(z4 >= zlim_1) & (z4 <= zlim_2)],y4[(z4 >= zlim_1) & (z4 <= zlim_2)],L4[(z4 >= zlim_1) & (z4 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax4.plot(np.nanmedian(median_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(median_flux4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),color='k',lw=3.5)
		# axcb4 = fig.colorbar(lc4)
		# axcb4.mappable.set_clim(clim1,clim2)
		# ax4.plot(np.nanmedian(median_upper_wave42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax4.plot(np.nanmedian(median_upper_wave42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),color='k',lw=2.0)
		# # ax4.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# # ax4.plot(np.nanmedian(wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0), np.nanmedian(ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# # ax4.plot(np.nanmean(wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0), np.nanmean(ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# # axcb4.remove()

		# ax4.set_aspect(1)
		# ax4.set_xscale('log')
		# ax4.set_yscale('log')
		# ax4.set_xlim(8E-5,7E2)
		# ax4.set_ylim(1E-4,120)
		# ax4.set_xticklabels([])
		# ax4.set_xticks(xticks)
		# ax4.set_yticks(yticks)
		# # ax4.set_xticklabels(xticks_labels)
		# ax4.text(0.05,0.7,f'n = {len(x4[(z4 >= zlim_1) & (z4 <= zlim_2)])}',transform=ax4.transAxes)
		# ax4.text(0.75,0.08,str((len(x4[(z4 >= zlim_1) & (z4 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		# # ax4.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax4.text(-0.45,0.5,'4',transform=ax4.transAxes,fontsize=38,weight='bold')

		# ax5 = plt.subplot(gs[4,0])

		# upper_seg5 = np.stack((median_upper_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)], cosmos_s82x_list_5[(z5 >= zlim_1) & (z5 <= zlim_2)]), axis=2)
		# upper_all5 = LineCollection(upper_seg5,color='gray',alpha=0.3)
		# ax5.add_collection(upper_all5)

		# lc5 = self.multilines(x5[(z5 >= zlim_1) & (z5 <= zlim_2)],y5[(z5 >= zlim_1) & (z5 <= zlim_2)],L5[(z5 >= zlim_1) & (z5 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax5.plot(np.nanmedian(median_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(median_flux5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),color='k',lw=3.5)
		# axcb5 = fig.colorbar(lc5)
		# axcb5.mappable.set_clim(clim1,clim2)
		# ax5.plot(np.nanmedian(median_upper_wave52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax5.plot(np.nanmedian(median_upper_wave52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),color='k',lw=2.0)
		# # ax5.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# # ax5.plot(np.nanmedian(wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0), np.nanmedian(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# # ax5.plot(np.nanmean(wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0), np.nanmean(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# # axcb5.remove()

		# ax5.set_aspect(1)
		# ax5.set_xscale('log')
		# ax5.set_yscale('log')
		# ax5.set_xlim(8E-5,7E2)
		# ax5.set_ylim(1E-4,120)
		# ax5.set_xticks(xticks)
		# ax5.set_yticks(yticks)
		# ax5.set_xticklabels(xticks_labels)
		# ax5.text(0.05,0.7,f'n = {len(x5[(z5 >= zlim_1) & (z5 <= zlim_2)])}',transform=ax5.transAxes)
		# ax5.text(0.75,0.08,str((len(x5[(z5 >= zlim_1) & (z5 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		# # ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		# # ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		# ax5.text(-0.45,0.5,'5',transform=ax5.transAxes,fontsize=38,weight='bold')




		# ax6 = plt.subplot(gs[0,1])

		# upper_seg6 = np.stack((median_upper_wave1[(z1 > zlim_2) & (z1 <= zlim_3)], cosmos_s82x_list_1[(z1 > zlim_2) & (z1 <= zlim_3)]), axis=2)
		# upper_all6 = LineCollection(upper_seg6,color='gray',alpha=0.3)
		# ax6.add_collection(upper_all6)

		# lc6 = self.multilines(x1[(z1 > zlim_2) & (z1 <= zlim_3)],y1[(z1 > zlim_2) & (z1 <= zlim_3)],L1[(z1 > zlim_2) & (z1 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax6.plot(np.nanmedian(median_wave1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(median_flux1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),color='k',lw=3.5)
		# axcb6 = fig.colorbar(lc6)
		# axcb6.mappable.set_clim(clim1,clim2)
		# ax6.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax6.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),color='k',lw=2.0)
		# # ax6.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# # ax6.plot(np.nanmedian(wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0), np.nanmedian(ffir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0),'-x',color='orange',lw=1.75)
		# # ax6.plot(np.nanmean(wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0), np.nanmean(ffir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# # axcb6.remove()

		# ax6.set_aspect(1)
		# ax6.set_xscale('log')
		# ax6.set_yscale('log')
		# ax6.set_xlim(8E-5,7E2)
		# ax6.set_ylim(1E-4,120)
		# ax6.set_xticklabels([])
		# ax6.set_yticklabels([])
		# ax6.set_xticks(xticks)
		# ax6.set_yticks(yticks)
		# # ax6.set_xticklabels(xticks_labels)
		# ax6.text(0.05,0.7,f'n = {len(x1[(z1 > zlim_2) & (z1 <= zlim_3)])}',transform=ax6.transAxes)
		# ax6.text(0.75,0.08,str((len(x1[(z1 > zlim_2) & (z1 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax6.transAxes,weight='bold')
		# ax6.text(0.0,1.03,r'B',transform=ax6.transAxes,fontsize=27,weight='bold')
		# ax6.set_title(str(zlim_2)+' < z < '+str(zlim_3))

		# ax7 = plt.subplot(gs[1,1])

		# upper_seg7 = np.stack((median_upper_wave2[(z2 > zlim_2) & (z2 <= zlim_3)], cosmos_s82x_list_2[(z2 > zlim_2) & (z2 <= zlim_3)]), axis=2)
		# upper_all7 = LineCollection(upper_seg7,color='gray',alpha=0.3)
		# ax7.add_collection(upper_all7)

		# lc7 = self.multilines(x2[(z2 > zlim_2) & (z2 <= zlim_3)],y2[(z2 > zlim_2) & (z2 <= zlim_3)],L2[(z2 > zlim_2) & (z2 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax7.plot(np.nanmedian(median_wave2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), np.nanmedian(median_flux2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), color='k', lw=3.5)		
		# axcb7 = fig.colorbar(lc7)
		# axcb7.mappable.set_clim(clim1,clim2)
		# ax7.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax7.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),color='k',lw=2.0)
		# # ax7.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# # ax7.plot(np.nanmedian(wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), np.nanmedian(ffir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0),'-x',color='orange',lw=1.75)
		# # ax7.plot(np.nanmean(wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), np.nanmean(ffir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# # axcb7.remove()

		# ax7.set_aspect(1)
		# ax7.set_xscale('log')
		# ax7.set_yscale('log')
		# ax7.set_xlim(8E-5,7E2)
		# ax7.set_ylim(1E-4,120)
		# ax7.set_xticklabels([])
		# ax7.set_yticklabels([])
		# ax7.set_xticks(xticks)
		# ax7.set_yticks(yticks)
		# # ax7.set_xticklabels(xticks_labels)
		# ax7.text(0.05,0.7,f'n = {len(x2[(z2 > zlim_2) & (z2 <= zlim_3)])}',transform=ax7.transAxes)
		# ax7.text(0.75,0.08,str((len(x2[(z2 > zlim_2) & (z2 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax7.transAxes,weight='bold')

		# ax8 = plt.subplot(gs[2,1])

		# upper_seg8 = np.stack((median_upper_wave3[(z3 > zlim_2) & (z3 <= zlim_3)], cosmos_s82x_list_3[(z3 > zlim_2) & (z3 <= zlim_3)]), axis=2)
		# upper_all8 = LineCollection(upper_seg8,color='gray',alpha=0.3)
		# ax8.add_collection(upper_all8)

		# lc8 = self.multilines(x3[(z3 > zlim_2) & (z3 <= zlim_3)],y3[(z3 > zlim_2) & (z3 <= zlim_3)],L3[(z3 > zlim_2) & (z3 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax8.plot(np.nanmedian(median_wave3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(median_flux3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),color='k',lw=3.5)
		# axcb8 = fig.colorbar(lc8)
		# axcb8.mappable.set_clim(clim1,clim2)
		# ax8.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax8.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),color='k',lw=2.0)
		# # ax8.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# # ax8.plot(np.nanmedian(wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0), np.nanmedian(ffir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0),'-x',color='orange',lw=1.75)
		# # ax8.plot(np.nanmean(wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0), np.nanmean(ffir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# axcb8.remove()

		# ax8.set_aspect(1)
		# ax8.set_xscale('log')
		# ax8.set_yscale('log')
		# ax8.set_xlim(8E-5,7E2)
		# ax8.set_ylim(1E-4,120)
		# ax8.set_xticklabels([])
		# ax8.set_yticklabels([])
		# ax8.set_xticks(xticks)
		# ax8.set_yticks(yticks)
		# # ax8.set_xticklabels(xticks_labels)
		# ax8.text(0.05,0.7,f'n = {len(x3[(z3 > zlim_2) & (z3 <= zlim_3)])}',transform=ax8.transAxes)
		# ax8.text(0.75,0.08,str((len(x3[(z3 > zlim_2) & (z3 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax8.transAxes,weight='bold')

		# ax9 = plt.subplot(gs[3,1])

		# upper_seg9 = np.stack((median_upper_wave4[(z4 > zlim_2) & (z4 <= zlim_3)], cosmos_s82x_list_4[(z4 > zlim_2) & (z4 <= zlim_3)]), axis=2)
		# upper_all9 = LineCollection(upper_seg9,color='gray',alpha=0.3)
		# ax9.add_collection(upper_all9)

		# lc9 = self.multilines(x4[(z4 > zlim_2) & (z4 <= zlim_3)],y4[(z4 > zlim_2) & (z4 <= 0.8)],L4[(z4 > zlim_2) & (z4 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax9.plot(np.nanmedian(median_wave4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(median_flux4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),color='k',lw=3.5)
		# axcb9 = fig.colorbar(lc9)
		# axcb9.mappable.set_clim(clim1,clim2)
		# ax9.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax9.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),color='k',lw=2.0)
		# # ax9.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# # ax9.plot(np.nanmedian(wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0), np.nanmedian(ffir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0),'-x',color='orange',lw=1.75)
		# # ax9.plot(np.nanmean(wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0), np.nanmean(ffir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# # axcb9.remove()

		# ax9.set_aspect(1)
		# ax9.set_xscale('log')
		# ax9.set_yscale('log')
		# ax9.set_xlim(8E-5,7E2)
		# ax9.set_ylim(1E-4,120)
		# ax9.set_xticklabels([])
		# ax9.set_yticklabels([])
		# ax9.set_xticks(xticks)
		# ax9.set_yticks(yticks)
		# # ax9.set_xticklabels(xticks_labels)
		# ax9.text(0.05,0.7,f'n = {len(x4[(z4 > zlim_2) & (z4 <= zlim_3)])}',transform=ax9.transAxes)
		# ax9.text(0.75,0.08,str((len(x4[(z4 > zlim_2) & (z4 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax9.transAxes,weight='bold')

		# ax10 = plt.subplot(gs[4,1])

		# upper_seg10 = np.stack((median_upper_wave5[(z5 > zlim_2) & (z5 <= zlim_3)], cosmos_s82x_list_5[(z5 > zlim_2) & (z5 <= zlim_3)]), axis=2)
		# upper_all10 = LineCollection(upper_seg10,color='gray',alpha=0.3)
		# ax10.add_collection(upper_all10)

		# lc10 = self.multilines(x5[(z5 > zlim_2) & (z5 <= zlim_3)],y5[(z5 > zlim_2) & (z5 <= 0.8)],L5[(z5 > zlim_2) & (z5 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax10.plot(np.nanmedian(median_wave5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(median_flux5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),color='k',lw=3.5)
		# axcb10 = fig.colorbar(lc10)	
		# axcb10.mappable.set_clim(clim1,clim2)
		# ax10.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax10.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),color='k',lw=2.0)
		# # ax10.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# # ax10.plot(np.nanmedian(wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0), np.nanmedian(ffir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0),'-x',color='orange',lw=1.75)
		# # ax10.plot(np.nanmean(wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0), np.nanmean(ffir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)		
		# axcb10.remove()

		# ax10.set_aspect(1)
		# ax10.set_xscale('log')
		# ax10.set_yscale('log')
		# ax10.set_xlim(8E-5,7E2)
		# ax10.set_ylim(1E-4,120)
		# ax10.set_yticklabels([])
		# ax10.set_xticks(xticks)
		# ax10.set_yticks(yticks)
		# ax10.set_xticklabels(xticks_labels)
		# ax10.text(0.05,0.7,f'n = {len(x5[(z5 > zlim_2) & (z5 <= zlim_3)])}',transform=ax10.transAxes)
		# ax10.text(0.75,0.08,str((len(x5[(z5 > zlim_2) & (z5 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax10.transAxes,weight='bold')
		# ax10.set_xlabel(r'Rest Wavelength [$\mu$m]', fontsize=40)




		ax11 = plt.subplot(gs[0,0])

		upper_seg11 = np.stack((median_upper_wave1[(z1 > zlim_3) & (z1 <= zlim_4)], cosmos_s82x_list_1[(z1 > zlim_3) & (z1 <= zlim_4)]), axis=2)
		upper_all11 = LineCollection(upper_seg11,color='gray',alpha=0.3)
		ax11.add_collection(upper_all11)
		
		test = ax11.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc11 = self.multilines(x1[(z1 > zlim_3) & (z1 <= zlim_4)],y1[(z1 > zlim_3) & (z1 <= zlim_4)],L1[(z1 > zlim_3) & (z1 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=2,rasterized=True)
		ax11.plot(x1[(z1 > zlim_3) & (z1 <= zlim_4)],y1[(z1 > zlim_3) & (z1 <= zlim_4)],'x',color='k', alpha = alpha,lw=2,rasterized=True)
		ax11.plot(np.nanmedian(median_wave1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(median_flux1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),color='k',lw=4)
		axcb11 = fig.colorbar(lc11)
		axcb11.mappable.set_clim(clim1,clim2)
		ax11.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),'v',ms=10,color='k')
		ax11.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),color='k',lw=4.0)
		# ax11.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax11.plot(np.nanmedian(wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0), np.nanmedian(ffir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0),'-x',color='orange',lw=1.75)
		# ax11.plot(np.nanmean(wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0), np.nanmean(ffir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		ax11.plot(np.nanmedian(x1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0)[0:2],np.nanmedian(y1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0)[0:2],color='k',lw=4)
		axcb11.remove()

		# ax11.set_aspect(1)
		ax11.set_xscale('log')
		ax11.set_yscale('log')
		ax11.set_xlim(8E-5,7E2)
		ax11.set_ylim(1E-2,50)
		# ax11.set_xticklabels([])
		# ax11.set_yticklabels([])
		ax11.set_xticks(xticks)
		ax11.set_yticks(yticks)
		# ax11.set_xticklabels(xticks_labels)
		ax11.text(0.05,0.7,f'n = {len(x1[(z1 > zlim_3) & (z1 <= zlim_4)])}',transform=ax11.transAxes)
		# ax11.text(0.75,0.08,str((len(x1[(z1 > zlim_3) & (z1 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax11.transAxes,weight='bold')
		# ax11.text(0.0,1.03,r'C',transform=ax11.transAxes,fontsize=27,weight='bold')
		ax11.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		ax11.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax11.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')



		# ax12 = plt.subplot(gs[1,2])

		# upper_seg12 = np.stack((median_upper_wave2[(z2 > zlim_3) & (z2 <= zlim_4)], cosmos_s82x_list_2[(z2 > zlim_3) & (z2 <= zlim_4)]), axis=2)
		# upper_all12 = LineCollection(upper_seg12,color='gray',alpha=0.3)
		# ax12.add_collection(upper_all12)

		# test = ax12.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		# lc12 = self.multilines(x2[(z2 > zlim_3) & (z2 <= zlim_4)],y2[(z2 > zlim_3) & (z2 <= zlim_4)],L2[(z2 > zlim_3) & (z2 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax12.plot(np.nanmedian(median_wave2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(median_flux2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),color='k',lw=3.5)		
		# axcb12 = fig.colorbar(lc12)
		# axcb12.mappable.set_clim(clim1,clim2)
		# ax12.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax12.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),color='k',lw=2.0)
		# # ax12.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# # ax12.plot(np.nanmedian(wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0), np.nanmedian(ffir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0),'-x',color='orange',lw=1.75)
		# # ax12.plot(np.nanmean(wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0), np.nanmean(ffir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		# axcb12.remove()

		# ax12.set_aspect(1)
		# ax12.set_xscale('log')
		# ax12.set_yscale('log')
		# ax12.set_xlim(8E-5,7E2)
		# ax12.set_ylim(1E-4,120)
		# ax12.set_xticklabels([])
		# ax12.set_yticklabels([])
		# ax12.set_xticks(xticks)
		# ax12.set_yticks(yticks)
		# # ax12.set_xticklabels(xticks_labels)
		# ax12.text(0.05,0.7,f'n = {len(x2[(z2 > zlim_3) & (z2 <= zlim_4)])}',transform=ax12.transAxes)
		# ax12.text(0.75,0.08,str((len(x2[(z2 > zlim_3) & (z2 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax12.transAxes,weight='bold')

		# ax13 = plt.subplot(gs[2,2])

		# upper_seg13 = np.stack((median_upper_wave3[(z3 > zlim_3) & (z3 <= zlim_4)], cosmos_s82x_list_3[(z3 > zlim_3) & (z3 <= zlim_4)]), axis=2)
		# upper_all13 = LineCollection(upper_seg13,color='gray',alpha=0.3)
		# ax13.add_collection(upper_all13)

		# test = ax13.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		# lc13 = self.multilines(x3[(z3 > zlim_3) & (z3 <= zlim_4)],y3[(z3 > zlim_3) & (z3 <= zlim_4)],L3[(z3 > zlim_3) & (z3 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax13.plot(np.nanmedian(median_wave3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(median_flux3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),color='k',lw=3.5)		
		# axcb13 = fig.colorbar(lc13)
		# axcb13.mappable.set_clim(clim1,clim2)
		# ax13.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax13.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),color='k',lw=2.0)
		# # ax13.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# # ax13.plot(np.nanmedian(wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0), np.nanmedian(ffir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0),'-x',color='orange',lw=1.75)
		# # ax13.plot(np.nanmean(wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0), np.nanmean(ffir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		# axcb13.remove()

		# ax13.set_aspect(1)
		# ax13.set_xscale('log')
		# ax13.set_yscale('log')
		# ax13.set_xlim(8E-5,7E2)
		# ax13.set_ylim(1E-4,120)
		# ax13.set_xticklabels([])
		# ax13.set_yticklabels([])
		# ax13.set_xticks(xticks)
		# ax13.set_yticks(yticks)
		# # ax13.set_xticklabels(xticks_labels)
		# ax13.text(0.05,0.7,f'n = {len(x3[(z3 > zlim_3) & (z3 <= zlim_4)])}',transform=ax13.transAxes)
		# ax13.text(0.75,0.08,str((len(x3[(z3 > zlim_3) & (z3 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax13.transAxes,weight='bold')

		# ax14 = plt.subplot(gs[3,2])

		# upper_seg14 = np.stack((median_upper_wave4[(z4 > zlim_3) & (z4 <= zlim_4)], cosmos_s82x_list_4[(z4 > zlim_3) & (z4 <= zlim_4)]), axis=2)
		# upper_all14 = LineCollection(upper_seg14,color='gray',alpha=0.3)
		# ax14.add_collection(upper_all14)

		# test = ax14.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		# lc14 = self.multilines(x4[(z4 > zlim_3) & (z4 <= zlim_4)],y4[(z4 > zlim_3) & (z4 <= zlim_4)],L4[(z4 > zlim_3) & (z4 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		# ax14.plot(np.nanmedian(median_wave4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(median_flux4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),color='k',lw=3.5)
		# axcb14 = fig.colorbar(lc14)
		# axcb14.mappable.set_clim(clim1,clim2)
		# ax14.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax14.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),color='k',lw=2.0)
		# # ax14.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# # ax14.plot(np.nanmedian(wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0), np.nanmedian(ffir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0),'-x',color='orange',lw=1.75)
		# # ax14.plot(np.nanmean(wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0), np.nanmean(ffir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		# axcb14.remove()

		# ax14.set_aspect(1)
		# ax14.set_xscale('log')
		# ax14.set_yscale('log')
		# ax14.set_xlim(8E-5,7E2)
		# ax14.set_ylim(1E-4,120)
		# ax14.set_xticklabels([])
		# ax14.set_yticklabels([])
		# ax14.set_xticks(xticks)
		# ax14.set_yticks(yticks)
		# # ax14.set_xticklabels(xticks_labels)
		# ax14.text(0.05,0.7,f'n = {len(x4[(z4 > zlim_3) & (z4 <= zlim_4)])}',transform=ax14.transAxes)
		# ax14.text(0.75,0.08,str((len(x4[(z4 > zlim_3) & (z4<= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax14.transAxes,weight='bold')

		# ax15 = plt.subplot(gs[4,2])

		# upper_seg15 = np.stack((median_upper_wave5[(z5 > zlim_3) & (z5 <= zlim_4)], cosmos_s82x_list_5[(z5 > zlim_3) & (z5 <= zlim_4)]), axis=2)
		# upper_all15 = LineCollection(upper_seg15,color='gray',alpha=0.3)
		# ax15.add_collection(upper_all15)

		# test = ax15.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		# lc15 = self.multilines(x5[(z5 > zlim_3) & (z5 <= zlim_4)], y5[(z5 > zlim_3) & (z5 <= 1.1)], L5[(z5 > zlim_3) & (z5 <= zlim_4)], cmap='rainbow_r', alpha=alpha, lw=1.5, rasterized=True)
		# ax15.plot(np.nanmedian(median_wave5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(median_flux5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),color='k',lw=3.5)		
		# axcb15 = fig.colorbar(lc15)
		# axcb15.mappable.set_clim(clim1,clim2)
		# ax15.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax15.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),color='k',lw=2.0)
		# # ax15.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# # ax15.plot(np.nanmedian(wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0), np.nanmedian(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0),'-x',color='orange',lw=1.75)
		# # ax15.plot(np.nanmean(wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0), np.nanmean(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		# axcb15.remove()

		# ax15.set_aspect(1)
		# ax15.set_xscale('log')
		# ax15.set_yscale('log')
		# ax15.set_xlim(8E-5,7E2)
		# ax15.set_ylim(1E-4, 120)
		# # ax15.set_xticklabels([])
		# ax15.set_yticklabels([])
		# ax15.set_xticks(xticks)
		# ax15.set_yticks(yticks)
		# ax15.set_xticklabels(xticks_labels)
		# ax15.text(0.05,0.7,f'n = {len(x5[(z5 > zlim_3) & (z5 <= zlim_4)])}',transform=ax15.transAxes)
		# ax15.text(0.75,0.08,str((len(x5[(z5 > zlim_3) & (z5<= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax15.transAxes,weight='bold')

		
		# ax1.grid()
		# ax2.grid()
		# ax3.grid()
		# ax4.grid()
		# ax5.grid()
		# ax6.grid()
		# ax7.grid()
		# ax8.grid()
		# ax9.grid()
		# ax10.grid()
		ax11.grid()
		# ax12.grid()
		# ax13.grid()
		# ax14.grid()
		# ax15.grid()


		cbar_ax = fig.add_subplot(gs[:,-1:])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg/s]')

		# plt.tight_layout()
	
		plt.savefig('/Users/connor_auge/Desktop/final_paper_43/5paneles_zbins'+savestring+'.pdf')
		plt.show() 

	def SEDs_Lx_Lone(self,x,y,Lx,L1,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None):	
		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-3] = np.nan
		y[y < 1E-3] = np.nan

		z_med = np.nanmedian(spec_z)

		clim1 = 42.5
		clim2 = 46

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(Lx)
		Lone = np.asarray(L1)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)




		w250 = 2536859.83
		rest_w250 = w250/(1+z_med)
		rest_w250_cgs = rest_w250*1E-8
		rest_w250_microns = rest_w250*1E-4
		rest_w250_freq = 3E10/rest_w250_cgs

		upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
		rest_upper_w = upper_wave/(1+1.0)
		rest_upper_w_cgs = rest_upper_w*1E-8
		rest_upper_w_microns = rest_upper_w*1E-4
		rest_upper_w_freq = 3E10/rest_upper_w_cgs

		cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
		s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6

		# cosmos_upper_lim_jy = 8.1E3*1E-6
		# s82X_upper_lim_jy = 13.0E3*1E-6

		cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
		s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

		cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
		s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq

		cosmos_nuLnu_upper = Flux_to_Lum(cosmos_nuFnu_upper,z_med)
		s82X_nuLnu_upper = Flux_to_Lum(s82X_nuFnu_upper,z_med)

		cosmos_s82x_list = []
		for i in range(len(y)):
			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(cosmos_nuLnu_upper/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
			if mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(s82X_nuLnu_upper/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)



		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		# B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		# B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		# B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		# B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		# B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]


		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		cosmos_norm1 = norm1[mark1 == 0]
		s82X_norm1 = norm1[mark1 == 1]

		cosmos_norm2 = norm2[mark2 == 0]
		s82X_norm2 = norm2[mark2 == 1]

		cosmos_norm3 = norm3[mark3 == 0]
		s82X_norm3 = norm3[mark3 == 1]

		cosmos_norm4 = norm4[mark4 == 0]
		s82X_norm4 = norm4[mark4 == 1]

		cosmos_norm5 = norm5[mark5 == 0]
		s82X_norm5 = norm5[mark5 == 1]

		cosmos_s82x_list_1 = cosmos_s82x_list[B1]
		cosmos_s82x_list_2 = cosmos_s82x_list[B2]
		cosmos_s82x_list_3 = cosmos_s82x_list[B3]
		cosmos_s82x_list_4 = cosmos_s82x_list[B4]
		cosmos_s82x_list_5 = cosmos_s82x_list[B5]
	
		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]


		fig = plt.figure(figsize=(20,15),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=5, ncols=2, left=0.1,right=0.4,wspace=-0.25,hspace=0.1,width_ratios=[3,0.25])
		gs2 = fig.add_gridspec(nrows=5, ncols=2, left=0.5,right=0.95,wspace=0.1,hspace=0.1)

		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		Lone1 = Lone[B1]
		Lone1 = Lone1[np.isfinite(Lone1)]
		spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		ax1.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_1,axis=0),'v',ms=5,color='k')
		ax1.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_1,axis=0),color='k',lw=2.0)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		# ax1.text(0.9,0.08,'1',transform=ax1.transAxes,weight='bold')
		ax1.text(0.75,0.08,str((len(x1)/len(x))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax1.text(0.0,1.03,r'A',transform=ax1.transAxes,fontsize=27,weight='bold')
		ax1.set_title('0.3 < z < 1.1')


		ax2 = fig.add_subplot(gs1[1,0])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		Lone2 = Lone[B2]
		Lone2 = Lone2[np.isfinite(Lone2)]
		spec_type2 = spec_type[B2]


		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		ax2.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_2,axis=0),'v',ms=5,color='k')
		ax2.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_2,axis=0),color='k',lw=2.0)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.text(0.75,0.08,str((len(x2)/len(x))*100)[0:4]+'%',transform=ax2.transAxes,weight='bold')
		# ax2.text(0.9,0.08,'2',transform=ax2.transAxes,weight='bold')

		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')
		

		ax3 = fig.add_subplot(gs1[2,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		Lone3 = Lone[B3]
		Lone3 = Lone3[np.isfinite(Lone3)]
		spec_type3 = spec_type[B3]

		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		ax3.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_3,axis=0),'v',ms=5,color='k')
		ax3.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_3,axis=0),color='k',lw=2.0)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax3.text(0.9,0.08,'3',transform=ax3.transAxes,weight='bold')
		ax3.text(0.75,0.08,str((len(x3)/len(x))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')



		ax4 = fig.add_subplot(gs1[3,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		Lone4 = Lone[B4]
		Lone4 = Lone4[np.isfinite(Lone4)]
		spec_type4 = spec_type[B4]

		lc4 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5,rasterized=True)
		ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		ax4.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_4,axis=0),'v',ms=5,color='k')
		ax4.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_4,axis=0),color='k',lw=2.0)
		axcb4.remove()

		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(5E-3,50)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		# ax4.text(0.9,0.08,'4',transform=ax4.transAxes,weight='bold')
		ax4.text(0.75,0.08,str((len(x4)/len(x))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax5 = fig.add_subplot(gs1[4,0])
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		Lone5 = Lone[B5]
		Lone5 = Lone5[np.isfinite(Lone5)]
		spec_type5 = spec_type[B5]

		x6 = x5
		y6 = y5

		lc5 = self.multilines(x6,y6,L5,cmap='rainbow',lw=1.5,rasterized=True)
		ax5.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0),color='k',lw=3.5)
		axcb5 = fig.colorbar(lc5)
		axcb5.mappable.set_clim(clim1,clim2)
		ax5.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_5,axis=0),'v',ms=5,color='k')
		ax5.plot(rest_upper_w_microns,np.nanmedian(cosmos_s82x_list_5,axis=0),color='k',lw=2.0)
		axcb5.remove()

		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(5E-3,50)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)

		ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		# ax5.text(0.9,0.08,'5',transform=ax5.transAxes,weight='bold')
		ax5.text(0.75,0.08,str((len(x5)/len(x))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()




		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)



		yticks = [0,10,20,30]
		
		cbar_ax = fig.add_subplot(gs1[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')


		ax6 = fig.add_subplot(gs2[0,0])
		ax6.hist(L1,bins=np.arange(42,47,0.25),color='gray')
		ax6.axvline(np.median(L1),ls='--',color='k',lw=3)
		ax6.set_xlim(42,46)
		ax6.set_ylim(0,35)
		ax6.set_yticks(yticks)
		ax6.set_xticklabels([])
		ax6.grid()
		ax6.text(0.0,1.03,r'B',transform=ax6.transAxes,fontsize=27,weight='bold')

		secax = ax6.secondary_xaxis('top',functions=(solar, ergs))
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [L$_{\odot}$]')

		ax7 = fig.add_subplot(gs2[1,0])
		ax7.hist(L2,bins=np.arange(42,47,0.25),color='gray')
		ax7.axvline(np.median(L2),ls='--',color='k',lw=3)
		ax7.set_xlim(42,46)
		ax7.set_ylim(0,35)
		ax7.set_yticks(yticks)
		ax7.set_xticklabels([])
		ax7.grid()

		ax8 = fig.add_subplot(gs2[2,0])
		ax8.hist(L3,bins=np.arange(42,47,0.25),color='gray')
		ax8.axvline(np.median(L3),ls='--',color='k',lw=3)
		ax8.set_xlim(42,46)
		ax8.set_ylim(0,35)
		ax8.set_yticks(yticks)
		ax8.set_xticklabels([])
		ax8.grid()

		ax9 = fig.add_subplot(gs2[3,0])
		ax9.hist(L4,bins=np.arange(42,47,0.25),color='gray')
		ax9.axvline(np.median(L4),ls='--',color='k',lw=3)
		ax9.set_xlim(42,46)
		ax9.set_ylim(0,35)
		ax9.set_yticks(yticks)
		ax9.set_xticklabels([])
		ax9.grid()

		ax10 = fig.add_subplot(gs2[4,0])
		ax10.hist(L5,bins=np.arange(42,47,0.25),color='gray')
		ax10.axvline(np.median(L5),ls='--',color='k',lw=3)
		ax10.set_xlim(42,46)
		ax10.set_ylim(0,35)
		ax10.set_yticks(yticks)
		ax10.set_xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
		ax10.grid()

		ax11 = fig.add_subplot(gs2[0,1])
		# ax11.hist(Lone1,bins=np.arange(44,49,0.25),color='gray')
		ax11.hist(Lone1[~np.isnan(Lone1)],bins=np.arange(7,10,0.25),color='gray')
		ax11.axvline(np.median(Lone1),ls='--',color='k',lw=3)
		# ax11.set_xlim(42,46)
		# ax11.set_xlim(44,48)
		ax11.set_xlim(7,10)
		ax11.set_ylim(0,35)
		ax11.set_yticks(yticks)
		ax11.set_yticklabels([])
		ax11.set_xticklabels([])
		ax11.grid()
		ax11.text(0.0,1.03,r'C',transform=ax11.transAxes,fontsize=27,weight='bold')


		# secax = ax11.secondary_xaxis('top',functions=(solar, ergs))
		# secax.set_xlabel(r'log $\mathrm{L}_{1\mu\mathrm{m}}$ [L$_{\odot}$]')

		ax12 = fig.add_subplot(gs2[1,1])
		# ax12.hist(Lone2,bins=np.arange(44,49,0.25),color='gray')
		ax12.hist(Lone2[~np.isnan(Lone2)],bins=np.arange(7,10,0.25),color='gray')		
		ax12.axvline(np.median(Lone2),ls='--',color='k',lw=3)
		# ax12.set_xlim(42,46)
		# ax12.set_xlim(44,48)
		ax12.set_xlim(7,10)
		ax12.set_ylim(0,35)
		ax12.set_yticks(yticks)
		ax12.set_yticklabels([])
		ax12.set_xticklabels([])
		ax12.grid()

		ax13 = fig.add_subplot(gs2[2,1])
		# ax13.hist(Lone3,bins=np.arange(44,49,0.25),color='gray')
		ax13.hist(Lone3[~np.isnan(Lone3)],bins=np.arange(7,10,0.25),color='gray')
		ax13.axvline(np.median(Lone3),ls='--',color='k',lw=3)
		# ax13.set_xlim(42,46)
		# ax13.set_xlim(44,48)
		ax13.set_xlim(7,10)
		ax13.set_ylim(0,35)
		ax13.set_yticks(yticks)
		ax13.set_yticklabels([])
		ax13.set_xticklabels([])
		ax13.grid()

		ax14 = fig.add_subplot(gs2[3,1])
		# ax14.hist(Lone4,bins=np.arange(44,49,0.25),color='gray')
		ax14.hist(Lone4[~np.isnan(Lone4)],bins=np.arange(7,10,0.25),color='gray')
		ax14.axvline(np.median(Lone4),ls='--',color='k',lw=3)
		# ax14.set_xlim(42,46)
		# ax14.set_xlim(44,48)
		ax14.set_xlim(7,10)
		ax14.set_ylim(0,35)
		ax14.set_yticks(yticks)
		ax14.set_yticklabels([])
		ax14.set_xticklabels([])
		ax14.grid()

		ax15 = fig.add_subplot(gs2[4,1])
		# ax15.hist(Lone5,bins=np.arange(44,49,0.25),color='gray')
		ax15.hist(Lone5[~np.isnan(Lone5)],bins=np.arange(7,10,0.25),color='gray')
		ax15.axvline(np.median(Lone5),ls='--',color='k',lw=3)
		# ax15.set_xlim(42,46)
		# ax15.set_xlim(44,48)
		ax15.set_xlim(7,10)
		ax15.set_ylim(0,35)
		ax15.set_yticks(yticks)
		ax15.set_yticklabels([])
		# ax15.set_xlabel(r'log L$_{1\mu\mathrm{m}}$ [erg/s]')
		ax15.set_xlabel(r'log M$_{\mathrm{BH}}$')
		ax15.grid()

		plt.savefig('/Users/connor_auge/Desktop/Paper/SED_Lx_Mbh_new.pdf')
		plt.show()

		print('bin 1: ', np.median(Lone1))
		print('bin 2: ', np.median(Lone2))
		print('bin 3: ', np.median(Lone3))


	def SEDs_UV_MIR_FIR(self,param,param2,Fx1,Fx2,Fx3,emis1,emis2,emis3,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,uv_slope=None,mir_slope1=None,mir_slope2=None):	
		print(np.shape(x))
		print(np.shape(y))

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.5
		clim2 = 46

		L = np.asarray(L)
		x = np.asarray(x)
		y = np.asarray(y)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)

		if param == 'hard':
			Fx = Fx1
			xlabel = r'log $\lambda$L$_{0.25\mu \mathrm{m}}$/$\lambda$L$_{2-10\mathrm{kev}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'log $\lambda$L$_{10\mu \mathrm{m}}$/$\lambda$L$_{0.5-2\mathrm{kev}}$'
		elif param == 'full':
			Fx = Fx3
			# xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-10\mathrm{kev}}$'
			xlabel1 = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(0.25\mu \mathrm{m})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'
			xlabel2 = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(10\mu \mathrm{m})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'

		if param2 == '01,10':
			
			legend1 = r'a = 0.25$\mu$m'
			legend2 = r'a = 5$\mu$m'
			# legend3 = r'a = 5$\mu$m')
			legend3 = r'a = 100$\mu$m'

			c1 = 'blue'
			c2 = 'red'
			c3 = 'green'

		elif param2 == '025,5':
			emis1 = 10**f1
			emis2 = 10**f2

			legend1 = r'a = 0.25$\mu$m'
			legend2 = r'a = 5$\mu$m'
			# legend3 = r'a = 5$\mu$m')
			legend3 = r'a = 10$\mu$m'

			c1 = 'blue'
			c2 = 'red'
			c3 = 'red'

		elif param2 == '100':
			emis1 = 10**f3
			emis2 = f3*np.nan

			legend1 = r'a = 100$\mu$m'
			legend2 = ''

			c1 = 'black'
			c2 = 'black'

		L_flt = np.asarray([10**i for i in L])
		# print(L[0:10])
		# print(L_flt[0:10])
		# print(Fx[0:10])
		# L1e = np.log10(emis1/Fx)
		# L2e = np.log10(emis2/Fx)
		# L3e = np.log10(emis3/Fx)
		L1e = np.log10(emis1/(L_flt/F1))
		L2e = np.log10(emis2/(L_flt/F1))
		L3e = np.log10(emis2/(L_flt/F1))
		L4e = np.log10(emis3/(L_flt/F1))
		bin_size1 = np.arange(-3,3,0.25)
		bin_size2 = np.arange(-3,3,0.25)


		c1 = 'blue'
		c2 = 'red'
		c3 = 'green'

		xlabel1 = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(0.25\mu \mathrm{m})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'
		xlabel2 = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(10\mu \mathrm{m})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'
		xlabel3 = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(100\mu \mathrm{m})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'


		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		cosmos_norm1 = norm1[mark1 == 0]
		s82X_norm1 = norm1[mark1 == 1]

		cosmos_norm2 = norm2[mark2 == 0]
		s82X_norm2 = norm2[mark2 == 1]

		cosmos_norm3 = norm3[mark3 == 0]
		s82X_norm3 = norm3[mark3 == 1]

		cosmos_norm4 = norm4[mark4 == 0]
		s82X_norm4 = norm4[mark4 == 1]

		cosmos_norm5 = norm5[mark5 == 0]
		s82X_norm5 = norm5[mark5 == 1]
	
		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]


		fig = plt.figure(figsize=(20,15),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=5, ncols=3, left=0.1,right=0.9,wspace=0.1,hspace=0.1)
		# gs2 = fig.add_gridspec(nrows=5, ncols=2, left=0.5,right=0.95,wspace=0.1,hspace=0.1)


		# ax1 = fig.add_subplot(gs1[0,0])
		# x1 = x[B1]
		# y1 = y[B1]
		# L1 = L[B1]
		# spec_type1 = spec_type[B1]

		# test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		# lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		# ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		# axcb1 = fig.colorbar(lc1)
		# axcb1.mappable.set_clim(clim1,clim2)
		# axcb1.remove()

		# ax1.set_xscale('log')
		# ax1.set_yscale('log')
		# ax1.set_xlim(8E-5,7E2)
		# ax1.set_ylim(5E-3,50)
		# ax1.set_xticklabels([])
		# ax1.set_xticks(xticks)
		# ax1.set_yticks(yticks)
		# ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		# ax1.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax2 = fig.add_subplot(gs1[1,0])
		# x2 = x[B2]
		# y2 = y[B2]
		# L2 = L[B2]
		# spec_type2 = spec_type[B2]


		# lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		# ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		# axcb2 = fig.colorbar(lc2)
		# axcb2.mappable.set_clim(clim1,clim2)
		# axcb2.remove()

		# ax2.set_xscale('log')
		# ax2.set_yscale('log')
		# ax2.set_xlim(8E-5,7E2)
		# ax2.set_ylim(5E-3,50)
		# ax2.set_xticklabels([])
		# ax2.set_xticks(xticks)
		# ax2.set_yticks(yticks)
		# ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax3 = fig.add_subplot(gs1[2,0])
		# x3 = x[B3]
		# y3 = y[B3]
		# L3 = L[B3]
		# spec_type3 = spec_type[B3]

		# lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		# ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		# axcb3 = fig.colorbar(lc3)
		# axcb3.mappable.set_clim(clim1,clim2)
		# axcb3.remove()

		# ax3.set_xscale('log')
		# ax3.set_yscale('log')
		# ax3.set_xlim(8E-5,7E2)
		# ax3.set_ylim(5E-3,50)
		# ax3.set_xticklabels([])
		# ax3.set_xticks(xticks)
		# ax3.set_yticks(yticks)
		# ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax3.set_ylabel(r'$\lambda$ L$_\lambda$')


		# ax4 = fig.add_subplot(gs1[3,0])
		# x4 = x[B4]
		# y4 = y[B4]
		# L4 = L[B4]
		# spec_type4 = spec_type[B4]

		# lc4 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5)
		# ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		# axcb4 = fig.colorbar(lc4)
		# axcb4.mappable.set_clim(clim1,clim2)
		# axcb4.remove()

		# ax4.set_xscale('log')
		# ax4.set_yscale('log')
		# ax4.set_xlim(8E-5,7E2)
		# ax4.set_ylim(5E-3,50)
		# ax4.set_xticklabels([])
		# ax4.set_xticks(xticks)
		# ax4.set_yticks(yticks)
		# ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		# ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax5 = fig.add_subplot(gs1[4,0])
		# x5 = x[B5]
		# y5 = y[B5]
		# L5 = L[B5]
		# spec_type5 = spec_type[B5]

		# x6 = x5
		# y6 = y5

		# lc5 = self.multilines(x6,y6,L5,cmap='rainbow',lw=1.5)
		# ax5.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0),color='k',lw=3.5)
		# axcb5 = fig.colorbar(lc5)
		# axcb5.mappable.set_clim(clim1,clim2)
		# axcb5.remove()

		# ax5.set_xscale('log')
		# ax5.set_yscale('log')
		# ax5.set_xlim(8E-5,7E2)
		# ax5.set_ylim(5E-3,50)
		# ax5.set_xticks(xticks)
		# ax5.set_yticks(yticks)

		# ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		# ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		# ax1.grid()
		# ax2.grid()
		# ax3.grid()
		# ax4.grid()
		# ax5.grid()
		
		# cbar_ax = fig.add_subplot(gs1[:,-1:])
		# # fig.tight_layout()
		# # fig.subplots_adjust(bottom=0.17)
		# # fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		# cb = fig.colorbar(test,cax=cbar_ax)
		# cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')

		yticks = [0,10,20,30]
		xticks = [-2,-1,0,1,2]

		ax1 = fig.add_subplot(gs1[0,0])
		ax1.hist(L1e[B1],bins=bin_size1,histtype='step',color=c1,alpha=1,lw=3)
		ax1.axvline(np.nanmedian(L1e[B1]),color=c1,ls='--',lw=3,alpha=1,label=legend1)
		# ax6.hist(L2e[B1],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax6.axvline(np.nanmedian(L2e[B1]),color=c2,ls='--',lw=3,alpha=1,label=legend2)
		# ax6.hist(L3e[B1],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax6.axvline(np.nanmedian(L3e[B1]),color=c3,ls='--',lw=3,label=legend3)
		ax1.set_ylim(0,35)
		ax1.set_xlim(-3,3)
		ax1.set_yticks(yticks)
		ax1.set_xticks(xticks)
		ax1.set_xticklabels([])
		ax1.grid()
		# ax6.legend(loc='upper left',fontsize=14)

		ax2 = fig.add_subplot(gs1[1,0])
		ax2.hist(L1e[B2],bins=bin_size1,histtype='step',color=c1,alpha=1,lw=3)
		ax2.axvline(np.nanmedian(L1e[B2]),color=c1,ls='--',lw=3,alpha=1)
		# ax7.hist(L2e[B2],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax7.axvline(np.nanmedian(L2e[B2]),color=c2,ls='--',lw=3,alpha=1)
		# ax7.hist(L3e[B2],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax7.axvline(np.nanmedian(L3e[B2]),color=c3,ls='--',lw=3)
		ax2.set_ylim(0,35)
		ax2.set_xlim(-3,3)
		ax2.set_yticks(yticks)
		ax2.set_xticks(xticks)
		ax2.set_xticklabels([])
		ax2.grid()

		ax3 = fig.add_subplot(gs1[2,0])
		ax3.hist(L1e[B3],bins=bin_size1,histtype='step',color=c1,alpha=1,lw=3)
		ax3.axvline(np.nanmedian(L1e[B3]),color=c1,ls='--',lw=3,alpha=1)
		# ax8.hist(L2e[B3],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax8.axvline(np.nanmedian(L2e[B3]),color=c2,ls='--',lw=3,alpha=1)
		# ax8.hist(L3e[B3],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax8.axvline(np.nanmedian(L3e[B3]),color=c3,ls='--',lw=3)
		ax3.set_ylim(0,35)
		ax3.set_xlim(-3,3)
		ax3.set_yticks(yticks)
		ax3.set_xticks(xticks)
		ax3.set_xticklabels([])
		# ax8.legend(loc='upper left')
		ax3.grid()

		ax4 = fig.add_subplot(gs1[3,0])
		ax4.hist(L1e[B4],bins=bin_size1,histtype='step',color=c1,alpha=1,lw=3)
		ax4.axvline(np.nanmedian(L1e[B4]),color=c1,ls='--',lw=3,alpha=1)
		# ax9.hist(L2e[B4],bins=bin_size2,histtype='step',color=c2,alpha=1,label=legend2,lw=3)
		# ax9.axvline(np.nanmedian(L2e[B4]),color=c2,ls='--',lw=3,alpha=1)
		# ax9.hist(L3e[B4],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax9.axvline(np.nanmedian(L3e[B4]),color=c3,ls='--',lw=3)
		ax4.set_ylim(0,35)
		ax4.set_xlim(-3,3)
		ax4.set_yticks(yticks)
		ax4.set_xticks(xticks)
		ax4.set_xticklabels([])
		# ax9.legend(loc='upper left')
		ax4.grid()

		ax5 = fig.add_subplot(gs1[4,0])
		ax5.hist(L1e[B5],bins=bin_size1,histtype='step',color=c1,alpha=1,lw=3)
		ax5.axvline(np.nanmedian(L1e[B5]),color=c1,ls='--',lw=3,alpha=1)
		# ax10.hist(L2e[B5],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax10.axvline(np.nanmedian(L2e[B5]),color=c2,ls='--',lw=3,alpha=1)
		# ax10.hist(L3e[B5],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax10.axvline(np.nanmedian(L3e[B5]),color=c3,ls='--',lw=3)
		ax5.set_xlabel(xlabel1)
		ax5.set_ylim(0,35)
		ax5.set_xlim(-3,3)
		ax5.set_yticks(yticks)
		ax5.set_xticks(xticks)
		ax5.grid()

		ax6 = fig.add_subplot(gs1[0,1])
		ax6.hist(L3e[B1],bins=bin_size1,histtype='step',color=c2,alpha=1,lw=3)
		ax6.axvline(np.nanmedian(L3e[B1]),color=c2,ls='--',lw=3,alpha=1,label=legend1)
		# ax6.hist(L2e[B1],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax6.axvline(np.nanmedian(L2e[B1]),color=c2,ls='--',lw=3,alpha=1,label=legend2)
		# ax6.hist(L3e[B1],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax6.axvline(np.nanmedian(L3e[B1]),color=c3,ls='--',lw=3,label=legend3)
		ax6.set_ylim(0,35)
		ax6.set_xlim(-3,3)
		ax6.set_yticklabels([])
		ax6.set_yticks(yticks)
		ax6.set_xticks(xticks)
		ax6.set_xticklabels([])
		ax6.grid()
		# ax6.legend(loc='upper left',fontsize=14)

		ax7 = fig.add_subplot(gs1[1,1])
		ax7.hist(L3e[B2],bins=bin_size1,histtype='step',color=c2,alpha=1,lw=3)
		ax7.axvline(np.nanmedian(L3e[B2]),color=c2,ls='--',lw=3,alpha=1)
		# ax7.hist(L2e[B2],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax7.axvline(np.nanmedian(L2e[B2]),color=c2,ls='--',lw=3,alpha=1)
		# ax7.hist(L3e[B2],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax7.axvline(np.nanmedian(L3e[B2]),color=c3,ls='--',lw=3)
		ax7.set_ylim(0,35)
		ax7.set_xlim(-3,3)
		ax7.set_yticklabels([])
		ax7.set_yticks(yticks)
		ax7.set_xticks(xticks)
		ax7.set_xticklabels([])
		ax7.grid()

		ax8 = fig.add_subplot(gs1[2,1])
		ax8.hist(L3e[B3],bins=bin_size1,histtype='step',color=c2,alpha=1,lw=3)
		ax8.axvline(np.nanmedian(L3e[B3]),color=c2,ls='--',lw=3,alpha=1)
		# ax8.hist(L2e[B3],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax8.axvline(np.nanmedian(L2e[B3]),color=c2,ls='--',lw=3,alpha=1)
		# ax8.hist(L3e[B3],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax8.axvline(np.nanmedian(L3e[B3]),color=c3,ls='--',lw=3)
		ax8.set_ylim(0,35)
		ax8.set_xlim(-3,3)
		ax8.set_yticklabels([])
		ax8.set_yticks(yticks)
		ax8.set_xticks(xticks)
		ax8.set_xticklabels([])
		# ax8.legend(loc='upper left')
		ax8.grid()

		ax9 = fig.add_subplot(gs1[3,1])
		ax9.hist(L3e[B4],bins=bin_size1,histtype='step',color=c2,alpha=1,lw=3)
		ax9.axvline(np.nanmedian(L3e[B4]),color=c2,ls='--',lw=3,alpha=1)
		# ax9.hist(L2e[B4],bins=bin_size2,histtype='step',color=c2,alpha=1,label=legend2,lw=3)
		# ax9.axvline(np.nanmedian(L2e[B4]),color=c2,ls='--',lw=3,alpha=1)
		# ax9.hist(L3e[B4],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax9.axvline(np.nanmedian(L3e[B4]),color=c3,ls='--',lw=3)
		ax9.set_ylim(0,35)
		ax9.set_xlim(-3,3)
		ax9.set_yticklabels([])
		ax9.set_yticks(yticks)
		ax9.set_xticks(xticks)
		ax9.set_xticklabels([])
		# ax9.legend(loc='upper left')
		ax9.grid()

		ax10 = fig.add_subplot(gs1[4,1])
		ax10.hist(L3e[B5],bins=bin_size1,histtype='step',color=c2,alpha=1,lw=3)
		ax10.axvline(np.nanmedian(L3e[B5]),color=c2,ls='--',lw=3,alpha=1)
		# ax10.hist(L2e[B5],bins=bin_size2,histtype='step',color=c2,alpha=1,lw=3)
		# ax10.axvline(np.nanmedian(L2e[B5]),color=c2,ls='--',lw=3,alpha=1)
		# ax10.hist(L3e[B5],bins=bin_size2,histtype='step',color=c3,alpha=1,lw=3)
		# ax10.axvline(np.nanmedian(L3e[B5]),color=c3,ls='--',lw=3)
		ax10.set_xlabel(xlabel2)
		ax10.set_yticklabels([])
		ax10.set_ylim(0,35)
		ax10.set_xlim(-3,3)
		ax10.set_yticks(yticks)
		ax10.set_xticks(xticks)
		ax10.grid()
		# ax10.legend(loc='upper left')

		ax11 = fig.add_subplot(gs1[0,2])
		ax11.hist(L4e[B1],bins=bin_size1,histtype='step',color=c3,alpha=0.85,label=legend3,lw=3)
		ax11.axvline(np.nanmedian(L4e[B1]),color=c3,ls='--',lw=3)
		ax11.set_ylim(0,35)
		ax11.set_xlim(-3,3)
		ax11.set_yticks(yticks)
		ax11.set_xticks(xticks)
		ax11.set_yticklabels([])
		ax11.set_xticklabels([])
		ax11.grid()

		ax12 = fig.add_subplot(gs1[1,2])
		ax12.hist(L4e[B2],bins=bin_size1,histtype='step',color=c3,alpha=0.85,label=legend3,lw=3)
		ax12.axvline(np.nanmedian(L4e[B2]),color=c3,ls='--',lw=3)
		ax12.set_ylim(0,35)
		ax12.set_xlim(-3,3)
		ax12.set_yticks(yticks)
		ax12.set_xticks(xticks)
		ax12.set_yticklabels([])
		ax12.set_xticklabels([])
		ax12.grid()

		ax13 = fig.add_subplot(gs1[2,2])
		ax13.hist(L4e[B3],bins=bin_size1,histtype='step',color=c3,alpha=0.85,label=legend3,lw=3)
		ax13.axvline(np.nanmedian(L4e[B3]),color=c3,ls='--',lw=3)
		ax13.set_ylim(0,35)
		ax13.set_xlim(-3,3)
		ax13.set_yticks(yticks)
		ax13.set_xticks(xticks)
		ax13.set_yticklabels([])
		ax13.set_xticklabels([])
		ax13.grid()

		ax14 = fig.add_subplot(gs1[3,2])
		ax14.hist(L4e[B4],bins=bin_size1,histtype='step',color=c3,alpha=0.85,label=legend3,lw=3)
		ax14.axvline(np.nanmedian(L4e[B4]),color=c3,ls='--',lw=3) 
		ax14.set_ylim(0,35)
		ax14.set_xlim(-3,3)
		ax14.set_yticks(yticks)
		ax14.set_xticks(xticks)
		ax14.set_yticklabels([])
		ax14.set_xticklabels([])
		ax14.grid()

		ax15 = fig.add_subplot(gs1[4,2])
		ax15.hist(L4e[B5],bins=bin_size1,histtype='step',color=c3,alpha=0.85,label=legend3,lw=3)
		ax15.axvline(np.nanmedian(L4e[B5]),color=c3,ls='--',lw=3)
		ax15.set_ylim(0,35)
		ax15.set_xlim(-3,3)
		# ax15.legend()
		ax15.set_yticks(yticks)
		ax15.set_xticks(xticks)
		ax15.set_yticklabels([])
		ax15.set_xlabel(xlabel3)
		ax15.grid()

		plt.savefig('/Users/connor_auge/Desktop/final_paper1/UV_MIR_hists.pdf')
		plt.show()


	def plot_5panel_zbins(self,savestring,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,suptitle=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None,wfir=None,ffir=None):

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-3] = np.nan
		y[y < 1E-3] = np.nan

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		z_med = np.nanmedian(spec_z)

		clim1 = 43
		clim2 = 45.5

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq

			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]

			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		# B1_check = (uv_slope < -0.3) & (mir_slope1 >= -0.2)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		alpha = 0.7

		
		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]


		plt.rcParams['font.size']=24
		plt.rcParams['axes.linewidth']=2.5
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-3,1E-2,0.1,1,10,100]
		z = spec_z

	
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		z1 = spec_z[B1]
		cosmos_s82x_list_1 = cosmos_s82x_list[B1]
		median_upper_wave1 = cosmos_s82x_wave[B1]
		cosmos_s82x_list_12 = cosmos_s82x_list2[B1]
		median_upper_wave12 = cosmos_s82x_wave2[B1]
		median_wave1 = 10**median_wavelength[B1]
		median_flux1 = 10**median_flux[B1]
		# wfir1 = np.asarray([wfir[B1][i]/norm1[i] for i in range(len(wfir[B1]))])
		wfir1 = wfir[B1]
		ffir1 = np.asarray([ffir[B1][i]/norm1[i] for i in range(len(ffir[B1]))])

		# print(uv_slope[(z >= zlim_2) & (z <= zlim_3)])
		# print(mir_slope1[(z >= zlim_2) & (z <= zlim_3)])
		
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		z2 = spec_z[B2]
		cosmos_s82x_list_2 = cosmos_s82x_list[B2]
		median_upper_wave2 = cosmos_s82x_wave[B2]
		cosmos_s82x_list_22 = cosmos_s82x_list2[B2]
		median_upper_wave22 = cosmos_s82x_wave2[B2]
		median_wave2 = 10**median_wavelength[B2]
		median_flux2 = 10**median_flux[B2]
		# wfir2 = np.asarray([wfir[B2][i]/norm2[i] for i in range(len(wfir[B2]))])
		wfir2 = wfir[B2]
		ffir2 = np.asarray([ffir[B2][i]/norm2[i] for i in range(len(ffir[B2]))])
	
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		z3 = spec_z[B3]
		cosmos_s82x_list_3 = cosmos_s82x_list[B3]
		median_upper_wave3 = cosmos_s82x_wave[B3]
		cosmos_s82x_list_32 = cosmos_s82x_list2[B3]
		median_upper_wave32 = cosmos_s82x_wave2[B3]
		median_wave3 = 10**median_wavelength[B3]
		median_flux3 = 10**median_flux[B3]
		# wfir3 = np.asarray([wfir[B3][i]/norm3[i] for i in range(len(wfir[B3]))])
		wfir3 = wfir[B3]
		ffir3 = np.asarray([ffir[B3][i]/norm3[i] for i in range(len(ffir[B3]))])
		
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		z4 = spec_z[B4]
		cosmos_s82x_list_4 = cosmos_s82x_list[B4]
		median_upper_wave4 = cosmos_s82x_wave[B4]
		cosmos_s82x_list_42 = cosmos_s82x_list2[B4]
		median_upper_wave42 = cosmos_s82x_wave2[B4]
		median_wave4 = 10**median_wavelength[B4]
		median_flux4 = 10**median_flux[B4]
		# wfir4 = np.asarray([wfir[B4][i]/norm4[i] for i in range(len(wfir[B4]))])
		wfir4 = wfir[B4]
		ffir4 = np.asarray([ffir[B4][i]/norm4[i] for i in range(len(ffir[B4]))])

		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		z5 = spec_z[B5]
		cosmos_s82x_list_5 = cosmos_s82x_list[B5]
		median_upper_wave5 = cosmos_s82x_wave[B5]
		cosmos_s82x_list_52 = cosmos_s82x_list2[B5]
		median_upper_wave52 = cosmos_s82x_wave2[B5]
		median_wave5 = 10**median_wavelength[B5]
		median_flux5 = 10**median_flux[B5]
		# wfir5 = np.asarray([wfir[B5][i]/norm5[i] for i in range(len(wfir[B5]))])
		wfir5 = wfir[B5]
		ffir5 = np.asarray([ffir[B5][i]/norm5[i] for i in range(len(ffir[B5]))])

		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']
		xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]

		


		fig = plt.figure(figsize=(21,27))
		gs = fig.add_gridspec(nrows=5, ncols=4,width_ratios=[3.25,3.25,3.25,0.2])
		gs.update(left=0.1, right=0.9, top=0.93, bottom=0.08)
		gs.update(hspace=0.07,wspace=-0.22) # set the spacing between axes
		


		ax1 = plt.subplot(gs[0,0])

		# upper_seg1 = np.stack((median_upper_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)], cosmos_s82x_list_1[(z1 >= zlim_1) & (z1 <= zlim_2)]), axis=2)
		upper_seg1 = np.stack((wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)]), axis=2)
		upper_all1 = LineCollection(upper_seg1,color='gray',alpha=0.3)
		ax1.add_collection(upper_all1)

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc1 = self.multilines(x1[(z1 >= zlim_1) & (z1 <= zlim_2)], y1[(z1 >= zlim_1) & (z1 <= zlim_2)], L1[(z1 >= zlim_1) & (z1 <= zlim_2)], cmap='rainbow_r', lw=1.5, alpha = alpha, rasterized=True)
		ax1.plot(np.nanmedian(median_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmedian(median_flux1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		# ax1.plot(np.nanmedian(median_upper_wave12[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax1.plot(np.nanmedian(median_upper_wave12[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmedian(cosmos_s82x_list_12[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), color='k', lw=2.0)
		# ax1.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		ax1.plot(np.nanmedian(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0)[-3:], np.nanmedian(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax1.plot(np.append(np.nanmedian(median_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0)[-1],np.nanmedian(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0)[:3]), np.append(np.nanmedian(median_flux1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0)[-1],np.nanmedian(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax1.plot(np.nanmean(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmean(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		axcb1.remove()
	

		ax1.set_aspect(1)
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(1E-4,120)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		# ax1.set_xticklabels(xticks_labels)
		ax1.text(0.05,0.7,f'n = {len(x1[(z1 >= zlim_1) & (z1 <= zlim_2)])}',transform=ax1.transAxes)
		ax1.text(0.75,0.08,str((len(x1[(z1 >= zlim_1) & (z1 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		# ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax1.text(0.0,1.03,r'A',transform=ax1.transAxes,fontsize=27,weight='bold')
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		ax1.text(-0.45,0.5,'1',transform=ax1.transAxes,fontsize=38,weight='bold')

		ax2 = plt.subplot(gs[1,0])

		# upper_seg2 = np.stack((median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)], cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)]), axis=2)
		upper_seg2 = np.stack((wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)]), axis=2)
		upper_all2 = LineCollection(upper_seg2,color='gray',alpha=0.3)
		ax2.add_collection(upper_all2)

		lc2 = self.multilines(x2[(z2 >= zlim_1) & (z2 <= zlim_2)],y2[(z2 >= zlim_1) & (z2 <= zlim_2)],L2[(z2 >= zlim_1) & (z2 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(median_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(median_flux2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), color='k', lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		# ax2.plot(np.nanmedian(median_upper_wave22[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax2.plot(np.nanmedian(median_upper_wave22[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(cosmos_s82x_list_22[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), color='k', lw=2.0)
		# ax2.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		ax2.plot(np.nanmedian(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0)[-3:], np.nanmedian(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax2.plot(np.append(np.nanmedian(median_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0)[-1],np.nanmedian(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0)[:3]), np.append(np.nanmedian(median_flux2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0)[-1],np.nanmedian(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax2.plot(np.nanmean(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmean(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)		
		# axcb2.remove()

		ax2.set_aspect(1)
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(1E-4,120)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		# ax2.set_xticklabels(xticks_labels)
		ax2.text(0.05,0.7,f'n = {len(x2[(z2 >= zlim_1) & (z2 <= zlim_2)])}',transform=ax2.transAxes)
		ax2.text(0.75, 0.08, str((len(x2[(z2 >= zlim_1) & (z2 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%', transform=ax2.transAxes, weight='bold')
		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax2.text(-0.45,0.5,'2',transform=ax2.transAxes,fontsize=38,weight='bold')

		ax3 = plt.subplot(gs[2,0])

		# upper_seg3 = np.stack((median_upper_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)], cosmos_s82x_list_3[(z3 >= zlim_1) & (z3 <= zlim_2)]), axis=2)
		upper_seg3 = np.stack((wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)]), axis=2)
		upper_all3 = LineCollection(upper_seg3,color='gray',alpha=0.3)
		ax3.add_collection(upper_all3)

		lc3 = self.multilines(x3[(z3 >= zlim_1) & (z3 <= zlim_2)],y3[(z3 >= zlim_1) & (z3 <= zlim_2)],L3[(z3 >= zlim_1) & (z3 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(median_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(median_flux3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		# ax3.plot(np.nanmedian(median_upper_wave32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax3.plot(np.nanmedian(median_upper_wave32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),color='k',lw=2.0)
		# ax3.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		ax3.plot(np.nanmedian(wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0)[-3:], np.nanmedian(ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax3.plot(np.append(np.nanmedian(median_wave3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0)[-1],np.nanmedian(wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0)[:3]), np.append(np.nanmedian(median_flux3[(z3 >= zlim_1) & (z3 <= zlim_2)],axis=0)[-1],np.nanmedian(ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax3.plot(np.nanmean(wfir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0), np.nanmean(ffir3[(z3 >= zlim_1) & (z3 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		axcb3.remove()

		ax3.set_aspect(1)
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(1E-4,120)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_xticklabels(xticks_labels)
		ax3.text(0.05,0.7,f'n = {len(x3[(z3 >= zlim_1) & (z3 <= zlim_2)])}',transform=ax3.transAxes)
		ax3.text(0.75,0.08,str((len(x3[(z3 >= zlim_1) & (z3 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		ax3.set_ylabel(r'Normalized $\lambda$ L$_\lambda$',fontsize=40)
		ax3.text(-0.45,0.5,'3',transform=ax3.transAxes,fontsize=38,weight='bold')

		ax4 = plt.subplot(gs[3,0])

		# upper_seg4 = np.stack((median_upper_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)], cosmos_s82x_list_4[(z4 >= zlim_1) & (z4 <= zlim_2)]), axis=2)
		upper_seg4 = np.stack((wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)]), axis=2)
		upper_all4 = LineCollection(upper_seg4,color='gray',alpha=0.3)
		ax4.add_collection(upper_all4)

		lc4 = self.multilines(x4[(z4 >= zlim_1) & (z4 <= zlim_2)],y4[(z4 >= zlim_1) & (z4 <= zlim_2)],L4[(z4 >= zlim_1) & (z4 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax4.plot(np.nanmedian(median_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(median_flux4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		# ax4.plot(np.nanmedian(median_upper_wave42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax4.plot(np.nanmedian(median_upper_wave42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),color='k',lw=2.0)
		# ax4.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		ax4.plot(np.nanmedian(wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0)[-3:], np.nanmedian(ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax4.plot(np.append(np.nanmedian(median_wave4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0)[-1],np.nanmedian(wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0)[:3]), np.append(np.nanmedian(median_flux4[(z4 >= zlim_1) & (z4 <= zlim_2)],axis=0)[-1],np.nanmedian(ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax4.plot(np.nanmean(wfir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0), np.nanmean(ffir4[(z4 >= zlim_1) & (z4 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# axcb4.remove()

		ax4.set_aspect(1)
		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(1E-4,120)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		# ax4.set_xticklabels(xticks_labels)
		ax4.text(0.05,0.7,f'n = {len(x4[(z4 >= zlim_1) & (z4 <= zlim_2)])}',transform=ax4.transAxes)
		ax4.text(0.75,0.08,str((len(x4[(z4 >= zlim_1) & (z4 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		# ax4.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax4.text(-0.45,0.5,'4',transform=ax4.transAxes,fontsize=38,weight='bold')

		ax5 = plt.subplot(gs[4,0])

		# upper_seg5 = np.stack((median_upper_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)], cosmos_s82x_list_5[(z5 >= zlim_1) & (z5 <= zlim_2)]), axis=2)
		upper_seg5 = np.stack((wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)]), axis=2)
		upper_all5 = LineCollection(upper_seg5,color='gray',alpha=0.3)
		ax5.add_collection(upper_all5)

		lc5 = self.multilines(x5[(z5 >= zlim_1) & (z5 <= zlim_2)],y5[(z5 >= zlim_1) & (z5 <= zlim_2)],L5[(z5 >= zlim_1) & (z5 <= zlim_2)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax5.plot(np.nanmedian(median_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(median_flux5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),color='k',lw=3.5)
		axcb5 = fig.colorbar(lc5)
		axcb5.mappable.set_clim(clim1,clim2)
		# ax5.plot(np.nanmedian(median_upper_wave52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),'v',ms=5,color='k')
		# ax5.plot(np.nanmedian(median_upper_wave52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),color='k',lw=2.0)
		# ax5.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		ax5.plot(np.nanmedian(wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0)[-3:], np.nanmedian(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax5.plot(np.append(np.nanmedian(median_wave5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0)[-1],np.nanmedian(wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0)[:3]), np.append(np.nanmedian(median_flux5[(z5 >= zlim_1) & (z5 <= zlim_2)],axis=0)[-1],np.nanmedian(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax5.plot(np.nanmean(wfir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0), np.nanmean(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		# axcb5.remove()

		print('HERE HERE: ', np.nanmedian(ffir5[(z5 >= zlim_1) & (z5 <= zlim_2)], axis=0)[-3:])

		ax5.set_aspect(1)
		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(1E-4,120)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)
		ax5.set_xticklabels(xticks_labels)
		ax5.text(0.05,0.7,f'n = {len(x5[(z5 >= zlim_1) & (z5 <= zlim_2)])}',transform=ax5.transAxes)
		ax5.text(0.75,0.08,str((len(x5[(z5 >= zlim_1) & (z5 <= zlim_2)])/len(x[(z >= zlim_1) & (z <= zlim_2)]))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		# ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax5.text(-0.45,0.5,'5',transform=ax5.transAxes,fontsize=38,weight='bold')




		ax6 = plt.subplot(gs[0,1])

		# upper_seg6 = np.stack((median_upper_wave1[(z1 > zlim_2) & (z1 <= zlim_3)], cosmos_s82x_list_1[(z1 > zlim_2) & (z1 <= zlim_3)]), axis=2)
		upper_seg6 = np.stack((wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], ffir1[(z1 > zlim_2) & (z1 <= zlim_3)]), axis=2)
		upper_all6 = LineCollection(upper_seg6,color='gray',alpha=0.3)
		ax6.add_collection(upper_all6)

		lc6 = self.multilines(x1[(z1 > zlim_2) & (z1 <= zlim_3)],y1[(z1 > zlim_2) & (z1 <= zlim_3)],L1[(z1 > zlim_2) & (z1 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax6.plot(np.nanmedian(median_wave1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(median_flux1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),color='k',lw=3.5)
		axcb6 = fig.colorbar(lc6)
		axcb6.mappable.set_clim(clim1,clim2)
		# ax6.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax6.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0),color='k',lw=2.0)
		# ax6.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		ax6.plot(np.nanmedian(wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0)[-3:], np.nanmedian(ffir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax6.plot(np.append(np.nanmedian(median_wave1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0)[-1],np.nanmedian(wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0)[:3]), np.append(np.nanmedian(median_flux1[(z1 > zlim_2) & (z1 <= zlim_3)],axis=0)[-1],np.nanmedian(ffir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax6.plot(np.nanmean(wfir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0), np.nanmean(ffir1[(z1 > zlim_2) & (z1 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# axcb6.remove()

		ax6.set_aspect(1)
		ax6.set_xscale('log')
		ax6.set_yscale('log')
		ax6.set_xlim(8E-5,7E2)
		ax6.set_ylim(1E-4,120)
		ax6.set_xticklabels([])
		ax6.set_yticklabels([])
		ax6.set_xticks(xticks)
		ax6.set_yticks(yticks)
		# ax6.set_xticklabels(xticks_labels)
		ax6.text(0.05,0.7,f'n = {len(x1[(z1 > zlim_2) & (z1 <= zlim_3)])}',transform=ax6.transAxes)
		ax6.text(0.75,0.08,str((len(x1[(z1 > zlim_2) & (z1 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax6.transAxes,weight='bold')
		ax6.text(0.0,1.03,r'B',transform=ax6.transAxes,fontsize=27,weight='bold')
		ax6.set_title(str(zlim_2)+' < z < '+str(zlim_3))

		ax7 = plt.subplot(gs[1,1])

		# upper_seg7 = np.stack((median_upper_wave2[(z2 > zlim_2) & (z2 <= zlim_3)], cosmos_s82x_list_2[(z2 > zlim_2) & (z2 <= zlim_3)]), axis=2)
		upper_seg7 = np.stack((wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], ffir2[(z2 > zlim_2) & (z2 <= zlim_3)]), axis=2)
		upper_all7 = LineCollection(upper_seg7,color='gray',alpha=0.3)
		ax7.add_collection(upper_all7)

		lc7 = self.multilines(x2[(z2 > zlim_2) & (z2 <= zlim_3)],y2[(z2 > zlim_2) & (z2 <= zlim_3)],L2[(z2 > zlim_2) & (z2 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax7.plot(np.nanmedian(median_wave2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), np.nanmedian(median_flux2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), color='k', lw=3.5)		
		axcb7 = fig.colorbar(lc7)
		axcb7.mappable.set_clim(clim1,clim2)
		# ax7.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax7.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0),color='k',lw=2.0)
		# ax7.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		ax7.plot(np.nanmedian(wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0)[-3:], np.nanmedian(ffir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax7.plot(np.append(np.nanmedian(median_wave2[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0)[-1],np.nanmedian(wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0)[:3]), np.append(np.nanmedian(median_flux2[(z2 > zlim_2) & (z2 <= zlim_3)],axis=0)[-1],np.nanmedian(ffir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax7.plot(np.nanmean(wfir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0), np.nanmean(ffir2[(z2 > zlim_2) & (z2 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# axcb7.remove()

		ax7.set_aspect(1)
		ax7.set_xscale('log')
		ax7.set_yscale('log')
		ax7.set_xlim(8E-5,7E2)
		ax7.set_ylim(1E-4,120)
		ax7.set_xticklabels([])
		ax7.set_yticklabels([])
		ax7.set_xticks(xticks)
		ax7.set_yticks(yticks)
		# ax7.set_xticklabels(xticks_labels)
		ax7.text(0.05,0.7,f'n = {len(x2[(z2 > zlim_2) & (z2 <= zlim_3)])}',transform=ax7.transAxes)
		ax7.text(0.75,0.08,str((len(x2[(z2 > zlim_2) & (z2 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax7.transAxes,weight='bold')

		ax8 = plt.subplot(gs[2,1])

		# upper_seg8 = np.stack((median_upper_wave3[(z3 > zlim_2) & (z3 <= zlim_3)], cosmos_s82x_list_3[(z3 > zlim_2) & (z3 <= zlim_3)]), axis=2)
		upper_seg8 = np.stack((wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], ffir3[(z3 > zlim_2) & (z3 <= zlim_3)]), axis=2)
		upper_all8 = LineCollection(upper_seg8,color='gray',alpha=0.3)
		ax8.add_collection(upper_all8)

		lc8 = self.multilines(x3[(z3 > zlim_2) & (z3 <= zlim_3)],y3[(z3 > zlim_2) & (z3 <= zlim_3)],L3[(z3 > zlim_2) & (z3 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax8.plot(np.nanmedian(median_wave3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(median_flux3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),color='k',lw=3.5)
		axcb8 = fig.colorbar(lc8)
		axcb8.mappable.set_clim(clim1,clim2)
		# ax8.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax8.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0),color='k',lw=2.0)
		# ax8.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		ax8.plot(np.nanmedian(wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0)[-3:], np.nanmedian(ffir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax8.plot(np.append(np.nanmedian(median_wave3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0)[-1],np.nanmedian(wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0)[:3]), np.append(np.nanmedian(median_flux3[(z3 > zlim_2) & (z3 <= zlim_3)],axis=0)[-1],np.nanmedian(ffir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax8.plot(np.nanmean(wfir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0), np.nanmean(ffir3[(z3 > zlim_2) & (z3 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		axcb8.remove()

		ax8.set_aspect(1)
		ax8.set_xscale('log')
		ax8.set_yscale('log')
		ax8.set_xlim(8E-5,7E2)
		ax8.set_ylim(1E-4,120)
		ax8.set_xticklabels([])
		ax8.set_yticklabels([])
		ax8.set_xticks(xticks)
		ax8.set_yticks(yticks)
		# ax8.set_xticklabels(xticks_labels)
		ax8.text(0.05,0.7,f'n = {len(x3[(z3 > zlim_2) & (z3 <= zlim_3)])}',transform=ax8.transAxes)
		ax8.text(0.75,0.08,str((len(x3[(z3 > zlim_2) & (z3 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax8.transAxes,weight='bold')

		ax9 = plt.subplot(gs[3,1])

		# upper_seg9 = np.stack((median_upper_wave4[(z4 > zlim_2) & (z4 <= zlim_3)], cosmos_s82x_list_4[(z4 > zlim_2) & (z4 <= zlim_3)]), axis=2)
		upper_seg9 = np.stack((wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], ffir4[(z4 > zlim_2) & (z4 <= zlim_3)]), axis=2)
		upper_all9 = LineCollection(upper_seg9,color='gray',alpha=0.3)
		ax9.add_collection(upper_all9)

		lc9 = self.multilines(x4[(z4 > zlim_2) & (z4 <= zlim_3)],y4[(z4 > zlim_2) & (z4 <= 0.8)],L4[(z4 > zlim_2) & (z4 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax9.plot(np.nanmedian(median_wave4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(median_flux4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),color='k',lw=3.5)
		axcb9 = fig.colorbar(lc9)
		axcb9.mappable.set_clim(clim1,clim2)
		# ax9.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax9.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0),color='k',lw=2.0)
		# ax9.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		ax9.plot(np.nanmedian(wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0)[-3:], np.nanmedian(ffir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax9.plot(np.append(np.nanmedian(median_wave4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0)[-1],np.nanmedian(wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0)[:3]), np.append(np.nanmedian(median_flux4[(z4 > zlim_2) & (z4 <= zlim_3)],axis=0)[-1],np.nanmedian(ffir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax9.plot(np.nanmean(wfir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0), np.nanmean(ffir4[(z4 > zlim_2) & (z4 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)
		# axcb9.remove()

		ax9.set_aspect(1)
		ax9.set_xscale('log')
		ax9.set_yscale('log')
		ax9.set_xlim(8E-5,7E2)
		ax9.set_ylim(1E-4,120)
		ax9.set_xticklabels([])
		ax9.set_yticklabels([])
		ax9.set_xticks(xticks)
		ax9.set_yticks(yticks)
		# ax9.set_xticklabels(xticks_labels)
		ax9.text(0.05,0.7,f'n = {len(x4[(z4 > zlim_2) & (z4 <= zlim_3)])}',transform=ax9.transAxes)
		ax9.text(0.75,0.08,str((len(x4[(z4 > zlim_2) & (z4 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax9.transAxes,weight='bold')

		ax10 = plt.subplot(gs[4,1])

		# upper_seg10 = np.stack((median_upper_wave5[(z5 > zlim_2) & (z5 <= zlim_3)], cosmos_s82x_list_5[(z5 > zlim_2) & (z5 <= zlim_3)]), axis=2)
		upper_seg10 = np.stack((wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], ffir5[(z5 > zlim_2) & (z5 <= zlim_3)]), axis=2)
		upper_all10 = LineCollection(upper_seg10,color='gray',alpha=0.3)
		ax10.add_collection(upper_all10)

		lc10 = self.multilines(x5[(z5 > zlim_2) & (z5 <= zlim_3)],y5[(z5 > zlim_2) & (z5 <= 0.8)],L5[(z5 > zlim_2) & (z5 <= zlim_3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax10.plot(np.nanmedian(median_wave5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(median_flux5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),color='k',lw=3.5)
		axcb10 = fig.colorbar(lc10)	
		axcb10.mappable.set_clim(clim1,clim2)
		# ax10.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),'v',ms=5,color='k')
		# ax10.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0),color='k',lw=2.0)
		# ax10.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		ax10.plot(np.nanmedian(wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0)[-3:], np.nanmedian(ffir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax10.plot(np.append(np.nanmedian(median_wave5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0)[-1],np.nanmedian(wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0)[:3]), np.append(np.nanmedian(median_flux5[(z5 > zlim_2) & (z5 <= zlim_3)],axis=0)[-1],np.nanmedian(ffir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax10.plot(np.nanmean(wfir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0), np.nanmean(ffir5[(z5 > zlim_2) & (z5 <= zlim_3)], axis=0),'-^',color='red',lw=1.75)		
		axcb10.remove()

		ax10.set_aspect(1)
		ax10.set_xscale('log')
		ax10.set_yscale('log')
		ax10.set_xlim(8E-5,7E2)
		ax10.set_ylim(1E-4,120)
		ax10.set_yticklabels([])
		ax10.set_xticks(xticks)
		ax10.set_yticks(yticks)
		ax10.set_xticklabels(xticks_labels)
		ax10.text(0.05,0.7,f'n = {len(x5[(z5 > zlim_2) & (z5 <= zlim_3)])}',transform=ax10.transAxes)
		ax10.text(0.75,0.08,str((len(x5[(z5 > zlim_2) & (z5 <= zlim_3)])/len(x[(z > zlim_2) & (z <= zlim_3)]))*100)[0:4]+'%',transform=ax10.transAxes,weight='bold')
		ax10.set_xlabel(r'Rest Wavelength [$\mu$m]', fontsize=40)




		ax11 = plt.subplot(gs[0,2])

		# upper_seg11 = np.stack((median_upper_wave1[(z1 > zlim_3) & (z1 <= zlim_4)], cosmos_s82x_list_1[(z1 > zlim_3) & (z1 <= zlim_4)]), axis=2)
		upper_seg11 = np.stack((wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], ffir1[(z1 > zlim_3) & (z1 <= zlim_4)]), axis=2)
		upper_all11 = LineCollection(upper_seg11,color='gray',alpha=0.3)
		ax11.add_collection(upper_all11)
		
		test = ax11.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc11 = self.multilines(x1[(z1 > zlim_3) & (z1 <= zlim_4)],y1[(z1 > zlim_3) & (z1 <= zlim_4)],L1[(z1 > zlim_3) & (z1 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax11.plot(np.nanmedian(median_wave1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(median_flux1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),color='k',lw=3.5)
		axcb11 = fig.colorbar(lc11)
		axcb11.mappable.set_clim(clim1,clim2)
		# ax11.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax11.plot(np.nanmedian(median_upper_wave12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_12[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0),color='k',lw=2.0)
		# ax11.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		ax11.plot(np.nanmedian(wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0)[-3:], np.nanmedian(ffir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax11.plot(np.append(np.nanmedian(median_wave1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0)[-1],np.nanmedian(wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0)[:3]), np.append(np.nanmedian(median_flux1[(z1 > zlim_3) & (z1 <= zlim_4)],axis=0)[-1],np.nanmedian(ffir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax11.plot(np.nanmean(wfir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0), np.nanmean(ffir1[(z1 > zlim_3) & (z1 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		axcb11.remove()

		ax11.set_aspect(1)
		ax11.set_xscale('log')
		ax11.set_yscale('log')
		ax11.set_xlim(8E-5,7E2)
		ax11.set_ylim(1E-4,120)
		ax11.set_xticklabels([])
		ax11.set_yticklabels([])
		ax11.set_xticks(xticks)
		ax11.set_yticks(yticks)
		# ax11.set_xticklabels(xticks_labels)
		ax11.text(0.05,0.7,f'n = {len(x1[(z1 > zlim_3) & (z1 <= zlim_4)])}',transform=ax11.transAxes)
		ax11.text(0.75,0.08,str((len(x1[(z1 > zlim_3) & (z1 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax11.transAxes,weight='bold')
		ax11.text(0.0,1.03,r'C',transform=ax11.transAxes,fontsize=27,weight='bold')
		ax11.set_title(str(zlim_3)+' < z < '+str(zlim_4))

		ax12 = plt.subplot(gs[1,2])

		# upper_seg12 = np.stack((median_upper_wave2[(z2 > zlim_3) & (z2 <= zlim_4)], cosmos_s82x_list_2[(z2 > zlim_3) & (z2 <= zlim_4)]), axis=2)
		upper_seg12 = np.stack((wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], ffir2[(z2 > zlim_3) & (z2 <= zlim_4)]), axis=2)
		upper_all12 = LineCollection(upper_seg12,color='gray',alpha=0.3)
		ax12.add_collection(upper_all12)

		test = ax12.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc12 = self.multilines(x2[(z2 > zlim_3) & (z2 <= zlim_4)],y2[(z2 > zlim_3) & (z2 <= zlim_4)],L2[(z2 > zlim_3) & (z2 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax12.plot(np.nanmedian(median_wave2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(median_flux2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),color='k',lw=3.5)		
		axcb12 = fig.colorbar(lc12)
		axcb12.mappable.set_clim(clim1,clim2)
		# ax12.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax12.plot(np.nanmedian(median_upper_wave22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_22[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0),color='k',lw=2.0)
		# ax12.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		ax12.plot(np.nanmedian(wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0)[-3:], np.nanmedian(ffir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax12.plot(np.append(np.nanmedian(median_wave2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0)[-1],np.nanmedian(wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0)[:3]), np.append(np.nanmedian(median_flux2[(z2 > zlim_3) & (z2 <= zlim_4)],axis=0)[-1],np.nanmedian(ffir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax12.plot(np.nanmean(wfir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0), np.nanmean(ffir2[(z2 > zlim_3) & (z2 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		axcb12.remove()

		ax12.set_aspect(1)
		ax12.set_xscale('log')
		ax12.set_yscale('log')
		ax12.set_xlim(8E-5,7E2)
		ax12.set_ylim(1E-4,120)
		ax12.set_xticklabels([])
		ax12.set_yticklabels([])
		ax12.set_xticks(xticks)
		ax12.set_yticks(yticks)
		# ax12.set_xticklabels(xticks_labels)
		ax12.text(0.05,0.7,f'n = {len(x2[(z2 > zlim_3) & (z2 <= zlim_4)])}',transform=ax12.transAxes)
		ax12.text(0.75,0.08,str((len(x2[(z2 > zlim_3) & (z2 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax12.transAxes,weight='bold')

		ax13 = plt.subplot(gs[2,2])

		# upper_seg13 = np.stack((median_upper_wave3[(z3 > zlim_3) & (z3 <= zlim_4)], cosmos_s82x_list_3[(z3 > zlim_3) & (z3 <= zlim_4)]), axis=2)
		upper_seg13 = np.stack((wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], ffir3[(z3 > zlim_3) & (z3 <= zlim_4)]), axis=2)
		upper_all13 = LineCollection(upper_seg13,color='gray',alpha=0.3)
		ax13.add_collection(upper_all13)

		test = ax13.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc13 = self.multilines(x3[(z3 > zlim_3) & (z3 <= zlim_4)],y3[(z3 > zlim_3) & (z3 <= zlim_4)],L3[(z3 > zlim_3) & (z3 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax13.plot(np.nanmedian(median_wave3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(median_flux3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),color='k',lw=3.5)		
		axcb13 = fig.colorbar(lc13)
		axcb13.mappable.set_clim(clim1,clim2)
		# ax13.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax13.plot(np.nanmedian(median_upper_wave32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_32[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0),color='k',lw=2.0)
		# ax13.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		ax13.plot(np.nanmedian(wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0)[-3:], np.nanmedian(ffir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax13.plot(np.append(np.nanmedian(median_wave3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0)[-1],np.nanmedian(wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0)[:3]), np.append(np.nanmedian(median_flux3[(z3 > zlim_3) & (z3 <= zlim_4)],axis=0)[-1],np.nanmedian(ffir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax13.plot(np.nanmean(wfir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0), np.nanmean(ffir3[(z3 > zlim_3) & (z3 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		axcb13.remove()

		ax13.set_aspect(1)
		ax13.set_xscale('log')
		ax13.set_yscale('log')
		ax13.set_xlim(8E-5,7E2)
		ax13.set_ylim(1E-4,120)
		ax13.set_xticklabels([])
		ax13.set_yticklabels([])
		ax13.set_xticks(xticks)
		ax13.set_yticks(yticks)
		# ax13.set_xticklabels(xticks_labels)
		ax13.text(0.05,0.7,f'n = {len(x3[(z3 > zlim_3) & (z3 <= zlim_4)])}',transform=ax13.transAxes)
		ax13.text(0.75,0.08,str((len(x3[(z3 > zlim_3) & (z3 <= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax13.transAxes,weight='bold')

		ax14 = plt.subplot(gs[3,2])

		# upper_seg14 = np.stack((median_upper_wave4[(z4 > zlim_3) & (z4 <= zlim_4)], cosmos_s82x_list_4[(z4 > zlim_3) & (z4 <= zlim_4)]), axis=2)
		upper_seg14 = np.stack((wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], ffir4[(z4 > zlim_3) & (z4 <= zlim_4)]), axis=2)
		upper_all14 = LineCollection(upper_seg14,color='gray',alpha=0.3)
		ax14.add_collection(upper_all14)

		test = ax14.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc14 = self.multilines(x4[(z4 > zlim_3) & (z4 <= zlim_4)],y4[(z4 > zlim_3) & (z4 <= zlim_4)],L4[(z4 > zlim_3) & (z4 <= zlim_4)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax14.plot(np.nanmedian(median_wave4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(median_flux4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),color='k',lw=3.5)
		axcb14 = fig.colorbar(lc14)
		axcb14.mappable.set_clim(clim1,clim2)
		# ax14.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax14.plot(np.nanmedian(median_upper_wave42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_42[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0),color='k',lw=2.0)
		# ax14.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		ax14.plot(np.nanmedian(wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0)[-3:], np.nanmedian(ffir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax14.plot(np.append(np.nanmedian(median_wave4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0)[-1],np.nanmedian(wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0)[:3]), np.append(np.nanmedian(median_flux4[(z4 > zlim_3) & (z4 <= zlim_4)],axis=0)[-1],np.nanmedian(ffir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax14.plot(np.nanmean(wfir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0), np.nanmean(ffir4[(z4 > zlim_3) & (z4 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		axcb14.remove()

		ax14.set_aspect(1)
		ax14.set_xscale('log')
		ax14.set_yscale('log')
		ax14.set_xlim(8E-5,7E2)
		ax14.set_ylim(1E-4,120)
		ax14.set_xticklabels([])
		ax14.set_yticklabels([])
		ax14.set_xticks(xticks)
		ax14.set_yticks(yticks)
		# ax14.set_xticklabels(xticks_labels)
		ax14.text(0.05,0.7,f'n = {len(x4[(z4 > zlim_3) & (z4 <= zlim_4)])}',transform=ax14.transAxes)
		ax14.text(0.75,0.08,str((len(x4[(z4 > zlim_3) & (z4<= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax14.transAxes,weight='bold')

		ax15 = plt.subplot(gs[4,2])

		# upper_seg15 = np.stack((median_upper_wave5[(z5 > zlim_3) & (z5 <= zlim_4)], cosmos_s82x_list_5[(z5 > zlim_3) & (z5 <= zlim_4)]), axis=2)
		upper_seg15 = np.stack((wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], ffir5[(z5 > zlim_3) & (z5 <= zlim_4)]), axis=2)
		upper_all15 = LineCollection(upper_seg15,color='gray',alpha=0.3)
		ax15.add_collection(upper_all15)

		test = ax15.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc15 = self.multilines(x5[(z5 > zlim_3) & (z5 <= zlim_4)], y5[(z5 > zlim_3) & (z5 <= 1.1)], L5[(z5 > zlim_3) & (z5 <= zlim_4)], cmap='rainbow_r', alpha=alpha, lw=1.5, rasterized=True)
		ax15.plot(np.nanmedian(median_wave5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(median_flux5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),color='k',lw=3.5)		
		axcb15 = fig.colorbar(lc15)
		axcb15.mappable.set_clim(clim1,clim2)
		# ax15.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),'v',ms=5,color='k')
		# ax15.plot(np.nanmedian(median_upper_wave52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmedian(cosmos_s82x_list_52[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0),color='k',lw=2.0)
		# ax15.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		ax15.plot(np.nanmedian(wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0)[-3:], np.nanmedian(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0)[-3:],'-v',color='k',lw=2.0)
		ax15.plot(np.append(np.nanmedian(median_wave5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0)[-1],np.nanmedian(wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0)[:3]), np.append(np.nanmedian(median_flux5[(z5 > zlim_3) & (z5 <= zlim_4)],axis=0)[-1],np.nanmedian(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0)[:3]),'--',color='k',lw=2.0)
		# ax15.plot(np.nanmean(wfir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0), np.nanmean(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0),'-^',color='red',lw=1.75)
		axcb15.remove()

		ax15.set_aspect(1)
		ax15.set_xscale('log')
		ax15.set_yscale('log')
		ax15.set_xlim(8E-5,7E2)
		ax15.set_ylim(1E-4, 120)
		# ax15.set_xticklabels([])
		ax15.set_yticklabels([])
		ax15.set_xticks(xticks)
		ax15.set_yticks(yticks)
		ax15.set_xticklabels(xticks_labels)
		ax15.text(0.05,0.7,f'n = {len(x5[(z5 > zlim_3) & (z5 <= zlim_4)])}',transform=ax15.transAxes)
		ax15.text(0.75,0.08,str((len(x5[(z5 > zlim_3) & (z5<= zlim_4)])/len(x[(z > zlim_3) & (z <= zlim_4)]))*100)[0:4]+'%',transform=ax15.transAxes,weight='bold')
		# ax15.set_xlabel(r'Rest Wavelength [$\mu$m]')

		print(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)])
		print(np.nanmedian(ffir5[(z5 > zlim_3) & (z5 <= zlim_4)], axis=0))

		
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
		ax11.grid()
		ax12.grid()
		ax13.grid()
		ax14.grid()
		ax15.grid()


		cbar_ax = fig.add_subplot(gs[:,-1:])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg/s]',fontsize=32)

		# plt.tight_layout()
	
		plt.savefig('/Users/connor_auge/Desktop/New_plots3/5paneles_zbins'+savestring+'.pdf')
		plt.show()

	def plot_5panel_field(self,savestring,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,suptitle=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None,wfir=None,ffir=None):

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-3] = np.nan
		y[y < 1E-3] = np.nan

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		z_med = np.nanmedian(spec_z)

		clim1 = 43
		clim2 = 45.5

		cosmos_s82x_list, cosmos_s82x_list2 = [], []
		cosmos_s82x_wave, cosmos_s82x_wave2 = [], []
		for i in range(len(y)):

			upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq



			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]



			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(cosmos_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(s82X_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-8], y[i][-7], y[i][-6]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsN_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
					cosmos_s82x_list2.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
					cosmos_s82x_list2.append(Flux_to_Lum(goodsS_nuFnu_upper[-3:],spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					cosmos_s82x_list2.append([y[i][-5], y[i][-4], y[i][-3]])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			else:
				cosmos_s82x_list.append([np.nan,np.nan,np.nan,np.nan,np.nan])
				cosmos_s82x_list2.append([np.nan,np.nan,np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)
			cosmos_s82x_wave2.append(rest_upper_w_microns[-3:])


		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)
		cosmos_s82x_wave2 = np.asarray(cosmos_s82x_wave2)
		cosmos_s82x_list2 = np.asarray(cosmos_s82x_list2)


		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		# B1_check = (uv_slope < -0.3) & (mir_slope1 >= -0.2)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		alpha = 0.7

		
		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]


		plt.rcParams['font.size']=24
		plt.rcParams['axes.linewidth']=2.5
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-3,1E-2,0.1,1,10,100]
		z = spec_z

	
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		z1 = spec_z[B1]
		cosmos_s82x_list_1 = cosmos_s82x_list[B1]
		median_upper_wave1 = cosmos_s82x_wave[B1]
		cosmos_s82x_list_12 = cosmos_s82x_list2[B1]
		median_upper_wave12 = cosmos_s82x_wave2[B1]
		median_wave1 = 10**median_wavelength[B1]
		median_flux1 = 10**median_flux[B1]
		# wfir1 = np.asarray([wfir[B1][i]/norm1[i] for i in range(len(wfir[B1]))])
		wfir1 = wfir[B1]
		ffir1 = np.asarray([ffir[B1][i]/norm1[i] for i in range(len(ffir[B1]))])


		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		z2 = spec_z[B2]
		cosmos_s82x_list_2 = cosmos_s82x_list[B2]
		median_upper_wave2 = cosmos_s82x_wave[B2]
		cosmos_s82x_list_22 = cosmos_s82x_list2[B2]
		median_upper_wave22 = cosmos_s82x_wave2[B2]
		median_wave2 = 10**median_wavelength[B2]
		median_flux2 = 10**median_flux[B2]
		# wfir2 = np.asarray([wfir[B2][i]/norm2[i] for i in range(len(wfir[B2]))])
		wfir2 = wfir[B2]
		ffir2 = np.asarray([ffir[B2][i]/norm2[i] for i in range(len(ffir[B2]))])
	
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		z3 = spec_z[B3]
		cosmos_s82x_list_3 = cosmos_s82x_list[B3]
		median_upper_wave3 = cosmos_s82x_wave[B3]
		cosmos_s82x_list_32 = cosmos_s82x_list2[B3]
		median_upper_wave32 = cosmos_s82x_wave2[B3]
		median_wave3 = 10**median_wavelength[B3]
		median_flux3 = 10**median_flux[B3]
		# wfir3 = np.asarray([wfir[B3][i]/norm3[i] for i in range(len(wfir[B3]))])
		wfir3 = wfir[B3]
		ffir3 = np.asarray([ffir[B3][i]/norm3[i] for i in range(len(ffir[B3]))])
		
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		z4 = spec_z[B4]
		cosmos_s82x_list_4 = cosmos_s82x_list[B4]
		median_upper_wave4 = cosmos_s82x_wave[B4]
		cosmos_s82x_list_42 = cosmos_s82x_list2[B4]
		median_upper_wave42 = cosmos_s82x_wave2[B4]
		median_wave4 = 10**median_wavelength[B4]
		median_flux4 = 10**median_flux[B4]
		# wfir4 = np.asarray([wfir[B4][i]/norm4[i] for i in range(len(wfir[B4]))])
		wfir4 = wfir[B4]
		ffir4 = np.asarray([ffir[B4][i]/norm4[i] for i in range(len(ffir[B4]))])

		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		z5 = spec_z[B5]
		cosmos_s82x_list_5 = cosmos_s82x_list[B5]
		median_upper_wave5 = cosmos_s82x_wave[B5]
		cosmos_s82x_list_52 = cosmos_s82x_list2[B5]
		median_upper_wave52 = cosmos_s82x_wave2[B5]
		median_wave5 = 10**median_wavelength[B5]
		median_flux5 = 10**median_flux[B5]
		# wfir5 = np.asarray([wfir[B5][i]/norm5[i] for i in range(len(wfir[B5]))])
		wfir5 = wfir[B5]
		ffir5 = np.asarray([ffir[B5][i]/norm5[i] for i in range(len(ffir[B5]))])

		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']
		xticks = [1E-4,1E-3,1E-2,1E-1,1,10,100]

		


		fig = plt.figure(figsize=(21,27))
		gs = fig.add_gridspec(nrows=5, ncols=4,width_ratios=[3.25,3.25,3.25,0.2])
		gs.update(left=0.1, right=0.9, top=0.93, bottom=0.08)
		gs.update(hspace=0.07,wspace=-0.22) # set the spacing between axes
		


		ax1 = plt.subplot(gs[0,0])

		upper_seg1 = np.stack((median_upper_wave1[np.logical_or(mark1 == 2, mark1 == 3)], cosmos_s82x_list_1[np.logical_or(mark1 == 2, mark1 == 3)]), axis=2)
		upper_all1 = LineCollection(upper_seg1,color='gray',alpha=0.3)
		ax1.add_collection(upper_all1)

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc1 = self.multilines(x1[(z1 >= zlim_1) & (z1 <= zlim_2)], y1[np.logical_or(mark1 == 2, mark1 == 3)], L1[np.logical_or(mark1 == 2, mark1 == 3)], cmap='rainbow_r', lw=1.5, alpha = alpha, rasterized=True)
		ax1.plot(np.nanmedian(median_wave1[np.logical_or(mark1 == 2, mark1 == 3)],axis=0),np.nanmedian(median_flux1[np.logical_or(mark1 == 2, mark1 == 3)],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		ax1.plot(np.nanmedian(median_upper_wave12[np.logical_or(mark1 == 2, mark1 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_12[np.logical_or(mark1 == 2, mark1 == 3)],axis=0),'v',ms=5,color='k')
		ax1.plot(np.nanmedian(median_upper_wave12[np.logical_or(mark1 == 2, mark1 == 3)], axis=0), np.nanmedian(cosmos_s82x_list_12[np.logical_or(mark1 == 2, mark1 == 3)], axis=0), color='k', lw=2.0)
		# ax1.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_1) & (z1 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# ax1.plot(np.nanmedian(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmedian(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# ax1.plot(np.nanmean(wfir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0), np.nanmean(ffir1[(z1 >= zlim_1) & (z1 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)
		axcb1.remove()
	

		ax1.set_aspect(1)
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(1E-4,120)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		# ax1.set_xticklabels(xticks_labels)
		ax1.text(0.05,0.7,f'n = {len(x1[np.logical_or(mark1 == 2, mark1 == 3)])}',transform=ax1.transAxes)
		ax1.text(0.75,0.08,str((len(x1[np.logical_or(mark1 == 2, mark1 == 3)])/len(x[np.logical_or(mark == 2, mark == 3)]))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		# ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax1.text(0.0,1.03,r'A',transform=ax1.transAxes,fontsize=27,weight='bold')
		# ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		ax1.set_title('GOODS-N/S')
		ax1.text(-0.45,0.5,'1',transform=ax1.transAxes,fontsize=38,weight='bold')

		ax2 = plt.subplot(gs[1,0])

		upper_seg2 = np.stack((median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)], cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)]), axis=2)
		upper_all2 = LineCollection(upper_seg2,color='gray',alpha=0.3)
		ax2.add_collection(upper_all2)

		lc2 = self.multilines(x2[np.logical_or(mark2 == 2, mark2 == 3)],y2[np.logical_or(mark2 == 2, mark2 == 3)],L2[np.logical_or(mark2 == 2, mark2 == 3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(median_wave2[np.logical_or(mark2 == 2, mark2 == 3)], axis=0), np.nanmedian(median_flux2[np.logical_or(mark2 == 2, mark2 == 3)], axis=0), color='k', lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		ax2.plot(np.nanmedian(median_upper_wave22[np.logical_or(mark2 == 2, mark2 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_22[np.logical_or(mark2 == 2, mark2 == 3)],axis=0),'v',ms=5,color='k')
		ax2.plot(np.nanmedian(median_upper_wave22[np.logical_or(mark2 == 2, mark2 == 3)], axis=0), np.nanmedian(cosmos_s82x_list_22[np.logical_or(mark2 == 2, mark2 == 3)], axis=0), color='k', lw=2.0)
		# ax2.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_1) & (z2 <= zlim_2)],axis=0),'--',color='k',lw=2.0)
		# ax2.plot(np.nanmedian(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmedian(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0),'-x',color='orange',lw=1.75)
		# ax2.plot(np.nanmean(wfir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0), np.nanmean(ffir2[(z2 >= zlim_1) & (z2 <= zlim_2)], axis=0),'-^',color='red',lw=1.75)		
		# axcb2.remove()

		ax2.set_aspect(1)
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(1E-4,120)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		# ax2.set_xticklabels(xticks_labels)
		ax2.text(0.05,0.7,f'n = {len(x2[np.logical_or(mark2 == 2, mark2 == 3)])}',transform=ax2.transAxes)
		ax2.text(0.75, 0.08, str((len(x2[np.logical_or(mark2 == 2, mark2 == 3)])/len(x[np.logical_or(mark == 2, mark == 3)]))*100)[0:4]+'%', transform=ax2.transAxes, weight='bold')
		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax2.text(-0.45,0.5,'2',transform=ax2.transAxes,fontsize=38,weight='bold')

		ax3 = plt.subplot(gs[2,0])

		upper_seg3 = np.stack((median_upper_wave3[np.logical_or(mark3 == 2, mark3 == 3)], cosmos_s82x_list_3[np.logical_or(mark3 == 2, mark3 == 3)]), axis=2)
		upper_all3 = LineCollection(upper_seg3,color='gray',alpha=0.3)
		ax3.add_collection(upper_all3)

		lc3 = self.multilines(x3[np.logical_or(mark3 == 2, mark3 == 3)],y3[np.logical_or(mark3 == 2, mark3 == 3)],L3[np.logical_or(mark3 == 2, mark3 == 3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(median_wave3[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),np.nanmedian(median_flux3[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		ax3.plot(np.nanmedian(median_upper_wave32[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_32[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),'v',ms=5,color='k')
		ax3.plot(np.nanmedian(median_upper_wave32[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_32[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),color='k',lw=2.0)
		# ax3.plot(np.nanmean(median_upper_wave3[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),np.nanmean(cosmos_s82x_list_3[np.logical_or(mark3 == 2, mark3 == 3)],axis=0),'--',color='k',lw=2.0)
		# ax3.plot(np.nanmedian(wfir3[np.logical_or(mark3 == 2, mark3 == 3)], axis=0), np.nanmedian(ffir3[np.logical_or(mark3 == 2, mark3 == 3)], axis=0),'-x',color='orange',lw=1.75)
		# ax3.plot(np.nanmean(wfir3[np.logical_or(mark3 == 2, mark3 == 3)], axis=0), np.nanmean(ffir3[np.logical_or(mark3 == 2, mark3 == 3)], axis=0),'-^',color='red',lw=1.75)
		axcb3.remove()

		ax3.set_aspect(1)
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(1E-4,120)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_xticklabels(xticks_labels)
		ax3.text(0.05,0.7,f'n = {len(x3[np.logical_or(mark3 == 2, mark3 == 3)])}',transform=ax3.transAxes)
		ax3.text(0.75,0.08,str((len(x3[np.logical_or(mark3 == 2, mark3 == 3)])/len(x[np.logical_or(mark == 2, mark == 3)]))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		ax3.set_ylabel(r'Normalized $\lambda$ L$_\lambda$',fontsize=40)
		ax3.text(-0.45,0.5,'3',transform=ax3.transAxes,fontsize=38,weight='bold')

		ax4 = plt.subplot(gs[3,0])

		upper_seg4 = np.stack((median_upper_wave4[np.logical_or(mark4 == 2, mark4 == 3)], cosmos_s82x_list_4[np.logical_or(mark4 == 2, mark4 == 3)]), axis=2)
		upper_all4 = LineCollection(upper_seg4,color='gray',alpha=0.3)
		ax4.add_collection(upper_all4)

		lc4 = self.multilines(x4[np.logical_or(mark4 == 2, mark4 == 3)],y4[np.logical_or(mark4 == 2, mark4 == 3)],L4[np.logical_or(mark4 == 2, mark4 == 3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax4.plot(np.nanmedian(median_wave4[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),np.nanmedian(median_flux4[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		ax4.plot(np.nanmedian(median_upper_wave42[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_42[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),'v',ms=5,color='k')
		ax4.plot(np.nanmedian(median_upper_wave42[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_42[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),color='k',lw=2.0)
		# ax4.plot(np.nanmean(median_upper_wave4[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),np.nanmean(cosmos_s82x_list_4[np.logical_or(mark4 == 2, mark4 == 3)],axis=0),'--',color='k',lw=2.0)
		# ax4.plot(np.nanmedian(wfir4[np.logical_or(mark4 == 2, mark4 == 3)], axis=0), np.nanmedian(ffir4[np.logical_or(mark4 == 2, mark4 == 3)], axis=0),'-x',color='orange',lw=1.75)
		# ax4.plot(np.nanmean(wfir4[np.logical_or(mark4 == 2, mark4 == 3)], axis=0), np.nanmean(ffir4[np.logical_or(mark4 == 2, mark4 == 3)], axis=0),'-^',color='red',lw=1.75)
		# axcb4.remove()

		ax4.set_aspect(1)
		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(1E-4,120)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		# ax4.set_xticklabels(xticks_labels)
		ax4.text(0.05,0.7,f'n = {len(x4[np.logical_or(mark4 == 2, mark4 == 3)])}',transform=ax4.transAxes)
		ax4.text(0.75,0.08,str((len(x4[np.logical_or(mark4 == 2, mark4 == 3)])/len(x[np.logical_or(mark == 2, mark == 3)]))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		# ax4.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax4.text(-0.45,0.5,'4',transform=ax4.transAxes,fontsize=38,weight='bold')

		ax5 = plt.subplot(gs[4,0])

		upper_seg5 = np.stack((median_upper_wave5[np.logical_or(mark5 == 2, mark5 == 3)], cosmos_s82x_list_5[np.logical_or(mark5 == 2, mark5 == 3)]), axis=2)
		upper_all5 = LineCollection(upper_seg5,color='gray',alpha=0.3)
		ax5.add_collection(upper_all5)

		lc5 = self.multilines(x5[np.logical_or(mark5 == 2, mark5 == 3)],y5[np.logical_or(mark5 == 2, mark5 == 3)],L5[np.logical_or(mark5 == 2, mark5 == 3)],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax5.plot(np.nanmedian(median_wave5[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),np.nanmedian(median_flux5[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),color='k',lw=3.5)
		axcb5 = fig.colorbar(lc5)
		axcb5.mappable.set_clim(clim1,clim2)
		ax5.plot(np.nanmedian(median_upper_wave52[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_52[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),'v',ms=5,color='k')
		ax5.plot(np.nanmedian(median_upper_wave52[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),np.nanmedian(cosmos_s82x_list_52[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),color='k',lw=2.0)
		# ax5.plot(np.nanmean(median_upper_wave5[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),np.nanmean(cosmos_s82x_list_5[np.logical_or(mark5 == 2, mark5 == 3)],axis=0),'--',color='k',lw=2.0)
		# ax5.plot(np.nanmedian(wfir5[np.logical_or(mark5 == 2, mark5 == 3)], axis=0), np.nanmedian(ffir5[np.logical_or(mark5 == 2, mark5 == 3)], axis=0),'-x',color='orange',lw=1.75)
		# ax5.plot(np.nanmean(wfir5[np.logical_or(mark5 == 2, mark5 == 3)], axis=0), np.nanmean(ffir5[np.logical_or(mark5 == 2, mark5 == 3)], axis=0),'-^',color='red',lw=1.75)
		# axcb5.remove()

		ax5.set_aspect(1)
		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(1E-4,120)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)
		ax5.set_xticklabels(xticks_labels)
		ax5.text(0.05,0.7,f'n = {len(x5[np.logical_or(mark5 == 2, mark5 == 3)])}',transform=ax5.transAxes)
		ax5.text(0.75,0.08,str((len(x5[np.logical_or(mark5 == 2, mark5 == 3)])/len(x[np.logical_or(mark == 2, mark == 3)]))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		# ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax5.text(-0.45,0.5,'5',transform=ax5.transAxes,fontsize=38,weight='bold')




		ax6 = plt.subplot(gs[0,1])

		upper_seg6 = np.stack((median_upper_wave1[mark1 == 0], cosmos_s82x_list_1[mark1 == 0]), axis=2)
		upper_all6 = LineCollection(upper_seg6,color='gray',alpha=0.3)
		ax6.add_collection(upper_all6)

		lc6 = self.multilines(x1[mark1 == 0],y1[mark1 == 0],L1[mark1 == 0],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax6.plot(np.nanmedian(median_wave1[mark1 == 0],axis=0),np.nanmedian(median_flux1[mark1 == 0],axis=0),color='k',lw=3.5)
		axcb6 = fig.colorbar(lc6)
		axcb6.mappable.set_clim(clim1,clim2)
		ax6.plot(np.nanmedian(median_upper_wave12[mark1 == 0],axis=0),np.nanmedian(cosmos_s82x_list_12[mark1 == 0],axis=0),'v',ms=5,color='k')
		ax6.plot(np.nanmedian(median_upper_wave12[mark1 == 0],axis=0),np.nanmedian(cosmos_s82x_list_12[mark1 == 0],axis=0),color='k',lw=2.0)
		# ax6.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_2) & (z1 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# ax6.plot(np.nanmedian(wfir1[mark1 == 0], axis=0), np.nanmedian(ffir1[mark1 == 0], axis=0),'-x',color='orange',lw=1.75)
		# ax6.plot(np.nanmean(wfir1[mark1 == 0], axis=0), np.nanmean(ffir1[mark1 == 0], axis=0),'-^',color='red',lw=1.75)
		# axcb6.remove()

		ax6.set_aspect(1)
		ax6.set_xscale('log')
		ax6.set_yscale('log')
		ax6.set_xlim(8E-5,7E2)
		ax6.set_ylim(1E-4,120)
		ax6.set_xticklabels([])
		ax6.set_yticklabels([])
		ax6.set_xticks(xticks)
		ax6.set_yticks(yticks)
		# ax6.set_xticklabels(xticks_labels)
		ax6.text(0.05,0.7,f'n = {len(x1[mark1 == 0])}',transform=ax6.transAxes)
		ax6.text(0.75,0.08,str((len(x1[mark1 == 0])/len(x[mark == 0]))*100)[0:4]+'%',transform=ax6.transAxes,weight='bold')
		ax6.text(0.0,1.03,r'B',transform=ax6.transAxes,fontsize=27,weight='bold')
		# ax6.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		ax6.set_title('COSMOS')

		ax7 = plt.subplot(gs[1,1])

		upper_seg7 = np.stack((median_upper_wave2[mark2 == 0], cosmos_s82x_list_2[mark2 == 0]), axis=2)
		upper_all7 = LineCollection(upper_seg7,color='gray',alpha=0.3)
		ax7.add_collection(upper_all7)

		lc7 = self.multilines(x2[mark2 == 0],y2[mark2 == 0],L2[mark2 == 0],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax7.plot(np.nanmedian(median_wave2[mark2 == 0], axis=0), np.nanmedian(median_flux2[mark2 == 0], axis=0), color='k', lw=3.5)		
		axcb7 = fig.colorbar(lc7)
		axcb7.mappable.set_clim(clim1,clim2)
		ax7.plot(np.nanmedian(median_upper_wave22[mark2 == 0],axis=0),np.nanmedian(cosmos_s82x_list_22[mark2 == 0],axis=0),'v',ms=5,color='k')
		ax7.plot(np.nanmedian(median_upper_wave22[mark2 == 0],axis=0),np.nanmedian(cosmos_s82x_list_22[mark2 == 0],axis=0),color='k',lw=2.0)
		# ax7.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_2) & (z2 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# ax7.plot(np.nanmedian(wfir2[mark2 == 0], axis=0), np.nanmedian(ffir2[mark2 == 0], axis=0),'-x',color='orange',lw=1.75)
		# ax7.plot(np.nanmean(wfir2[mark2 == 0], axis=0), np.nanmean(ffir2[mark2 == 0], axis=0),'-^',color='red',lw=1.75)
		# axcb7.remove()

		ax7.set_aspect(1)
		ax7.set_xscale('log')
		ax7.set_yscale('log')
		ax7.set_xlim(8E-5,7E2)
		ax7.set_ylim(1E-4,120)
		ax7.set_xticklabels([])
		ax7.set_yticklabels([])
		ax7.set_xticks(xticks)
		ax7.set_yticks(yticks)
		# ax7.set_xticklabels(xticks_labels)
		ax7.text(0.05,0.7,f'n = {len(x2[mark2 == 0])}',transform=ax7.transAxes)
		ax7.text(0.75,0.08,str((len(x2[mark2 == 0])/len(x[mark == 0]))*100)[0:4]+'%',transform=ax7.transAxes,weight='bold')

		ax8 = plt.subplot(gs[2,1])

		upper_seg8 = np.stack((median_upper_wave3[mark3 == 0], cosmos_s82x_list_3[mark3 == 0]), axis=2)
		upper_all8 = LineCollection(upper_seg8,color='gray',alpha=0.3)
		ax8.add_collection(upper_all8)

		lc8 = self.multilines(x3[mark3 == 0],y3[mark3 == 0],L3[mark3 == 0],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax8.plot(np.nanmedian(median_wave3[mark3 == 0],axis=0),np.nanmedian(median_flux3[mark3 == 0],axis=0),color='k',lw=3.5)
		axcb8 = fig.colorbar(lc8)
		axcb8.mappable.set_clim(clim1,clim2)
		ax8.plot(np.nanmedian(median_upper_wave32[mark3 == 0],axis=0),np.nanmedian(cosmos_s82x_list_32[mark3 == 0],axis=0),'v',ms=5,color='k')
		ax8.plot(np.nanmedian(median_upper_wave32[mark3 == 0],axis=0),np.nanmedian(cosmos_s82x_list_32[mark3 == 0],axis=0),color='k',lw=2.0)
		# ax8.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_2) & (z3 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# ax8.plot(np.nanmedian(wfir3[mark3 == 0], axis=0), np.nanmedian(ffir3[mark3 == 0], axis=0),'-x',color='orange',lw=1.75)
		# ax8.plot(np.nanmean(wfir3[mark3 == 0], axis=0), np.nanmean(ffir3[mark3 == 0], axis=0),'-^',color='red',lw=1.75)
		axcb8.remove()

		ax8.set_aspect(1)
		ax8.set_xscale('log')
		ax8.set_yscale('log')
		ax8.set_xlim(8E-5,7E2)
		ax8.set_ylim(1E-4,120)
		ax8.set_xticklabels([])
		ax8.set_yticklabels([])
		ax8.set_xticks(xticks)
		ax8.set_yticks(yticks)
		# ax8.set_xticklabels(xticks_labels)
		ax8.text(0.05,0.7,f'n = {len(x3[mark3 == 0])}',transform=ax8.transAxes)
		ax8.text(0.75,0.08,str((len(x3[mark3 == 0])/len(x[mark == 0]))*100)[0:4]+'%',transform=ax8.transAxes,weight='bold')

		ax9 = plt.subplot(gs[3,1])

		upper_seg9 = np.stack((median_upper_wave4[mark4 == 0], cosmos_s82x_list_4[mark4 == 0]), axis=2)
		upper_all9 = LineCollection(upper_seg9,color='gray',alpha=0.3)
		ax9.add_collection(upper_all9)

		lc9 = self.multilines(x4[mark4 == 0],y4[(z4 > zlim_2) & (z4 <= 0.8)],L4[mark4 == 0],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax9.plot(np.nanmedian(median_wave4[mark4 == 0],axis=0),np.nanmedian(median_flux4[mark4 == 0],axis=0),color='k',lw=3.5)
		axcb9 = fig.colorbar(lc9)
		axcb9.mappable.set_clim(clim1,clim2)
		ax9.plot(np.nanmedian(median_upper_wave42[mark4 == 0],axis=0),np.nanmedian(cosmos_s82x_list_42[mark4 == 0],axis=0),'v',ms=5,color='k')
		ax9.plot(np.nanmedian(median_upper_wave42[mark4 == 0],axis=0),np.nanmedian(cosmos_s82x_list_42[mark4 == 0],axis=0),color='k',lw=2.0)
		# ax9.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_2) & (z4 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# ax9.plot(np.nanmedian(wfir4[mark4 == 0], axis=0), np.nanmedian(ffir4[mark4 == 0], axis=0),'-x',color='orange',lw=1.75)
		# ax9.plot(np.nanmean(wfir4[mark4 == 0], axis=0), np.nanmean(ffir4[mark4 == 0], axis=0),'-^',color='red',lw=1.75)
		# axcb9.remove()

		ax9.set_aspect(1)
		ax9.set_xscale('log')
		ax9.set_yscale('log')
		ax9.set_xlim(8E-5,7E2)
		ax9.set_ylim(1E-4,120)
		ax9.set_xticklabels([])
		ax9.set_yticklabels([])
		ax9.set_xticks(xticks)
		ax9.set_yticks(yticks)
		# ax9.set_xticklabels(xticks_labels)
		ax9.text(0.05,0.7,f'n = {len(x4[mark4 == 0])}',transform=ax9.transAxes)
		ax9.text(0.75,0.08,str((len(x4[mark4 == 0])/len(x[mark == 0]))*100)[0:4]+'%',transform=ax9.transAxes,weight='bold')

		ax10 = plt.subplot(gs[4,1])

		upper_seg10 = np.stack((median_upper_wave5[mark5 == 0], cosmos_s82x_list_5[mark5 == 0]), axis=2)
		upper_all10 = LineCollection(upper_seg10,color='gray',alpha=0.3)
		ax10.add_collection(upper_all10)

		lc10 = self.multilines(x5[mark5 == 0],y5[(z5 > zlim_2) & (z5 <= 0.8)],L5[mark5 == 0],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax10.plot(np.nanmedian(median_wave5[mark5 == 0],axis=0),np.nanmedian(median_flux5[mark5 == 0],axis=0),color='k',lw=3.5)
		axcb10 = fig.colorbar(lc10)	
		axcb10.mappable.set_clim(clim1,clim2)
		ax10.plot(np.nanmedian(median_upper_wave52[mark5 == 0],axis=0),np.nanmedian(cosmos_s82x_list_52[mark5 == 0],axis=0),'v',ms=5,color='k')
		ax10.plot(np.nanmedian(median_upper_wave52[mark5 == 0],axis=0),np.nanmedian(cosmos_s82x_list_52[mark5 == 0],axis=0),color='k',lw=2.0)
		# ax10.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_2) & (z5 <= zlim_3)],axis=0),'--',color='k',lw=2.0)
		# ax10.plot(np.nanmedian(wfir5[mark5 == 0], axis=0), np.nanmedian(ffir5[mark5 == 0], axis=0),'-x',color='orange',lw=1.75)
		# ax10.plot(np.nanmean(wfir5[mark5 == 0], axis=0), np.nanmean(ffir5[mark5 == 0], axis=0),'-^',color='red',lw=1.75)		
		axcb10.remove()

		ax10.set_aspect(1)
		ax10.set_xscale('log')
		ax10.set_yscale('log')
		ax10.set_xlim(8E-5,7E2)
		ax10.set_ylim(1E-4,120)
		ax10.set_yticklabels([])
		ax10.set_xticks(xticks)
		ax10.set_yticks(yticks)
		ax10.set_xticklabels(xticks_labels)
		ax10.text(0.05,0.7,f'n = {len(x5[mark5 == 0])}',transform=ax10.transAxes)
		ax10.text(0.75,0.08,str((len(x5[mark5 == 0])/len(x[mark == 0]))*100)[0:4]+'%',transform=ax10.transAxes,weight='bold')
		ax10.set_xlabel(r'Rest Wavelength [$\mu$m]', fontsize=40)




		ax11 = plt.subplot(gs[0,2])

		upper_seg11 = np.stack((median_upper_wave1[mark1 == 1], cosmos_s82x_list_1[mark1 == 1]), axis=2)
		upper_all11 = LineCollection(upper_seg11,color='gray',alpha=0.3)
		ax11.add_collection(upper_all11)
		
		test = ax11.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc11 = self.multilines(x1[mark1 == 1],y1[mark1 == 1],L1[mark1 == 1],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax11.plot(np.nanmedian(median_wave1[mark1 == 1],axis=0),np.nanmedian(median_flux1[mark1 == 1],axis=0),color='k',lw=3.5)
		axcb11 = fig.colorbar(lc11)
		axcb11.mappable.set_clim(clim1,clim2)
		ax11.plot(np.nanmedian(median_upper_wave12[mark1 == 1],axis=0),np.nanmedian(cosmos_s82x_list_12[mark1 == 1],axis=0),'v',ms=5,color='k')
		ax11.plot(np.nanmedian(median_upper_wave12[mark1 == 1],axis=0),np.nanmedian(cosmos_s82x_list_12[mark1 == 1],axis=0),color='k',lw=2.0)
		# ax11.plot(np.nanmean(median_upper_wave1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_1[(z1 >= zlim_3) & (z1 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax11.plot(np.nanmedian(wfir1[mark1 == 1], axis=0), np.nanmedian(ffir1[mark1 == 1], axis=0),'-x',color='orange',lw=1.75)
		# ax11.plot(np.nanmean(wfir1[mark1 == 1], axis=0), np.nanmean(ffir1[mark1 == 1], axis=0),'-^',color='red',lw=1.75)
		axcb11.remove()

		ax11.set_aspect(1)
		ax11.set_xscale('log')
		ax11.set_yscale('log')
		ax11.set_xlim(8E-5,7E2)
		ax11.set_ylim(1E-4,120)
		ax11.set_xticklabels([])
		ax11.set_yticklabels([])
		ax11.set_xticks(xticks)
		ax11.set_yticks(yticks)
		# ax11.set_xticklabels(xticks_labels)
		ax11.text(0.05,0.7,f'n = {len(x1[mark1 == 1])}',transform=ax11.transAxes)
		ax11.text(0.75,0.08,str((len(x1[mark1 == 1])/len(x[mark == 1]))*100)[0:4]+'%',transform=ax11.transAxes,weight='bold')
		ax11.text(0.0,1.03,r'C',transform=ax11.transAxes,fontsize=27,weight='bold')
		# ax11.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		ax11.set_title('Stripe 82X')

		ax12 = plt.subplot(gs[1,2])

		upper_seg12 = np.stack((median_upper_wave2[mark2 == 1], cosmos_s82x_list_2[mark2 == 1]), axis=2)
		upper_all12 = LineCollection(upper_seg12,color='gray',alpha=0.3)
		ax12.add_collection(upper_all12)

		test = ax12.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc12 = self.multilines(x2[mark2 == 1],y2[mark2 == 1],L2[mark2 == 1],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax12.plot(np.nanmedian(median_wave2[mark2 == 1],axis=0),np.nanmedian(median_flux2[mark2 == 1],axis=0),color='k',lw=3.5)		
		axcb12 = fig.colorbar(lc12)
		axcb12.mappable.set_clim(clim1,clim2)
		ax12.plot(np.nanmedian(median_upper_wave22[mark2 == 1],axis=0),np.nanmedian(cosmos_s82x_list_22[mark2 == 1],axis=0),'v',ms=5,color='k')
		ax12.plot(np.nanmedian(median_upper_wave22[mark2 == 1],axis=0),np.nanmedian(cosmos_s82x_list_22[mark2 == 1],axis=0),color='k',lw=2.0)
		# ax12.plot(np.nanmean(median_upper_wave2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_2[(z2 >= zlim_3) & (z2 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax12.plot(np.nanmedian(wfir2[mark2 == 1], axis=0), np.nanmedian(ffir2[mark2 == 1], axis=0),'-x',color='orange',lw=1.75)
		# ax12.plot(np.nanmean(wfir2[mark2 == 1], axis=0), np.nanmean(ffir2[mark2 == 1], axis=0),'-^',color='red',lw=1.75)
		axcb12.remove()

		ax12.set_aspect(1)
		ax12.set_xscale('log')
		ax12.set_yscale('log')
		ax12.set_xlim(8E-5,7E2)
		ax12.set_ylim(1E-4,120)
		ax12.set_xticklabels([])
		ax12.set_yticklabels([])
		ax12.set_xticks(xticks)
		ax12.set_yticks(yticks)
		# ax12.set_xticklabels(xticks_labels)
		ax12.text(0.05,0.7,f'n = {len(x2[mark2 == 1])}',transform=ax12.transAxes)
		ax12.text(0.75,0.08,str((len(x2[mark2 == 1])/len(x[mark == 1]))*100)[0:4]+'%',transform=ax12.transAxes,weight='bold')

		ax13 = plt.subplot(gs[2,2])

		upper_seg13 = np.stack((median_upper_wave3[mark3 == 1], cosmos_s82x_list_3[mark3 == 1]), axis=2)
		upper_all13 = LineCollection(upper_seg13,color='gray',alpha=0.3)
		ax13.add_collection(upper_all13)

		test = ax13.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc13 = self.multilines(x3[mark3 == 1],y3[mark3 == 1],L3[mark3 == 1],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax13.plot(np.nanmedian(median_wave3[mark3 == 1],axis=0),np.nanmedian(median_flux3[mark3 == 1],axis=0),color='k',lw=3.5)		
		axcb13 = fig.colorbar(lc13)
		axcb13.mappable.set_clim(clim1,clim2)
		ax13.plot(np.nanmedian(median_upper_wave32[mark3 == 1],axis=0),np.nanmedian(cosmos_s82x_list_32[mark3 == 1],axis=0),'v',ms=5,color='k')
		ax13.plot(np.nanmedian(median_upper_wave32[mark3 == 1],axis=0),np.nanmedian(cosmos_s82x_list_32[mark3 == 1],axis=0),color='k',lw=2.0)
		# ax13.plot(np.nanmean(median_upper_wave3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_3[(z3 >= zlim_3) & (z3 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax13.plot(np.nanmedian(wfir3[mark3 == 1], axis=0), np.nanmedian(ffir3[mark3 == 1], axis=0),'-x',color='orange',lw=1.75)
		# ax13.plot(np.nanmean(wfir3[mark3 == 1], axis=0), np.nanmean(ffir3[mark3 == 1], axis=0),'-^',color='red',lw=1.75)
		axcb13.remove()

		ax13.set_aspect(1)
		ax13.set_xscale('log')
		ax13.set_yscale('log')
		ax13.set_xlim(8E-5,7E2)
		ax13.set_ylim(1E-4,120)
		ax13.set_xticklabels([])
		ax13.set_yticklabels([])
		ax13.set_xticks(xticks)
		ax13.set_yticks(yticks)
		# ax13.set_xticklabels(xticks_labels)
		ax13.text(0.05,0.7,f'n = {len(x3[mark3 == 1])}',transform=ax13.transAxes)
		ax13.text(0.75,0.08,str((len(x3[mark3 == 1])/len(x[mark == 1]))*100)[0:4]+'%',transform=ax13.transAxes,weight='bold')

		ax14 = plt.subplot(gs[3,2])

		upper_seg14 = np.stack((median_upper_wave4[mark4 == 1], cosmos_s82x_list_4[mark4 == 1]), axis=2)
		upper_all14 = LineCollection(upper_seg14,color='gray',alpha=0.3)
		ax14.add_collection(upper_all14)

		test = ax14.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc14 = self.multilines(x4[mark4 == 1],y4[mark4 == 1],L4[mark4 == 1],cmap='rainbow_r', alpha = alpha,lw=1.5,rasterized=True)
		ax14.plot(np.nanmedian(median_wave4[mark4 == 1],axis=0),np.nanmedian(median_flux4[mark4 == 1],axis=0),color='k',lw=3.5)
		axcb14 = fig.colorbar(lc14)
		axcb14.mappable.set_clim(clim1,clim2)
		ax14.plot(np.nanmedian(median_upper_wave42[mark4 == 1],axis=0),np.nanmedian(cosmos_s82x_list_42[mark4 == 1],axis=0),'v',ms=5,color='k')
		ax14.plot(np.nanmedian(median_upper_wave42[mark4 == 1],axis=0),np.nanmedian(cosmos_s82x_list_42[mark4 == 1],axis=0),color='k',lw=2.0)
		# ax14.plot(np.nanmean(median_upper_wave4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_4[(z4 >= zlim_3) & (z4 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax14.plot(np.nanmedian(wfir4[mark4 == 1], axis=0), np.nanmedian(ffir4[mark4 == 1], axis=0),'-x',color='orange',lw=1.75)
		# ax14.plot(np.nanmean(wfir4[mark4 == 1], axis=0), np.nanmean(ffir4[mark4 == 1], axis=0),'-^',color='red',lw=1.75)
		axcb14.remove()

		ax14.set_aspect(1)
		ax14.set_xscale('log')
		ax14.set_yscale('log')
		ax14.set_xlim(8E-5,7E2)
		ax14.set_ylim(1E-4,120)
		ax14.set_xticklabels([])
		ax14.set_yticklabels([])
		ax14.set_xticks(xticks)
		ax14.set_yticks(yticks)
		# ax14.set_xticklabels(xticks_labels)
		ax14.text(0.05,0.7,f'n = {len(x4[mark4 == 1])}',transform=ax14.transAxes)
		ax14.text(0.75,0.08,str((len(x4[(z4 > zlim_3) & (z4<= zlim_4)])/len(x[mark == 1]))*100)[0:4]+'%',transform=ax14.transAxes,weight='bold')

		ax15 = plt.subplot(gs[4,2])

		upper_seg15 = np.stack((median_upper_wave5[mark5 == 1], cosmos_s82x_list_5[mark5 == 1]), axis=2)
		upper_all15 = LineCollection(upper_seg15,color='gray',alpha=0.3)
		ax15.add_collection(upper_all15)

		test = ax15.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		lc15 = self.multilines(x5[mark5 == 1], y5[(z5 > zlim_3) & (z5 <= 1.1)], L5[mark5 == 1], cmap='rainbow_r', alpha=alpha, lw=1.5, rasterized=True)
		ax15.plot(np.nanmedian(median_wave5[mark5 == 1],axis=0),np.nanmedian(median_flux5[mark5 == 1],axis=0),color='k',lw=3.5)		
		axcb15 = fig.colorbar(lc15)
		axcb15.mappable.set_clim(clim1,clim2)
		ax15.plot(np.nanmedian(median_upper_wave52[mark5 == 1],axis=0),np.nanmedian(cosmos_s82x_list_52[mark5 == 1],axis=0),'v',ms=5,color='k')
		ax15.plot(np.nanmedian(median_upper_wave52[mark5 == 1],axis=0),np.nanmedian(cosmos_s82x_list_52[mark5 == 1],axis=0),color='k',lw=2.0)
		# ax15.plot(np.nanmean(median_upper_wave5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),np.nanmean(cosmos_s82x_list_5[(z5 >= zlim_3) & (z5 <= zlim_4)],axis=0),'--',color='k',lw=2.0)
		# ax15.plot(np.nanmedian(wfir5[mark5 == 1], axis=0), np.nanmedian(ffir5[mark5 == 1], axis=0),'-x',color='orange',lw=1.75)
		# ax15.plot(np.nanmean(wfir5[mark5 == 1], axis=0), np.nanmean(ffir5[mark5 == 1], axis=0),'-^',color='red',lw=1.75)
		axcb15.remove()

		ax15.set_aspect(1)
		ax15.set_xscale('log')
		ax15.set_yscale('log')
		ax15.set_xlim(8E-5,7E2)
		ax15.set_ylim(1E-4, 120)
		# ax15.set_xticklabels([])
		ax15.set_yticklabels([])
		ax15.set_xticks(xticks)
		ax15.set_yticks(yticks)
		ax15.set_xticklabels(xticks_labels)
		ax15.text(0.05,0.7,f'n = {len(x5[mark5 == 1])}',transform=ax15.transAxes)
		ax15.text(0.75,0.08,str((len(x5[mark5 == 1])/len(x[mark == 1]))*100)[0:4]+'%',transform=ax15.transAxes,weight='bold')
		# ax15.set_xlabel(r'Rest Wavelength [$\mu$m]')

		
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
		ax11.grid()
		ax12.grid()
		ax13.grid()
		ax14.grid()
		ax15.grid()


		cbar_ax = fig.add_subplot(gs[:,-1:])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg/s]',fontsize=32)

		# plt.tight_layout()
	
		plt.savefig('/Users/connor_auge/Desktop/final_paper_43/5paneles_field'+savestring+'.pdf')
		plt.show()


	def plot_median(self,emis1,emis2,emis3,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,median_xray_w=None,median_xray_f=None,median_FIR_w=None,median_FIR_f=None,goals_x=None,goals_y=None,goals_median_w=None,goals_median_f=None,goals_f1=None):

		inf = ascii.read('/Users/connor_auge/Desktop/A10_templates.txt')

		wave = np.asarray(inf['Wave'])
		wave_cgs = wave*1E-4
		freq = 3E10/wave_cgs
		E_flux = np.asarray(inf['E'])*1E-16 # erg/s/Hz
		E_nuFnu = E_flux*freq
		Im_flux = np.asarray(inf['Im'])*1E-15 # erg/s/Hz
		Im_nuFnu = Im_flux*freq
		Sbc_flux = np.asarray(inf['Sbc'])*1E-17 # erg/s/Hz
		Sbc_nuFnu = Sbc_flux*freq
		AGN1_flux = np.asarray(inf['AGN'])*1E-11 # erg/s/Hz
		AGN1_nuFnu = AGN1_flux*freq
		AGN2_flux = np.asarray(inf['AGN2'])*1E-14 # erg/s/Hz
		AGN2_nuFnu = AGN2_flux*freq
		E_Im_nuFnu = Im_nuFnu+E_nuFnu
		E_L = E_nuFnu*4*np.pi*3E19**2
		E_AGN1 = AGN1_nuFnu*4*np.pi*3E19**2
		E_AGN2 = AGN2_nuFnu*4*np.pi*3E19**2
		E_Sbc = Sbc_nuFnu*4*np.pi*3E19**2
		E_Im = Im_nuFnu*4*np.pi*3E19**2
		temp_comb = E_L+E_Sbc+E_Im

		clim1 = 42.5
		clim2 = 46

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		temp1 = temp_comb[wave == 1.0050]
		# norm_temp = np.nanmedian(F1[B4])/temp1
		norm_temp = 4.215928239976833e+44/temp1
		temp_comb *= norm_temp
		

		median_flux1 = np.nanmedian(10**median_flux[B1],axis=0)*np.nanmedian(F1[B1])
		median_flux2 = np.nanmedian(10**median_flux[B2],axis=0)*np.nanmedian(F1[B2])
		median_flux3 = np.nanmedian(10**median_flux[B3],axis=0)*np.nanmedian(F1[B3])
		median_flux4 = np.nanmedian(10**median_flux[B4],axis=0)*np.nanmedian(F1[B4])
		median_flux5 = np.nanmedian(10**median_flux[B5],axis=0)*np.nanmedian(F1[B5])
		# median_flux_goals = np.nanmedian(10**goals_median_f,axis=0)*np.nanmedian(goals_f1)


		percentile_20_flux1 = np.nanpercentile(10**median_flux[B1],20,axis=0)*np.nanpercentile(F1[B1],20)
		percentile_80_flux1 = np.nanpercentile(10**median_flux[B1],80,axis=0)*np.nanpercentile(F1[B1],80)
		percentile_20_flux2 = np.nanpercentile(10**median_flux[B2],20,axis=0)*np.nanpercentile(F1[B2],20)
		percentile_80_flux2 = np.nanpercentile(10**median_flux[B2],80,axis=0)*np.nanpercentile(F1[B2],80)
		percentile_20_flux3 = np.nanpercentile(10**median_flux[B3],20,axis=0)*np.nanpercentile(F1[B3],20)
		percentile_80_flux3 = np.nanpercentile(10**median_flux[B3],80,axis=0)*np.nanpercentile(F1[B3],80)
		percentile_20_flux4 = np.nanpercentile(10**median_flux[B4],20,axis=0)*np.nanpercentile(F1[B4],20)
		percentile_80_flux4 = np.nanpercentile(10**median_flux[B4],80,axis=0)*np.nanpercentile(F1[B4],80)
		percentile_20_flux5 = np.nanpercentile(10**median_flux[B5],20,axis=0)*np.nanpercentile(F1[B5],20)
		percentile_80_flux5 = np.nanpercentile(10**median_flux[B5],80,axis=0)*np.nanpercentile(F1[B5],80)


		median_xray_f1 = np.nanmedian(y[B1],axis=0)*np.nanmedian(F1[B1])
		median_xray_f2 = np.nanmedian(y[B2],axis=0)*np.nanmedian(F1[B2])
		median_xray_f3 = np.nanmedian(y[B3],axis=0)*np.nanmedian(F1[B3])
		median_xray_f4 = np.nanmedian(y[B4],axis=0)*np.nanmedian(F1[B4])
		median_xray_f5 = np.nanmedian(y[B5],axis=0)*np.nanmedian(F1[B5])
		# median_xray_f_goals = np.nanmedian(goals_y,axis=0)*np.nanmean(goals_f1)

		percentile_20_xray_f1 = np.nanpercentile(y[B1],20,axis=0)*np.nanpercentile(F1[B1],20)
		percentile_20_xray_f2 = np.nanpercentile(y[B2],20,axis=0)*np.nanpercentile(F1[B2],20)
		percentile_20_xray_f3 = np.nanpercentile(y[B3],20,axis=0)*np.nanpercentile(F1[B3],20)
		percentile_20_xray_f4 = np.nanpercentile(y[B4],20,axis=0)*np.nanpercentile(F1[B4],20)
		percentile_20_xray_f5 = np.nanpercentile(y[B5],20,axis=0)*np.nanpercentile(F1[B5],20)
		# percentile_20_xray_f_goals = np.nanpercentile(goals_y,20,axis=0)*np.nanpercentile(goals_f1,20)

		percentile_80_xray_f1 = np.nanpercentile(y[B1],80,axis=0)*np.nanpercentile(F1[B1],80)
		percentile_80_xray_f2 = np.nanpercentile(y[B2],80,axis=0)*np.nanpercentile(F1[B2],80)
		percentile_80_xray_f3 = np.nanpercentile(y[B3],80,axis=0)*np.nanpercentile(F1[B3],80)
		percentile_80_xray_f4 = np.nanpercentile(y[B4],80,axis=0)*np.nanpercentile(F1[B4],80)
		percentile_80_xray_f5 = np.nanpercentile(y[B5],80,axis=0)*np.nanpercentile(F1[B5],80)
		# percentile_80_xray_f_goals = np.nanpercentile(goals_y,80,axis=0)*np.nanpercentile(goals_f1,80)

		median_xray_w = np.nanmedian(x,axis=0)
		# median_xray_w_goals = np.nanmedian(goals_x,axis=0)


		median_flux1 = np.nanmedian(10**median_flux[B1],axis=0)*np.nanmedian(F1[B1])
		median_flux2 = np.nanmedian(10**median_flux[B2],axis=0)*np.nanmedian(F1[B2])
		median_flux3 = np.nanmedian(10**median_flux[B3],axis=0)*np.nanmedian(F1[B3])
		median_flux4 = np.nanmedian(10**median_flux[B4],axis=0)*np.nanmedian(F1[B4])
		median_flux5 = np.nanmedian(10**median_flux[B5],axis=0)*np.nanmedian(F1[B5])
		median_flux_goals = np.nanmedian(10**goals_median_f,axis=0)*np.nanmean(goals_f1)

		median_xray_f1 = np.nanmedian(y[B1],axis=0)*np.nanmedian(F1[B1])
		median_xray_f2 = np.nanmedian(y[B2],axis=0)*np.nanmedian(F1[B2])
		median_xray_f3 = np.nanmedian(y[B3],axis=0)*np.nanmedian(F1[B3])
		median_xray_f4 = np.nanmedian(y[B4],axis=0)*np.nanmedian(F1[B4])
		median_xray_f5 = np.nanmedian(y[B5],axis=0)*np.nanmedian(F1[B5])
		median_xray_f_goals = np.nanmedian(goals_y,axis=0)*np.nanmean(goals_f1)

		# median_xray_w = np.nanmean(x,axis=0)
		median_xray_w_goals = np.nanmedian(goals_x,axis=0)


		# median_flux1 = np.append(median_flux1,np.nanmedian(emis3[B1]))
		# median_flux2 = np.append(median_flux2,np.nanmedian(emis3[B2]))
		# median_flux3 = np.append(median_flux3,np.nanmedian(emis3[B3]))
		# median_flux4 = np.append(median_flux4,np.nanmedian(emis3[B4]))
		# median_flux5 = np.append(median_flux5,np.nanmedian(emis3[B5]))

		median_wavelength_all = np.nanmedian(10**median_wavelength,axis=0)
		median_wavelength_goals = np.nanmedian(10**goals_median_w,axis=0)
		# median_wavelength_all = np.nanmean(10**median_wavelength,axis=0)
		# median_wavelength_goals = np.nanmean(10**goals_median_w,axis=0)

		# median_wavelength_all = np.append(median_wavelength_all,100)

		# median_flux1[9] = np.nan
		# median_flux2[9] = np.nan
		# median_flux3[9] = np.nan
		# median_flux4[9] = np.nan
		# median_flux5[9] = np.nan

		# median_flux1[10:46] = np.nan
		# median_flux2[10:46] = np.nan
		# median_flux3[10:46] = np.nan
		# median_flux4[10:46] = np.nan
		# median_flux5[10:46] = np.nan

		# print(median_flux1)
		# print(median_wavelength_all)

		# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
        #           '#f781bf', '#a65628', '#984ea3',
        #           '#999999', '#e41a1c', '#dede00']
		

		# fig = plt.figure(figsize=(16,8))
		fig, ax = plt.subplots(figsize=(12,8))

		plt.title('0.3 < z < 0.5')
		plt.plot(wave[wave > 0.1],temp_comb[wave > 0.1],color='#999999',lw=3.5,alpha=0.75,label='Galaxy SED')

		# plt.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0)*np.nanmedian(F1[B1]),color='b',lw=3.5,label='Panel 1')
		# plt.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0)*np.nanmedian(F1[B2]),color='purple',lw=3.5,label='Panel 2')
		# plt.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0)*np.nanmedian(F1[B3]),color='g',lw=3.5,label='Panel 3')
		# plt.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0)*np.nanmedian(F1[B4]),color='yellow',lw=3.5,label='Panel 4')
		# plt.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0)*np.nanmedian(F1[B5]),color='red',lw=3.5,label='Panel 5')
		# plt.plot(median_wavelength_goals,median_flux_goals,color='gray',lw=3.5,label='GOALS')

		plt.plot(median_wavelength_all,median_flux1,color='#377eb8',lw=3.5,label='Panel 1')
		# plt.plot(median_wavelength_all,percentile_80_flux1,color='b',ls='--',alpha=0.5)
		# plt.plot(median_wavelength_all,percentile_20_flux1,color='b',ls='--',alpha=0.5)
		# ax.fill_between(median_wavelength_all,percentile_80_flux1,percentile_20_flux1,color='b',alpha=0.4)

		plt.plot(median_wavelength_all,median_flux2,color='#984ea3',lw=3.5,label='Panel 2')
		# plt.plot(median_wavelength_all,percentile_80_flux2,color='purple',ls='--',alpha=0.5)
		# plt.plot(median_wavelength_all,percentile_20_flux2,color='purple',ls='--',alpha=0.5)
		# ax.fill_between(median_wavelength_all,percentile_80_flux2,percentile_20_flux2,color='purple',alpha=0.4)

		plt.plot(median_wavelength_all,median_flux3,color='#4daf4a',lw=3.5,label='Panel 3')
		# plt.plot(median_wavelength_all,percentile_80_flux3,color='g',ls='--',alpha=0.5)
		# plt.plot(median_wavelength_all,percentile_20_flux3,color='g',ls='--',alpha=0.5)
		# ax.fill_between(median_wavelength_all,percentile_80_flux3,percentile_20_flux3,color='g',alpha=0.4)

		plt.plot(median_wavelength_all,median_flux4,color='#ff7f00',lw=3.5,label='Panel 4')
		# plt.plot(median_wavelength_all,percentile_80_flux4,color='yellow',ls='--',alpha=0.5)
		# plt.plot(median_wavelength_all,percentile_20_flux4,color='yellow',ls='--',alpha=0.5)
		# ax.fill_between(median_wavelength_all,percentile_80_flux4,percentile_20_flux4,color='yellow',alpha=0.4)

		plt.plot(median_wavelength_all,median_flux5,color='#e41a1c',lw=3.5,label='Panel 5')
		# plt.plot(median_wavelength_all,percentile_80_flux5,color='red',ls='--',alpha=0.5)
		# plt.plot(median_wavelength_all,percentile_20_flux5,color='red',ls='--',alpha=0.5)
		# ax.fill_between(median_wavelength_all,percentile_80_flux5,percentile_20_flux5,color='red',alpha=0.4)

		# plt.plot(wave,E_L+E_Im+E_Sbc,color='#999999',lw=3.5,label='E+Im+Sbc')

		plt.plot(median_wavelength_goals,median_flux_goals,color='gray',lw=3.5,label='GOALS')



		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanmedian(emis3[B1],axis=0),marker='x',color='b',lw=3.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],20,axis=0),ls='--',color='b',alpha=0.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],80,axis=0),ls='--',color='b',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],80,axis=0),np.nanpercentile(emis3[B1],20,axis=0),color='b',alpha=0.4)

		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanmedian(emis3[B2],axis=0),marker='x',color='purple',lw=3.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],20,axis=0),ls='--',color='purple',alpha=0.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],80,axis=0),ls='--',color='purple',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],80,axis=0),np.nanpercentile(emis3[B2],20,axis=0),color='purple',alpha=0.4)

		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanmedian(emis3[B3],axis=0),marker='x',color='g',lw=3.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],20,axis=0),ls='--',color='g',alpha=0.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],80,axis=0),ls='--',color='g',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],80,axis=0),np.nanpercentile(emis3[B3],20,axis=0),color='g',alpha=0.4)

		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanmedian(emis3[B4],axis=0),marker='x',color='yellow',lw=3.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],20,axis=0),ls='--',color='yellow',alpha=0.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],80,axis=0),ls='--',color='yellow',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],80,axis=0),np.nanpercentile(emis3[B4],20,axis=0),color='yellow',alpha=0.4)

		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanmedian(emis3[B5],axis=0),marker='x',color='red',lw=3.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],20,axis=0),ls='--',color='red',alpha=0.5)
		# plt.plot(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],80,axis=0),ls='--',color='red',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],80,axis=0),np.nanpercentile(emis3[B5],20,axis=0),color='red',alpha=0.4)

		plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanmean(emis3[B1],axis=0),marker='v',ms=10,color='#377eb8',lw=3.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],20,axis=0),ls='--',color='b',alpha=0.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],80,axis=0),ls='--',color='b',alpha=0.5)
		# ax.fill_between(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B1],80,axis=0),np.nanpercentile(emis3[B1],20,axis=0),color='b',alpha=0.4)

		plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanmean(emis3[B2],axis=0),marker='v',ms=10,color='#984ea3',lw=3.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],20,axis=0),ls='--',color='purple',alpha=0.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],80,axis=0),ls='--',color='purple',alpha=0.5)
		# ax.fill_between(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B2],80,axis=0),np.nanpercentile(emis3[B2],20,axis=0),color='purple',alpha=0.4)

		plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanmean(emis3[B3],axis=0),marker='v',ms=10,color='#4daf4a',lw=3.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],20,axis=0),ls='--',color='g',alpha=0.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],80,axis=0),ls='--',color='g',alpha=0.5)
		# ax.fill_between(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B3],80,axis=0),np.nanpercentile(emis3[B3],20,axis=0),color='g',alpha=0.4)

		plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanmean(emis3[B4],axis=0),marker='v',ms=10,color='#ff7f00',lw=3.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],20,axis=0),ls='--',color='yellow',alpha=0.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],80,axis=0),ls='--',color='yellow',alpha=0.5)
		# ax.fill_between(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B4],80,axis=0),np.nanpercentile(emis3[B4],20,axis=0),color='yellow',alpha=0.4)

		plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanmean(emis3[B5],axis=0),marker='v',ms=10,color='#e41a1c',lw=3.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],20,axis=0),ls='--',color='red',alpha=0.5)
		# plt.plot(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],80,axis=0),ls='--',color='red',alpha=0.5)
		# ax.fill_between(np.nanmean(median_FIR_w,axis=0),np.nanpercentile(emis3[B5],80,axis=0),np.nanpercentile(emis3[B5],20,axis=0),color='red',alpha=0.4)




		plt.plot(median_xray_w[0:2],median_xray_f1[0:2],color='#377eb8',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f1[0:2],ls='--',color='b',alpha=0.5)
		# plt.plot(median_xray_w[0:2],percentile_80_xray_f1[0:2],ls='--',color='b',alpha=0.5)
		# ax.fill_between(median_xray_w[0:2],percentile_80_xray_f1[0:2],percentile_20_xray_f1[0:2],color='b',alpha=0.4)

		plt.plot(median_xray_w[0:2],median_xray_f2[0:2],color='#984ea3',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f2[0:2],ls='--',color='purple',alpha=0.5)
		# plt.plot(median_xray_w[0:2],percentile_80_xray_f2[0:2],ls='--',color='purple',alpha=0.5)
		# ax.fill_between(median_xray_w[0:2],percentile_80_xray_f2[0:2],percentile_20_xray_f2[0:2],color='purple',alpha=0.4)

		plt.plot(median_xray_w[0:2],median_xray_f3[0:2],color='#4daf4a',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f3[0:2],ls='--',color='g',alpha=0.5)
		# plt.plot(median_xray_w[0:2],percentile_80_xray_f3[0:2],ls='--',color='g',alpha=0.5)
		# ax.fill_between(median_xray_w[0:2],percentile_80_xray_f3[0:2],percentile_20_xray_f3[0:2],color='g',alpha=0.4)

		plt.plot(median_xray_w[0:2],median_xray_f4[0:2],color='#ff7f00',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f4[0:2],ls='--',color='yellow',alpha=0.5)
		# plt.plot(median_xray_w[0:2],percentile_80_xray_f4[0:2],ls='--',color='yellow',alpha=0.5)
		# ax.fill_between(median_xray_w[0:2],percentile_80_xray_f4[0:2],percentile_20_xray_f4[0:2],color='yellow',alpha=0.4)

		plt.plot(median_xray_w[0:2],median_xray_f5[0:2],color='#e41a1c',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f5[0:2],ls='--',color='red',alpha=0.5)
		# plt.plot(median_xray_w[0:2],percentile_80_xray_f5[0:2],ls='--',color='red',alpha=0.5)
		# ax.fill_between(median_xray_w[0:2],percentile_80_xray_f5[0:2],percentile_20_xray_f5[0:2],color='red',alpha=0.4)

		plt.plot(median_xray_w_goals[0:2],median_xray_f_goals[0:2],marker='x',color='gray',lw=3.5)
		# plt.plot(median_xray_w[0:2],percentile_20_xray_f_goals[0:2],ls='--',color='gray',alpha=0.5)
		# plt.plot(m/edian_xray_w[0:2],percentile_80_xray_f_goals[0:2],ls='--',color='gray',alpha=0.5)
		# ax.fill_between(np.nanmedian(median_xray_w[0:2],axis=0),percentile_80_xray_f_goals[0:2],percentile_20_xray_f_goals[0:2],color='gray',alpha=0.4)

		# print(median_xray_w[0:2],median_xray_f5[0:2])		
		# print(median_xray_w_goals[0:2],median_xray_f_goals[0:2])

		plt.xlim(8E-5,7E2)
		plt.ylim(3E42,1E46)
		plt.yscale('log')
		plt.xscale('log')
		plt.xlabel(r'Rest Wavelength [$\mu$m]')
		plt.ylabel(r'$\lambda$ L$_\lambda$ [erg/s]')
		plt.grid()
		plt.legend(fontsize=16)
		plt.savefig('/Users/connor_auge/Desktop/Paper/PlotMedian_03z05color.pdf')
		plt.show()


	def plot_median_zbins(self, emis3, x, y, L, spec_type, f1, f2, f3, f4, median_wavelength, median_flux, median_wavelength_ext=None, median_flux_ext=None, F1=None, F2=None, median_xray_w=None, median_xray_f=None, median_FIR_w=None, spec_z=None, uv_slope=None, mir_slope1=None, mir_slope2=None,goals_x=None,goals_y=None,goals_median_w=None,goals_median_f=None,goals_f1=None,single_x=None,single_y=None,single_one=None,single_id=None):

		# inf = ascii.read('/Users/connor_auge/Desktop/A10_templates.txt')

		# wave = np.asarray(inf['Wave'])
		# wave_cgs = wave*1E-4
		# freq = 3E10/wave_cgs
		# E_flux = np.asarray(inf['E'])*1E-16 # erg/s/Hz
		# E_nuFnu = E_flux*freq
		# Im_flux = np.asarray(inf['Im'])*1E-15 # erg/s/Hz
		# Im_nuFnu = Im_flux*freq
		# Sbc_flux = np.asarray(inf['Sbc'])*1E-17 # erg/s/Hz
		# Sbc_nuFnu = Sbc_flux*freq
		# AGN1_flux = np.asarray(inf['AGN'])*1E-11 # erg/s/Hz
		# AGN1_nuFnu = AGN1_flux*freq
		# AGN2_flux = np.asarray(inf['AGN2'])*1E-14 # erg/s/Hz
		# AGN2_nuFnu = AGN2_flux*freq
		# E_Im_nuFnu = Im_nuFnu+E_nuFnu
		# E_L = E_nuFnu*4*np.pi*3E19**2
		# E_AGN1 = AGN1_nuFnu*4*np.pi*3E19**2
		# E_AGN2 = AGN2_nuFnu*4*np.pi*3E19**2
		# E_Sbc = Sbc_nuFnu*4*np.pi*3E19**2
		# E_Im = Im_nuFnu*4*np.pi*3E19**2
		# temp_comb = E_L+E_Sbc+E_Im

		# for i in range(len(emis3)):
			# emis3[i][emis3[i] < 1E40] = np.nan

		norm = np.asarray(F1)
		mark = np.asarray(F2)

		cosmos_s82x_list = []
		cosmos_s82x_wave = []
		for i in range(len(y)):

			upper_wave = np.array([2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq


			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i]))
				else:
					cosmos_s82x_list.append(y[i][-3:])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i]))
				else:
					a = np.array([y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i]))
				else:
					cosmos_s82x_list.append(y[i][-3:])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i]))
				else:
					a = np.array([y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
			cosmos_s82x_wave.append(rest_upper_w_microns)

		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)

		if single_id is None:
			single_id = 'None'


		clim1 = 42.5
		clim2 = 46

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		print('One Micron Luminosity: ')
		print('0 < z < 0.6: ',np.nanmedian(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]]),np.nanmedian(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]]),np.nanmedian(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]]),np.nanmedian(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]]),np.nanmedian(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]]))
		print('0.6 < z < 0.9: ',np.nanmedian(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]]),np.nanmedian(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]]),np.nanmedian(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]]),np.nanmedian(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]]),np.nanmedian(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]]))
		print('0.9 < z < 1.2: ',np.nanmedian(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]]),np.nanmedian(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]]),np.nanmedian(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]]),np.nanmedian(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]]),np.nanmedian(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]]))


		# temp1 = temp_comb[wave == 1.0050]
		# # norm_temp = np.nanmedian(F1[B4])/temp1
		# norm_temp = 4.215928239976833e+44/temp1
		# temp_comb *= norm_temp


		# median_flux1_z1 = np.nanmean(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)*np.nanmean(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]])
		# median_flux2_z1 = np.nanmean(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)*np.nanmean(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]])
		# median_flux3_z1 = np.nanmean(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)*np.nanmean(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]])
		# median_flux4_z1 = np.nanmean(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)*np.nanmean(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])
		# median_flux5_z1 = np.nanmean(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)*np.nanmean(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]])

		# median_flux1_z2 = np.nanmean(10**median_flux[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)*np.nanmean(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]])
		# median_flux2_z2 = np.nanmean(10**median_flux[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)*np.nanmean(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]])
		# median_flux3_z2 = np.nanmean(10**median_flux[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)*np.nanmean(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]])
		# median_flux4_z2 = np.nanmean(10**median_flux[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)*np.nanmean(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]])
		# median_flux5_z2 = np.nanmean(10**median_flux[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)*np.nanmean(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]])

		# median_flux1_z3 = np.nanmean(10**median_flux[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)*np.nanmean(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]])
		# median_flux2_z3 = np.nanmean(10**median_flux[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)*np.nanmean(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]])
		# median_flux3_z3 = np.nanmean(10**median_flux[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)*np.nanmean(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]])
		# median_flux4_z3 = np.nanmean(10**median_flux[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)*np.nanmean(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]])
		# median_flux5_z3 = np.nanmean(10**median_flux[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)*np.nanmean(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]])

		median_flux1_z1 = np.nanmedian(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]])
		median_flux2_z1 = np.nanmedian(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]])
		median_flux3_z1 = np.nanmedian(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]])
		median_flux4_z1 = np.nanmedian(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])
		median_flux5_z1 = np.nanmedian(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]])

		median_flux1_z2 = np.nanmedian(10**median_flux[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]])
		median_flux2_z2 = np.nanmedian(10**median_flux[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]])
		median_flux3_z2 = np.nanmedian(10**median_flux[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]])
		median_flux4_z2 = np.nanmedian(10**median_flux[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]])
		median_flux5_z2 = np.nanmedian(10**median_flux[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]])

		median_flux1_z3 = np.nanmedian(10**median_flux[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]])
		median_flux2_z3 = np.nanmedian(10**median_flux[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]])
		median_flux3_z3 = np.nanmedian(10**median_flux[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]])
		median_flux4_z3 = np.nanmedian(10**median_flux[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]])
		median_flux5_z3 = np.nanmedian(10**median_flux[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]])

		percentile_20_flux1_z1 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],20)
		percentile_80_flux1_z1 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],80)
		percentile_20_flux2_z1 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],20)
		percentile_80_flux2_z1 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],80)
		percentile_20_flux3_z1 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],20)
		percentile_80_flux3_z1 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],80)
		percentile_20_flux4_z1 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],20)
		percentile_80_flux4_z1 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],80)
		percentile_20_flux5_z1 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],20)
		percentile_80_flux5_z1 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],80)

		percentile_20_flux1_z2 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],20)
		percentile_80_flux1_z2 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],80)
		percentile_20_flux2_z2 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],20)
		percentile_80_flux2_z2 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],80)
		percentile_20_flux3_z2 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],20)
		percentile_80_flux3_z2 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],80)
		percentile_20_flux4_z2 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],20)
		percentile_80_flux4_z2 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],80)
		percentile_20_flux5_z2 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],20)
		percentile_80_flux5_z2 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],80)

		percentile_20_flux1_z3 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],20)
		percentile_80_flux1_z3 = np.nanpercentile(10**median_flux[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],80)
		percentile_20_flux2_z3 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],20)
		percentile_80_flux2_z3 = np.nanpercentile(10**median_flux[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],80)
		percentile_20_flux3_z3 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],20)
		percentile_80_flux3_z3 = np.nanpercentile(10**median_flux[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],80)
		percentile_20_flux4_z3 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],20)
		percentile_80_flux4_z3 = np.nanpercentile(10**median_flux[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],80)
		percentile_20_flux5_z3 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],20)
		percentile_80_flux5_z3 = np.nanpercentile(10**median_flux[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],80)

		# percentile_20_flux1_z1 = median_flux1_z1 - np.std(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)*np.std(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]])*3
		# percentile_80_flux1_z1 = median_flux1_z1 + np.std(10**median_flux[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)*np.std(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]])*3
		# percentile_20_flux2_z1 = median_flux2_z1 - np.std(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)*np.std(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]])*3
		# percentile_80_flux2_z1 = median_flux2_z1 + np.std(10**median_flux[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)*np.std(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]])*3
		# percentile_20_flux3_z1 = median_flux3_z1 - np.std(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)*np.std(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]])*3
		# percentile_80_flux3_z1 = median_flux3_z1 + np.std(10**median_flux[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)*np.std(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]])*3
		# percentile_20_flux4_z1 = median_flux4_z1 - np.std(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)*np.std(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])*3
		# percentile_80_flux4_z1 = median_flux4_z1 + np.std(10**median_flux[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)*np.std(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])*3
		# percentile_20_flux5_z1 = median_flux5_z1 - np.std(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)*np.std(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]])*3
		# percentile_80_flux5_z1 = median_flux5_z1 + np.std(10**median_flux[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)*np.std(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]])*3

		# percentile_20_flux1_z2 = median_flux1_z2 - np.std(10**median_flux[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],axis=0)*np.std(F1[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]])*3
		# percentile_80_flux1_z2 = median_flux1_z2 + np.std(10**median_flux[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]],axis=0)*np.std(F1[B1[(z1 <= zlim_3)&(z1 >= zlim_2)]])*3
		# percentile_20_flux2_z2 = median_flux2_z2 - np.std(10**median_flux[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],axis=0)*np.std(F1[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]])*3
		# percentile_80_flux2_z2 = median_flux2_z2 + np.std(10**median_flux[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]],axis=0)*np.std(F1[B2[(z2 <= zlim_3)&(z2 >= zlim_2)]])*3
		# percentile_20_flux3_z2 = median_flux3_z2 - np.std(10**median_flux[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],axis=0)*np.std(F1[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]])*3
		# percentile_80_flux3_z2 = median_flux3_z2 + np.std(10**median_flux[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]],axis=0)*np.std(F1[B3[(z3 <= zlim_3)&(z3 >= zlim_2)]])*3
		# percentile_20_flux4_z2 = median_flux4_z2 - np.std(10**median_flux[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],axis=0)*np.std(F1[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]])*3
		# percentile_80_flux4_z2 = median_flux4_z2 + np.std(10**median_flux[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]],axis=0)*np.std(F1[B4[(z4 <= zlim_3)&(z4 >= zlim_2)]])*3
		# percentile_20_flux5_z2 = median_flux5_z2 - np.std(10**median_flux[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],axis=0)*np.std(F1[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]])*3
		# percentile_80_flux5_z2 = median_flux5_z2 + np.std(10**median_flux[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]],axis=0)*np.std(F1[B5[(z5 <= zlim_3)&(z5 >= zlim_2)]])*3

		# percentile_20_flux1_z3 = median_flux1_z3 - np.std(10**median_flux[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],axis=0)*np.std(F1[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]])*3
		# percentile_80_flux1_z3 = median_flux1_z3 + np.std(10**median_flux[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]],axis=0)*np.std(F1[B1[(z1 <= zlim_4)&(z1 >= zlim_3)]])*3
		# percentile_20_flux2_z3 = median_flux2_z3 - np.std(10**median_flux[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],axis=0)*np.std(F1[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]])*3
		# percentile_80_flux2_z3 = median_flux2_z3 + np.std(10**median_flux[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]],axis=0)*np.std(F1[B2[(z2 <= zlim_4)&(z2 >= zlim_3)]])*3
		# percentile_20_flux3_z3 = median_flux3_z3 - np.std(10**median_flux[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],axis=0)*np.std(F1[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]])*3
		# percentile_80_flux3_z3 = median_flux3_z3 + np.std(10**median_flux[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]],axis=0)*np.std(F1[B3[(z3 <= zlim_4)&(z3 >= zlim_3)]])*3
		# percentile_20_flux4_z3 = median_flux4_z3 - np.std(10**median_flux[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],axis=0)*np.std(F1[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]])*3
		# percentile_80_flux4_z3 = median_flux4_z3 + np.std(10**median_flux[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]],axis=0)*np.std(F1[B4[(z4 <= zlim_4)&(z4 >= zlim_3)]])*3
		# percentile_20_flux5_z3 = median_flux5_z3 - np.std(10**median_flux[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],axis=0)*np.std(F1[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]])*3
		# percentile_80_flux5_z3 = median_flux5_z3 + np.std(10**median_flux[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]],axis=0)*np.std(F1[B5[(z5 <= zlim_4)&(z5 >= zlim_3)]])*3





		median_xray_f1_z1 = np.nanmedian(y[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]])
		median_xray_f2_z1 = np.nanmedian(y[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]])
		median_xray_f3_z1 = np.nanmedian(y[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]])
		median_xray_f4_z1 = np.nanmedian(y[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])
		median_xray_f5_z1 = np.nanmedian(y[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]])

		median_xray_f1_z2 = np.nanmedian(y[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]])
		median_xray_f2_z2 = np.nanmedian(y[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]])
		median_xray_f3_z2 = np.nanmedian(y[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]])
		median_xray_f4_z2 = np.nanmedian(y[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]])
		median_xray_f5_z2 = np.nanmedian(y[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]])

		median_xray_f1_z3 = np.nanmedian(y[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)*np.nanmedian(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]])
		median_xray_f2_z3 = np.nanmedian(y[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)*np.nanmedian(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]])
		median_xray_f3_z3 = np.nanmedian(y[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)*np.nanmedian(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]])
		median_xray_f4_z3 = np.nanmedian(y[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)*np.nanmedian(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]])
		median_xray_f5_z3 = np.nanmedian(y[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)*np.nanmedian(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]])

		percentile_20_xray_f1_z1 = np.nanpercentile(y[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],20)
		percentile_80_xray_f1_z1 = np.nanpercentile(y[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],80)
		percentile_20_xray_f2_z1 = np.nanpercentile(y[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],20)
		percentile_80_xray_f2_z1 = np.nanpercentile(y[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],80)
		percentile_20_xray_f3_z1 = np.nanpercentile(y[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],20)
		percentile_80_xray_f3_z1 = np.nanpercentile(y[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],80)
		percentile_20_xray_f4_z1 = np.nanpercentile(y[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],20)
		percentile_80_xray_f4_z1 = np.nanpercentile(y[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],80)
		percentile_20_xray_f5_z1 = np.nanpercentile(y[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],20)
		percentile_80_xray_f5_z1 = np.nanpercentile(y[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],80)

		percentile_20_xray_f1_z2 = np.nanpercentile(y[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],20)
		percentile_80_xray_f1_z2 = np.nanpercentile(y[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],80)
		percentile_20_xray_f2_z2 = np.nanpercentile(y[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],20)
		percentile_80_xray_f2_z2 = np.nanpercentile(y[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],80)
		percentile_20_xray_f3_z2 = np.nanpercentile(y[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],20)
		percentile_80_xray_f3_z2 = np.nanpercentile(y[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],80)
		percentile_20_xray_f4_z2 = np.nanpercentile(y[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],20)
		percentile_80_xray_f4_z2 = np.nanpercentile(y[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],80)
		percentile_20_xray_f5_z2 = np.nanpercentile(y[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],20)
		percentile_80_xray_f5_z2 = np.nanpercentile(y[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],80)

		percentile_20_xray_f1_z3 = np.nanpercentile(y[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],20,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],20)
		percentile_80_xray_f1_z3 = np.nanpercentile(y[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],80,axis=0)*np.nanpercentile(F1[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],80)
		percentile_20_xray_f2_z3 = np.nanpercentile(y[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],20,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],20)
		percentile_80_xray_f2_z3 = np.nanpercentile(y[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],80,axis=0)*np.nanpercentile(F1[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],80)
		percentile_20_xray_f3_z3 = np.nanpercentile(y[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],20,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],20)
		percentile_80_xray_f3_z3 = np.nanpercentile(y[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],80,axis=0)*np.nanpercentile(F1[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],80)
		percentile_20_xray_f4_z3 = np.nanpercentile(y[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],20,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],20)
		percentile_80_xray_f4_z3 = np.nanpercentile(y[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],80,axis=0)*np.nanpercentile(F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],80)
		percentile_20_xray_f5_z3 = np.nanpercentile(y[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],20,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],20)
		percentile_80_xray_f5_z3 = np.nanpercentile(y[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],80,axis=0)*np.nanpercentile(F1[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],80)




		median_FIR_f1_z1 = np.nanmedian(emis3[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)
		median_FIR_f2_z1 = np.nanmedian(emis3[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0) 
		median_FIR_f3_z1 = np.nanmedian(emis3[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)
		median_FIR_f4_z1 = np.nanmedian(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)
		median_FIR_f5_z1 = np.nanmedian(emis3[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)

		median_FIR_f1_z2 = np.nanmedian(emis3[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)
		median_FIR_f2_z2 = np.nanmedian(emis3[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)
		median_FIR_f3_z2 = np.nanmedian(emis3[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)
		median_FIR_f4_z2 = np.nanmedian(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)
		median_FIR_f5_z2 = np.nanmedian(emis3[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)

		median_FIR_f1_z3 = np.nanmedian(emis3[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)
		median_FIR_f2_z3 = np.nanmedian(emis3[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)
		median_FIR_f3_z3 = np.nanmedian(emis3[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)
		median_FIR_f4_z3 = np.nanmedian(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)
		median_FIR_f5_z3 = np.nanmedian(emis3[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)

		# percentile_20_FIR_f1_z1 = median_FIR_f1_z1 - np.nanstd(emis3[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0)
		# percentile_80_FIR_f1_z1 = median_FIR_f1_z1 + np.nanstd(emis3[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],axis=0) 
		# percentile_20_FIR_f2_z1 = median_FIR_f2_z1 - np.nanstd(emis3[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0)
		# percentile_80_FIR_f2_z1 = median_FIR_f2_z1 + np.nanstd(emis3[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],axis=0) 
		# percentile_20_FIR_f3_z1 = median_FIR_f3_z1 - np.nanstd(emis3[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0)
		# percentile_80_FIR_f3_z1 = median_FIR_f3_z1 + np.nanstd(emis3[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],axis=0) 
		# percentile_20_FIR_f4_z1 = median_FIR_f4_z1 - np.nanstd(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0)
		# percentile_80_FIR_f4_z1 = median_FIR_f4_z1 + np.nanstd(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0) 
		# percentile_20_FIR_f5_z1 = median_FIR_f5_z1 - np.nanstd(emis3[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0)
		# percentile_80_FIR_f5_z1 = median_FIR_f5_z1 + np.nanstd(emis3[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],axis=0) 

		# percentile_20_FIR_f1_z2 = np.nanstd(emis3[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)*3  
		# percentile_80_FIR_f1_z2 = np.nanstd(emis3[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0)*3 
		# percentile_20_FIR_f2_z2 = np.nanstd(emis3[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)*3  
		# percentile_80_FIR_f2_z2 = np.nanstd(emis3[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0)*3 
		# percentile_20_FIR_f3_z2 = np.nanstd(emis3[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)*3  
		# percentile_80_FIR_f3_z2 = np.nanstd(emis3[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0)*3 
		# percentile_20_FIR_f4_z2 = np.nanstd(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)*3  
		# percentile_80_FIR_f4_z2 = np.nanstd(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0)*3 
		# percentile_20_FIR_f5_z2 = np.nanstd(emis3[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)*3  
		# percentile_80_FIR_f5_z2 = np.nanstd(emis3[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0)*3 

		# percentile_20_FIR_f1_z3 = np.nanstd(emis3[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)*3  
		# percentile_80_FIR_f1_z3 = np.nanstd(emis3[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0)*3 
		# percentile_20_FIR_f2_z3 = np.nanstd(emis3[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)*3  
		# percentile_80_FIR_f2_z3 = np.nanstd(emis3[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0)*3 
		# percentile_20_FIR_f3_z3 = np.nanstd(emis3[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)*3  
		# percentile_80_FIR_f3_z3 = np.nanstd(emis3[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0)*3 
		# percentile_20_FIR_f4_z3 = np.nanstd(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)*3  
		# percentile_80_FIR_f4_z3 = np.nanstd(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0)*3 
		# percentile_20_FIR_f5_z3 = np.nanstd(emis3[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)*3  
		# percentile_80_FIR_f5_z3 = np.nanstd(emis3[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0)*3 

		percentile_20_FIR_f1_z1 = np.percentile(emis3[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],20,axis=0)  
		percentile_80_FIR_f1_z1 = np.percentile(emis3[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],80,axis=0) 
		percentile_20_FIR_f2_z1 = np.percentile(emis3[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],20,axis=0)  
		percentile_80_FIR_f2_z1 = np.percentile(emis3[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],80,axis=0) 
		percentile_20_FIR_f3_z1 = np.percentile(emis3[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],20,axis=0)  
		percentile_80_FIR_f3_z1 = np.percentile(emis3[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],80,axis=0) 
		percentile_20_FIR_f4_z1 = np.percentile(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],20,axis=0)  
		percentile_80_FIR_f4_z1 = np.percentile(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],80,axis=0) 
		percentile_20_FIR_f5_z1 = np.percentile(emis3[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],20,axis=0)
		percentile_80_FIR_f5_z1 = np.percentile(emis3[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]],80,axis=0) 

		percentile_20_FIR_f1_z2 = np.percentile(emis3[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],20,axis=0)  
		percentile_80_FIR_f1_z2 = np.percentile(emis3[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],80,axis=0) 
		percentile_20_FIR_f2_z2 = np.percentile(emis3[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],20,axis=0)  
		percentile_80_FIR_f2_z2 = np.percentile(emis3[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],80,axis=0) 
		percentile_20_FIR_f3_z2 = np.percentile(emis3[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],20,axis=0)  
		percentile_80_FIR_f3_z2 = np.percentile(emis3[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],80,axis=0) 
		percentile_20_FIR_f4_z2 = np.percentile(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],20,axis=0)  
		percentile_80_FIR_f4_z2 = np.percentile(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],80,axis=0) 
		percentile_20_FIR_f5_z2 = np.percentile(emis3[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],20,axis=0)  
		percentile_80_FIR_f5_z2 = np.percentile(emis3[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],80,axis=0) 

		percentile_20_FIR_f1_z3 = np.percentile(emis3[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],20,axis=0)  
		percentile_80_FIR_f1_z3 = np.percentile(emis3[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],80,axis=0) 
		percentile_20_FIR_f2_z3 = np.percentile(emis3[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],20,axis=0)  
		percentile_80_FIR_f2_z3 = np.percentile(emis3[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],80,axis=0) 
		percentile_20_FIR_f3_z3 = np.percentile(emis3[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],20,axis=0)  
		percentile_80_FIR_f3_z3 = np.percentile(emis3[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],80,axis=0) 
		percentile_20_FIR_f4_z3 = np.percentile(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],20,axis=0)  
		percentile_80_FIR_f4_z3 = np.percentile(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],80,axis=0) 
		percentile_20_FIR_f5_z3 = np.percentile(emis3[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],20,axis=0)  
		percentile_80_FIR_f5_z3 = np.percentile(emis3[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],80,axis=0) 

		# print(percentile_80_FIR_f4_z1)
		# print('Data: ',emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]])
		# print('STD: ',np.std(emis3[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],axis=0))
		# print('Mean: ',median_FIR_f4_z1)
		# print(percentile_20_FIR_f4_z1)

		# print(percentile_80_FIR_f4_z2)
		# print(median_FIR_f4_z2)
		# print(percentile_20_FIR_f4_z2)
		
		# print('Check FIR bin 4 z 3:',emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]])
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],80,axis=0))
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],20,axis=0))

		# print('Check FIR bin 4 z 2:',emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]])
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],80,axis=0))
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],20,axis=0))

		# print('Check FIR bin 4 z 1:',emis3[B4[(z4 <= zlim_2)&(z4 > zlim_1)]])
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],80,axis=0))
		# print(np.nanpercentile(emis3[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],20,axis=0))

		# for i in range(len(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]])):
			# print(emis3[B4[(z4 <= zlim_4)&(z4 > zlim_3)]][i],F1[B4[(z4 <= zlim_4)&(z4 > zlim_3)]][i],mark[B4[(z4 <= zlim_4)&(z4 > zlim_3)][i]])




		median_xray_w_z1 = np.nanmedian(x[(spec_z <= zlim_2)&(spec_z >= zlim_1)],axis=0)
		median_xray_w_z2 = np.nanmedian(x[(spec_z <= zlim_3)&(spec_z >= zlim_2)],axis=0)
		median_xray_w_z3 = np.nanmedian(x[(spec_z <= zlim_4)&(spec_z >= zlim_3)],axis=0)


		median_wavelength_z1 = np.nanmedian(10**median_wavelength[(spec_z <= zlim_2)&(spec_z >= zlim_1)],axis=0)
		median_wavelength_z2 = np.nanmedian(10**median_wavelength[(spec_z <= zlim_3)&(spec_z >= zlim_2)],axis=0)
		median_wavelength_z3 = np.nanmedian(10**median_wavelength[(spec_z <= zlim_4)&(spec_z >= zlim_3)],axis=0)

		median_FIR_wavelength_z1 = np.nanmedian(median_FIR_w[(spec_z <= zlim_2)&(spec_z >= zlim_1)],axis=0) 
		median_FIR_wavelength_z2 = np.nanmedian(median_FIR_w[(spec_z <= zlim_3)&(spec_z >= zlim_2)],axis=0) 
		median_FIR_wavelength_z3 = np.nanmedian(median_FIR_w[(spec_z <= zlim_3)&(spec_z >= zlim_2)],axis=0)  




		# median_flux_goals = np.nanmean(10**goals_median_f,axis=0)*np.nanmean(goals_f1)
		# median_wavelength_goals = np.nanmean(10**goals_median_w,axis=0)
		# median_xray_w_goals = np.nanmean(goals_x,axis=0)
		# median_xray_f_goals = np.nanmean(goals_y, axis=0)*np.nanmean(goals_f1)


		yticks = [42,43,44,45,46]
		xticks = [1E-4,1E-3,1E-2,1E-1,1E0,1E1,1E2]
		ytick_labels = ['42','43','44','45','46']
		xticks_labels = [r'10$^{-4}$','',r'10$^{-2}$','',r'10$^{0}$','',r'10$^{2}$']


		def solar(x):
			return x/3.8E33

		def ergs(x):
			return x*3.8E33

		# GOALS_Y = np.asarray([goals_y[i]*goals_f1[i] for i in range(len(goals_f1))])




		fig = plt.figure(figsize=(21,6))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.08) # set the spacing between axes
		gs.update(left=0.06,right=0.94,top=0.9,bottom=0.15)

		ax1 = plt.subplot(gs[0])
		ax1.plot(median_wavelength_z1, percentile_20_flux1_z1, ls='--', color='#377eb8', lw=1)
		ax1.plot(median_wavelength_z1, percentile_80_flux1_z1, ls='--', color='#377eb8', lw=1)
		ax1.fill_between(median_wavelength_z1,percentile_20_flux1_z1,percentile_80_flux1_z1, color='#377eb8',alpha=0.15)

		ax1.plot(median_wavelength_z1, percentile_20_flux2_z1, ls='--', color='#984ea3', lw=1)
		ax1.plot(median_wavelength_z1, percentile_80_flux2_z1, ls='--', color='#984ea3', lw=1)
		ax1.fill_between(median_wavelength_z1,percentile_20_flux2_z1,percentile_80_flux2_z1, color='#984ea3',alpha=0.15)

		ax1.plot(median_wavelength_z1, percentile_20_flux3_z1, ls='--', color='#4daf4a', lw=1)
		ax1.plot(median_wavelength_z1, percentile_80_flux3_z1, ls='--', color='#4daf4a', lw=1)
		ax1.fill_between(median_wavelength_z1,percentile_20_flux3_z1,percentile_80_flux3_z1, color='#4daf4a',alpha=0.15)

		ax1.plot(median_wavelength_z1, percentile_20_flux4_z1, ls='--', color='#ff7f00', lw=1)
		ax1.plot(median_wavelength_z1, percentile_80_flux4_z1, ls='--', color='#ff7f00', lw=1)
		ax1.fill_between(median_wavelength_z1,percentile_20_flux4_z1,percentile_80_flux4_z1, color='#ff7f00',alpha=0.15)

		ax1.plot(median_wavelength_z1, percentile_20_flux5_z1, ls='--', color='#e41a1c', lw=1)
		ax1.plot(median_wavelength_z1, percentile_80_flux5_z1, ls='--', color='#e41a1c', lw=1)
		ax1.fill_between(median_wavelength_z1,percentile_20_flux5_z1,percentile_80_flux5_z1, color='#e41a1c',alpha=0.15)

		ax1.plot(median_wavelength_z1, median_flux1_z1, color='k', lw=4)
		ax1.plot(median_wavelength_z1, median_flux1_z1, color='#377eb8', lw=3.5)
		ax1.plot(median_wavelength_z1, median_flux2_z1, color='k', lw=4)
		ax1.plot(median_wavelength_z1, median_flux2_z1, color='#984ea3', lw=3.5)
		ax1.plot(median_wavelength_z1, median_flux3_z1, color='k', lw=4)
		ax1.plot(median_wavelength_z1, median_flux3_z1, color='#4daf4a', lw=3.5)
		ax1.plot(median_wavelength_z1, median_flux4_z1, color='k', lw=4)
		ax1.plot(median_wavelength_z1, median_flux4_z1, color='#ff7f00', lw=3.5)
		ax1.plot(median_wavelength_z1, median_flux5_z1, color='k', lw=4)
		ax1.plot(median_wavelength_z1, median_flux5_z1, color='#e41a1c', lw=3.5)

		# ax1.plot(median_wavelength_goals,median_flux_goals,color='gray',lw=3.5,label='ULIRGS')
		# ax1.plot(10**single_x[~np.isnan(single_y)],10**single_y[~np.isnan(single_y)]*single_one,color='k',lw=2.0,label='Arp 220')
		# lc = self.multilines(goals_x,goals_y,goals_f1)
		# ax1.legend(fontsize=16)


		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_20_FIR_f1_z1[-3:], ls='--', color='#377eb8', lw=1)
		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_80_FIR_f1_z1[-3:], ls='--', color='#377eb8', lw=1)
		ax1.fill_between(median_FIR_wavelength_z1[-3:],percentile_20_FIR_f1_z1[-3:],percentile_80_FIR_f1_z1[-3:], color='#377eb8',alpha=0.15)

		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_20_FIR_f2_z1[-3:], ls='--', color='#984ea3', lw=1)
		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_80_FIR_f2_z1[-3:], ls='--', color='#984ea3', lw=1)
		ax1.fill_between(median_FIR_wavelength_z1[-3:],percentile_20_FIR_f2_z1[-3:],percentile_80_FIR_f2_z1[-3:], color='#984ea3',alpha=0.15)

		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_20_FIR_f3_z1[-3:], ls='--', color='#4daf4a', lw=1)
		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_80_FIR_f3_z1[-3:], ls='--', color='#4daf4a', lw=1)
		ax1.fill_between(median_FIR_wavelength_z1[-3:],percentile_20_FIR_f3_z1[-3:],percentile_80_FIR_f3_z1[-3:], color='#4daf4a',alpha=0.15)

		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_20_FIR_f4_z1[-3:], ls='--', color='#ff7f00', lw=1)
		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_80_FIR_f4_z1[-3:], ls='--', color='#ff7f00', lw=1)
		ax1.fill_between(median_FIR_wavelength_z1[-3:],percentile_20_FIR_f4_z1[-3:],percentile_80_FIR_f4_z1[-3:], color='#ff7f00',alpha=0.15)

		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_20_FIR_f5_z1[-3:], ls='--', color='#e41a1c', lw=1)
		ax1.plot(median_FIR_wavelength_z1[-3:], percentile_80_FIR_f5_z1[-3:], ls='--', color='#e41a1c', lw=1)
		ax1.fill_between(median_FIR_wavelength_z1[-3:],percentile_20_FIR_f5_z1[-3:],percentile_80_FIR_f5_z1[-3:], color='#e41a1c',alpha=0.15)		

		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f1_z1[-3:],marker='v',ms=12,color='k',lw=4)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f1_z1[-3:],marker='v',ms=10,color='#377eb8',lw=3.5)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f2_z1[-3:],marker='v',ms=12,color='k',lw=4)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f2_z1[-3:],marker='v',ms=10,color='#984ea3',lw=3.5)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f3_z1[-3:],marker='v',ms=12,color='k',lw=4)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f3_z1[-3:],marker='v',ms=10,color='#4daf4a',lw=3.5)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f4_z1[-3:],marker='v',ms=12,color='k',lw=4)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f4_z1[-3:],marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f5_z1[-3:],marker='v',ms=12,color='k',lw=4)
		ax1.plot(median_FIR_wavelength_z1[-3:],median_FIR_f5_z1[-3:],marker='v',ms=10,color='#e41a1c',lw=3.5)


		ax1.plot(median_xray_w_z1[0:2], percentile_20_xray_f1_z1[0:2], ls='--', color='#377eb8', lw=1)
		ax1.plot(median_xray_w_z1[0:2], percentile_80_xray_f1_z1[0:2], ls='--', color='#377eb8', lw=1)
		ax1.fill_between(median_xray_w_z1[0:2],percentile_20_xray_f1_z1[0:2],percentile_80_xray_f1_z1[0:2], color='#377eb8',alpha=0.15)

		ax1.plot(median_xray_w_z1[0:2], percentile_20_xray_f2_z1[0:2], ls='--', color='#984ea3', lw=1)
		ax1.plot(median_xray_w_z1[0:2], percentile_80_xray_f2_z1[0:2], ls='--', color='#984ea3', lw=1)
		ax1.fill_between(median_xray_w_z1[0:2],percentile_20_xray_f2_z1[0:2],percentile_80_xray_f2_z1[0:2], color='#984ea3',alpha=0.15)

		ax1.plot(median_xray_w_z1[0:2], percentile_20_xray_f3_z1[0:2], ls='--', color='#4daf4a', lw=1)
		ax1.plot(median_xray_w_z1[0:2], percentile_80_xray_f3_z1[0:2], ls='--', color='#4daf4a', lw=1)
		ax1.fill_between(median_xray_w_z1[0:2],percentile_20_xray_f3_z1[0:2],percentile_80_xray_f3_z1[0:2], color='#4daf4a',alpha=0.15)

		ax1.plot(median_xray_w_z1[0:2], percentile_20_xray_f4_z1[0:2], ls='--', color='#ff7f00', lw=1)
		ax1.plot(median_xray_w_z1[0:2], percentile_80_xray_f4_z1[0:2], ls='--', color='#ff7f00', lw=1)
		ax1.fill_between(median_xray_w_z1[0:2],percentile_20_xray_f4_z1[0:2],percentile_80_xray_f4_z1[0:2], color='#ff7f00',alpha=0.15)

		ax1.plot(median_xray_w_z1[0:2], percentile_20_xray_f5_z1[0:2], ls='--', color='#e41a1c', lw=1)
		ax1.plot(median_xray_w_z1[0:2], percentile_80_xray_f5_z1[0:2], ls='--', color='#e41a1c', lw=1)
		ax1.fill_between(median_xray_w_z1[0:2],percentile_20_xray_f5_z1[0:2],percentile_80_xray_f5_z1[0:2], color='#e41a1c',alpha=0.15)

		ax1.plot(median_xray_w_z1[0:2],median_xray_f1_z1[0:2],color='k',lw=4)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f1_z1[0:2],color='#377eb8',lw=3.5)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f2_z1[0:2],color='k',lw=4)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f2_z1[0:2],color='#984ea3',lw=3.5)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f3_z1[0:2],color='k',lw=4)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f3_z1[0:2],color='#4daf4a',lw=3.5)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f4_z1[0:2],color='k',lw=4)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f4_z1[0:2],color='#ff7f00',lw=3.5)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f5_z1[0:2],color='k',lw=4)
		ax1.plot(median_xray_w_z1[0:2],median_xray_f5_z1[0:2],color='#e41a1c',lw=3.5)

		# ax1.plot(median_xray_w_goals[0:2],median_xray_f_goals[0:2],marker='x',color='gray',lw=3.5)
		# ax1.set_aspect(1)
		# ax1.set_yticks(yticks)
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		ax1.set_xlim(9E-5,6E2)
		ax1.set_ylim(1E42,1E46)
		ax1.set_yscale('log')
		ax1.set_xscale('log')
		ax1.set_xticks(xticks)
		ax1.set_xticklabels(xticks_labels)
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$ [erg/s]')
		ax1.grid()


		ax2 = plt.subplot(gs[1])
		ax2.plot(median_wavelength_z2, percentile_20_flux1_z2, ls='--', color='#377eb8', lw=1)
		ax2.plot(median_wavelength_z2, percentile_80_flux1_z2, ls='--', color='#377eb8', lw=1)
		ax2.fill_between(median_wavelength_z2,percentile_20_flux1_z2,percentile_80_flux1_z2, color='#377eb8',alpha=0.15)

		ax2.plot(median_wavelength_z2, percentile_20_flux2_z2, ls='--', color='#984ea3', lw=1)
		ax2.plot(median_wavelength_z2, percentile_80_flux2_z2, ls='--', color='#984ea3', lw=1)
		ax2.fill_between(median_wavelength_z2,percentile_20_flux2_z2,percentile_80_flux2_z2, color='#984ea3',alpha=0.15)

		ax2.plot(median_wavelength_z2, percentile_20_flux3_z2, ls='--', color='#4daf4a', lw=1)
		ax2.plot(median_wavelength_z2, percentile_80_flux3_z2, ls='--', color='#4daf4a', lw=1)
		ax2.fill_between(median_wavelength_z2,percentile_20_flux3_z2,percentile_80_flux3_z2, color='#4daf4a',alpha=0.15)

		ax2.plot(median_wavelength_z2, percentile_20_flux4_z2, ls='--', color='#ff7f00', lw=1)
		ax2.plot(median_wavelength_z2, percentile_80_flux4_z2, ls='--', color='#ff7f00', lw=1)
		ax2.fill_between(median_wavelength_z2,percentile_20_flux4_z2,percentile_80_flux4_z2, color='#ff7f00',alpha=0.15)

		ax2.plot(median_wavelength_z2, percentile_20_flux5_z2, ls='--', color='#e41a1c', lw=1)
		ax2.plot(median_wavelength_z2, percentile_80_flux5_z2, ls='--', color='#e41a1c', lw=1)
		ax2.fill_between(median_wavelength_z2,percentile_20_flux5_z2,percentile_80_flux5_z2, color='#e41a1c',alpha=0.15)

		ax2.plot(median_wavelength_z2,median_flux1_z2,color='k',lw=4)
		ax2.plot(median_wavelength_z2,median_flux1_z2,color='#377eb8',lw=3.5,label='Panel 1')
		ax2.plot(median_wavelength_z2,median_flux2_z2,color='k',lw=4)
		ax2.plot(median_wavelength_z2,median_flux2_z2,color='#984ea3',lw=3.5,label='Panel 2')
		ax2.plot(median_wavelength_z2,median_flux3_z2,color='k',lw=4)
		ax2.plot(median_wavelength_z2,median_flux3_z2,color='#4daf4a',lw=3.5,label='Panel 3')
		ax2.plot(median_wavelength_z2,median_flux4_z2,color='k',lw=4)
		ax2.plot(median_wavelength_z2,median_flux4_z2,color='#ff7f00',lw=3.5,label='Panel 4')
		ax2.plot(median_wavelength_z2,median_flux5_z2,color='k',lw=4)
		ax2.plot(median_wavelength_z2,median_flux5_z2,color='#e41a1c',lw=3.5,label='Panel 5')


		ax2.plot(median_FIR_wavelength_z2, percentile_20_FIR_f1_z2, ls='--', color='#377eb8', lw=1)
		ax2.plot(median_FIR_wavelength_z2, percentile_80_FIR_f1_z2, ls='--', color='#377eb8', lw=1)
		ax2.fill_between(median_FIR_wavelength_z2,percentile_20_FIR_f1_z2,percentile_80_FIR_f1_z2, color='#377eb8',alpha=0.15)

		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_20_FIR_f2_z2[-3:], ls='--', color='#984ea3', lw=1)
		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_80_FIR_f2_z2[-3:], ls='--', color='#984ea3', lw=1)
		ax2.fill_between(median_FIR_wavelength_z2[-3:],percentile_20_FIR_f2_z2[-3:],percentile_80_FIR_f2_z2[-3:], color='#984ea3',alpha=0.15)

		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_20_FIR_f3_z2[-3:], ls='--', color='#4daf4a', lw=1)
		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_80_FIR_f3_z2[-3:], ls='--', color='#4daf4a', lw=1)
		ax2.fill_between(median_FIR_wavelength_z2[-3:],percentile_20_FIR_f3_z2[-3:],percentile_80_FIR_f3_z2[-3:], color='#4daf4a',alpha=0.15)

		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_20_FIR_f4_z2[-3:], ls='--', color='#ff7f00', lw=1)
		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_80_FIR_f4_z2[-3:], ls='--', color='#ff7f00', lw=1)
		ax2.fill_between(median_FIR_wavelength_z2[-3:],percentile_20_FIR_f4_z2[-3:],percentile_80_FIR_f4_z2[-3:], color='#ff7f00',alpha=0.15)

		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_20_FIR_f5_z2[-3:], ls='--', color='#e41a1c', lw=1)
		ax2.plot(median_FIR_wavelength_z2[-3:], percentile_80_FIR_f5_z2[-3:], ls='--', color='#e41a1c', lw=1)
		ax2.fill_between(median_FIR_wavelength_z2[-3:],percentile_20_FIR_f5_z2[-3:],percentile_80_FIR_f5_z2[-3:], color='#e41a1c',alpha=0.15)

		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f1_z2[-3:],marker='v',ms=12,color='k',lw=4)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f1_z2[-3:],marker='v',ms=10,color='#377eb8',lw=3.5)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f2_z2[-3:],marker='v',ms=12,color='k',lw=4)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f2_z2[-3:],marker='v',ms=10,color='#984ea3',lw=3.5)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f3_z2[-3:],marker='v',ms=12,color='k',lw=4)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f3_z2[-3:],marker='v',ms=10,color='#4daf4a',lw=3.5)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f4_z2[-3:],marker='v',ms=12,color='k',lw=4)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f4_z2[-3:],marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f5_z2[-3:],marker='v',ms=12,color='k',lw=4)
		ax2.plot(median_FIR_wavelength_z2[-3:],median_FIR_f5_z2[-3:],marker='v',ms=10,color='#e41a1c',lw=3.5)


		ax2.plot(median_xray_w_z2[0:2], percentile_20_xray_f1_z2[0:2], ls='--', color='#377eb8', lw=1)
		ax2.plot(median_xray_w_z2[0:2], percentile_80_xray_f1_z2[0:2], ls='--', color='#377eb8', lw=1)
		ax2.fill_between(median_xray_w_z2[0:2],percentile_20_xray_f1_z2[0:2],percentile_80_xray_f1_z2[0:2], color='#377eb8',alpha=0.15)

		ax2.plot(median_xray_w_z2[0:2], percentile_20_xray_f2_z2[0:2], ls='--', color='#984ea3', lw=1)
		ax2.plot(median_xray_w_z2[0:2], percentile_80_xray_f2_z2[0:2], ls='--', color='#984ea3', lw=1)
		ax2.fill_between(median_xray_w_z2[0:2],percentile_20_xray_f2_z2[0:2],percentile_80_xray_f2_z2[0:2], color='#984ea3',alpha=0.15)

		ax2.plot(median_xray_w_z2[0:2], percentile_20_xray_f3_z2[0:2], ls='--', color='#4daf4a', lw=1)
		ax2.plot(median_xray_w_z2[0:2], percentile_80_xray_f3_z2[0:2], ls='--', color='#4daf4a', lw=1)
		ax2.fill_between(median_xray_w_z2[0:2],percentile_20_xray_f3_z2[0:2],percentile_80_xray_f3_z2[0:2], color='#4daf4a',alpha=0.15)

		ax2.plot(median_xray_w_z2[0:2], percentile_20_xray_f4_z2[0:2], ls='--', color='#ff7f00', lw=1)
		ax2.plot(median_xray_w_z2[0:2], percentile_80_xray_f4_z2[0:2], ls='--', color='#ff7f00', lw=1)
		ax2.fill_between(median_xray_w_z2[0:2],percentile_20_xray_f4_z2[0:2],percentile_80_xray_f4_z2[0:2], color='#ff7f00',alpha=0.15)

		ax2.plot(median_xray_w_z2[0:2], percentile_20_xray_f5_z2[0:2], ls='--', color='#e41a1c', lw=1)
		ax2.plot(median_xray_w_z2[0:2], percentile_80_xray_f5_z2[0:2], ls='--', color='#e41a1c', lw=1)
		ax2.fill_between(median_xray_w_z2[0:2],percentile_20_xray_f5_z2[0:2],percentile_80_xray_f5_z2[0:2], color='#e41a1c',alpha=0.15)
		
		ax2.plot(median_xray_w_z2[0:2],median_xray_f1_z2[0:2],color='k',lw=4)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f1_z2[0:2],color='#377eb8',lw=3.5)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f2_z2[0:2],color='k',lw=4)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f2_z2[0:2],color='#984ea3',lw=3.5)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f3_z2[0:2],color='k',lw=4)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f3_z2[0:2],color='#4daf4a',lw=3.5)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f4_z2[0:2],color='k',lw=4)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f4_z2[0:2],color='#ff7f00',lw=3.5)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f5_z2[0:2],color='k',lw=4)
		ax2.plot(median_xray_w_z2[0:2],median_xray_f5_z2[0:2],color='#e41a1c',lw=3.5)
	
		# ax2.set_aspect(1)
		# ax2.set_yticks(yticks)
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		ax2.set_xlim(9E-5,6E2)
		ax2.set_ylim(1E42,1E46)
		ax2.set_yscale('log')
		ax2.set_xscale('log')
		ax2.set_yticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_xticklabels(xticks_labels)
		ax2.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax2.grid()

		ax3 = plt.subplot(gs[2])
		ax3.plot(median_wavelength_z3, percentile_20_flux1_z3, ls='--', color='#377eb8', lw=1)
		ax3.plot(median_wavelength_z3, percentile_80_flux1_z3, ls='--', color='#377eb8', lw=1)
		ax3.fill_between(median_wavelength_z3,percentile_20_flux1_z3,percentile_80_flux1_z3, color='#377eb8',alpha=0.15)

		ax3.plot(median_wavelength_z3, percentile_20_flux2_z3, ls='--', color='#984ea3', lw=1)
		ax3.plot(median_wavelength_z3, percentile_80_flux2_z3, ls='--', color='#984ea3', lw=1)
		ax3.fill_between(median_wavelength_z3,percentile_20_flux2_z3,percentile_80_flux2_z3, color='#984ea3',alpha=0.15)

		ax3.plot(median_wavelength_z3, percentile_20_flux3_z3, ls='--', color='#4daf4a', lw=1)
		ax3.plot(median_wavelength_z3, percentile_80_flux3_z3, ls='--', color='#4daf4a', lw=1)
		ax3.fill_between(median_wavelength_z3,percentile_20_flux3_z3,percentile_80_flux3_z3, color='#4daf4a',alpha=0.15)

		ax3.plot(median_wavelength_z3, percentile_20_flux4_z3, ls='--', color='#ff7f00', lw=1)
		ax3.plot(median_wavelength_z3, percentile_80_flux4_z3, ls='--', color='#ff7f00', lw=1)
		ax3.fill_between(median_wavelength_z3,percentile_20_flux4_z3,percentile_80_flux4_z3, color='#ff7f00',alpha=0.15)

		ax3.plot(median_wavelength_z3, percentile_20_flux5_z3, ls='--', color='#e41a1c', lw=1)
		ax3.plot(median_wavelength_z3, percentile_80_flux5_z3, ls='--', color='#e41a1c', lw=1)
		ax3.fill_between(median_wavelength_z3,percentile_20_flux5_z3,percentile_80_flux5_z3, color='#e41a1c',alpha=0.15)

		ax3.plot(median_wavelength_z3,median_flux1_z3,color='k',lw=4)
		ax3.plot(median_wavelength_z3,median_flux1_z3,color='#377eb8',lw=3.5,label='Panel 1')
		ax3.plot(median_wavelength_z3,median_flux2_z3,color='k',lw=4)
		ax3.plot(median_wavelength_z3,median_flux2_z3,color='#984ea3',lw=3.5,label='Panel 2')
		ax3.plot(median_wavelength_z3,median_flux3_z3,color='k',lw=4)
		ax3.plot(median_wavelength_z3,median_flux3_z3,color='#4daf4a',lw=3.5,label='Panel 3')
		ax3.plot(median_wavelength_z3,median_flux4_z3,color='k',lw=4)
		ax3.plot(median_wavelength_z3,median_flux4_z3,color='#ff7f00',lw=3.5,label='Panel 4')
		ax3.plot(median_wavelength_z3,median_flux5_z3,color='k',lw=4)
		ax3.plot(median_wavelength_z3,median_flux5_z3,color='#e41a1c',lw=3.5,label='Panel 5')


		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_20_FIR_f1_z3[-3:], ls='--', color='#377eb8', lw=1)
		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_80_FIR_f1_z3[-3:], ls='--', color='#377eb8', lw=1)
		ax3.fill_between(median_FIR_wavelength_z3[-3:],percentile_20_FIR_f1_z3[-3:],percentile_80_FIR_f1_z3[-3:], color='#377eb8',alpha=0.15)

		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_20_FIR_f2_z3[-3:], ls='--', color='#984ea3', lw=1)
		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_80_FIR_f2_z3[-3:], ls='--', color='#984ea3', lw=1)
		ax3.fill_between(median_FIR_wavelength_z3[-3:],percentile_20_FIR_f2_z3[-3:],percentile_80_FIR_f2_z3[-3:], color='#984ea3',alpha=0.15)

		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_20_FIR_f3_z3[-3:], ls='--', color='#4daf4a', lw=1)
		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_80_FIR_f3_z3[-3:], ls='--', color='#4daf4a', lw=1)
		ax3.fill_between(median_FIR_wavelength_z3[-3:],percentile_20_FIR_f3_z3[-3:],percentile_80_FIR_f3_z3[-3:], color='#4daf4a',alpha=0.15)

		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_20_FIR_f4_z3[-3:], ls='--', color='#ff7f00', lw=1)
		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_80_FIR_f4_z3[-3:], ls='--', color='#ff7f00', lw=1)
		ax3.fill_between(median_FIR_wavelength_z3[-3:],percentile_20_FIR_f4_z3[-3:],percentile_80_FIR_f4_z3[-3:], color='#ff7f00',alpha=0.15)

		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_20_FIR_f5_z3[-3:], ls='--', color='#e41a1c', lw=1)
		ax3.plot(median_FIR_wavelength_z3[-3:], percentile_80_FIR_f5_z3[-3:], ls='--', color='#e41a1c', lw=1)
		ax3.fill_between(median_FIR_wavelength_z3[-3:],percentile_20_FIR_f5_z3[-3:],percentile_80_FIR_f5_z3[-3:], color='#e41a1c',alpha=0.15)

		# print('percentile 20 1:',percentile_20_FIR_f1_z3)
		# print('percentile 80 1:',percentile_80_FIR_f1_z3)

		# print('percentile 20 2:',percentile_20_FIR_f2_z3)
		# print('percentile 80 2:',percentile_80_FIR_f2_z3)

		# print('percentile 20 3:',percentile_20_FIR_f3_z3)
		# print('percentile 80 3:',percentile_80_FIR_f3_z3)

		# print('percentile 20 4:',percentile_20_FIR_f4_z2)
		# print('percentile 80 4:',percentile_80_FIR_f4_z2)

		# print('percentile 20 5:',percentile_20_FIR_f5_z3)
		# print('percentile 80 5:',percentile_80_FIR_f5_z3)


		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f1_z3[-3:],marker='v',ms=12,color='k',lw=4)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f1_z3[-3:],marker='v',ms=10,color='#377eb8',lw=3.5)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f2_z3[-3:],marker='v',ms=12,color='k',lw=4)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f2_z3[-3:],marker='v',ms=10,color='#984ea3',lw=3.5)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f3_z3[-3:],marker='v',ms=12,color='k',lw=4)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f3_z3[-3:],marker='v',ms=10,color='#4daf4a',lw=3.5)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f4_z3[-3:],marker='v',ms=12,color='k',lw=4)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f4_z3[-3:],marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f5_z3[-3:],marker='v',ms=12,color='k',lw=4)
		ax3.plot(median_FIR_wavelength_z3[-3:],median_FIR_f5_z3[-3:],marker='v',ms=10,color='#e41a1c',lw=3.5)


		ax3.plot(median_xray_w_z3[0:2], percentile_20_xray_f1_z3[0:2], ls='--', color='#377eb8', lw=1)
		ax3.plot(median_xray_w_z3[0:2], percentile_80_xray_f1_z3[0:2], ls='--', color='#377eb8', lw=1)
		ax3.fill_between(median_xray_w_z3[0:2],percentile_20_xray_f1_z3[0:2],percentile_80_xray_f1_z3[0:2], color='#377eb8',alpha=0.15)

		ax3.plot(median_xray_w_z3[0:2], percentile_20_xray_f2_z3[0:2], ls='--', color='#984ea3', lw=1)
		ax3.plot(median_xray_w_z3[0:2], percentile_80_xray_f2_z3[0:2], ls='--', color='#984ea3', lw=1)
		ax3.fill_between(median_xray_w_z3[0:2],percentile_20_xray_f2_z3[0:2],percentile_80_xray_f2_z3[0:2], color='#984ea3',alpha=0.15)

		ax3.plot(median_xray_w_z3[0:2], percentile_20_xray_f3_z3[0:2], ls='--', color='#4daf4a', lw=1)
		ax3.plot(median_xray_w_z3[0:2], percentile_80_xray_f3_z3[0:2], ls='--', color='#4daf4a', lw=1)
		ax3.fill_between(median_xray_w_z3[0:2],percentile_20_xray_f3_z3[0:2],percentile_80_xray_f3_z3[0:2], color='#4daf4a',alpha=0.15)

		ax3.plot(median_xray_w_z3[0:2], percentile_20_xray_f4_z3[0:2], ls='--', color='#ff7f00', lw=1)
		ax3.plot(median_xray_w_z3[0:2], percentile_80_xray_f4_z3[0:2], ls='--', color='#ff7f00', lw=1)
		ax3.fill_between(median_xray_w_z3[0:2],percentile_20_xray_f4_z3[0:2],percentile_80_xray_f4_z3[0:2], color='#ff7f00',alpha=0.15)

		ax3.plot(median_xray_w_z3[0:2], percentile_20_xray_f5_z3[0:2], ls='--', color='#e41a1c', lw=1)
		ax3.plot(median_xray_w_z3[0:2], percentile_80_xray_f5_z3[0:2], ls='--', color='#e41a1c', lw=1)
		ax3.fill_between(median_xray_w_z3[0:2],percentile_20_xray_f5_z3[0:2],percentile_80_xray_f5_z3[0:2], color='#e41a1c',alpha=0.15)

		ax3.plot(median_xray_w_z3[0:2],median_xray_f1_z3[0:2],color='k',lw=4)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f1_z3[0:2],color='#377eb8',lw=3.5)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f2_z3[0:2],color='k',lw=4)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f2_z3[0:2],color='#984ea3',lw=3.5)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f3_z3[0:2],color='k',lw=4)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f3_z3[0:2],color='#4daf4a',lw=3.5)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f4_z3[0:2],color='k',lw=4)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f4_z3[0:2],color='#ff7f00',lw=3.5)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f5_z3[0:2],color='k',lw=4)
		ax3.plot(median_xray_w_z3[0:2],median_xray_f5_z3[0:2],color='#e41a1c',lw=3.5)
	
		# ax3.set_aspect(1)		
		# ax3.set_yticks(yticks)
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		ax3.set_xlim(9E-5, 6E2)
		ax3.set_ylim(1E42,1E46)
		ax3.set_yscale('log')
		ax3.set_xscale('log')
		ax3.set_yticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_xticklabels(xticks_labels)
		secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
		secax3.set_ylabel(r'$\lambda$ L$_\lambda$ [L$_{\odot}$]')
		ax3.grid()

		ax2.legend(fontsize=16)
		plt.savefig('/Users/connor_auge/Desktop/New_plots3/PlotMean_percentilesNEW.pdf')
		plt.show()


	def check_FIR_med(self,x,y,spec_z,uv_slope,mir_slope1,mir_slope2,F1,F2,wfir,ffir):

		norm = np.asarray(F1)
		mark = np.asarray(F2)

		cosmos_s82x_list = []
		cosmos_s82x_wave = []
		for i in range(len(y)):

			upper_wave = np.array([2536859.83,3557125.92,5191371.41])
			rest_upper_w = upper_wave/(1+spec_z[i])
			rest_upper_w_cgs = rest_upper_w*1E-8
			rest_upper_w_microns = rest_upper_w*1E-4
			rest_upper_w_freq = 3E10/rest_upper_w_cgs

			cosmos_upper_lim_jy = np.array([8100.0,10700.0,15400.0])*1E-6
			s82X_upper_lim_jy = np.array([13000.0,12900.0,14800.0])*1E-6
			goodsN_upper_lim_jy = np.array([9000.0,12900.0,12600.0])*1E-6
			goodsS_upper_lim_jy = np.array([8300.0,11500.0,11300.0])*1E-6

			cosmos_upper_lim_cgs = (cosmos_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			s82X_upper_lim_cgs = (s82X_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsN_upper_lim_cgs = (goodsN_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs
			goodsS_upper_lim_cgs = (goodsS_upper_lim_jy*1E-23)/3 # 1σ upper limits in cgs

			cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_upper_w_freq
			s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_upper_w_freq
			goodsN_nuFnu_upper = goodsN_upper_lim_cgs*rest_upper_w_freq
			goodsS_nuFnu_upper = goodsS_upper_lim_cgs*rest_upper_w_freq


			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i]))
				else:
					cosmos_s82x_list.append(y[i][-3:])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i]))
				else:
					a = np.array([y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i]))
				else:
					cosmos_s82x_list.append(y[i][-3:])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i]))
				else:
					a = np.array([y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
			cosmos_s82x_wave.append(rest_upper_w_microns)

		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		fig = plt.figure(figsize=(21,6))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.08) # set the spacing between axes
		gs.update(left=0.06,right=0.94,top=0.9,bottom=0.15)


		ax1 = plt.subplot(gs[0])

		ax1.plot(np.nanmedian(cosmos_s82x_wave[B1[(z1 <= zlim_2)&(z1 > zlim_1)]],axis=0),np.nanmedian(cosmos_s82x_list[B1[(z1 <= zlim_2)&(z1 > zlim_1)]],axis=0),marker='v',ms=10,color='#377eb8',lw=3.5)
		# ax1.plot(np.nanmedian(cosmos_s82x_wave[B2[(z2 <= zlim_2)&(z2 > zlim_1)]],axis=0),np.nanmedian(cosmos_s82x_list[B2[(z2 <= zlim_2)&(z2 > zlim_1)]],axis=0),marker='v',ms=10,color='#984ea3',lw=3.5)
		# ax1.plot(np.nanmedian(cosmos_s82x_wave[B3[(z3 <= zlim_2)&(z3 > zlim_1)]],axis=0),np.nanmedian(cosmos_s82x_list[B3[(z3 <= zlim_2)&(z3 > zlim_1)]],axis=0),marker='v',ms=10,color='#4daf4a',lw=3.5)
		# ax1.plot(np.nanmedian(cosmos_s82x_wave[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],axis=0),np.nanmedian(cosmos_s82x_list[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],axis=0),marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax1.plot(np.nanmedian(cosmos_s82x_wave[B5[(z5 <= zlim_2)&(z5 > zlim_1)]],axis=0),np.nanmedian(cosmos_s82x_list[B5[(z5 <= zlim_2)&(z5 > zlim_1)]],axis=0),marker='v',ms=10,color='#e41a1c',lw=3.5)

		ax1.plot(np.nanmedian(wfir[B1[(z1 <= zlim_2)&(z1 > zlim_1)]],axis=0),np.nanmedian(ffir[B1[(z1 <= zlim_2)&(z1 > zlim_1)]],axis=0),marker='^',ms=10,color='#377eb8',lw=3.5)
		# ax1.plot(np.nanmedian(wfir[B2[(z2 <= zlim_2)&(z2 > zlim_1)]],axis=0),np.nanmedian(ffir[B2[(z2 <= zlim_2)&(z2 > zlim_1)]],axis=0),marker='^',ms=10,color='#984ea3',lw=3.5)
		# ax1.plot(np.nanmedian(wfir[B3[(z3 <= zlim_2)&(z3 > zlim_1)]],axis=0),np.nanmedian(ffir[B3[(z3 <= zlim_2)&(z3 > zlim_1)]],axis=0),marker='^',ms=10,color='#4daf4a',lw=3.5)
		# ax1.plot(np.nanmedian(wfir[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],axis=0),np.nanmedian(ffir[B4[(z4 <= zlim_2)&(z4 > zlim_1)]],axis=0),marker='^',ms=10,color='#ff7f00',lw=3.5)
		ax1.plot(np.nanmedian(wfir[B5[(z5 <= zlim_2)&(z5 > zlim_1)]],axis=0),np.nanmedian(ffir[B5[(z5 <= zlim_2)&(z5 > zlim_1)]],axis=0),marker='^',ms=10,color='#e41a1c',lw=3.5)

		ax1.set_yscale('log')
		ax1.set_xscale('log')
		ax1.set_yticks([1E43,5E43,1E44])
		ax1.set_xticks([100,500,1000])
		ax1.set_ylim(5E42,9E44)
		ax1.set_xlim(100,1000)

		ax2 = plt.subplot(gs[1])

		ax2.plot(np.nanmedian(cosmos_s82x_wave[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0),np.nanmedian(cosmos_s82x_list[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0),marker='v',ms=10,color='#377eb8',lw=3.5)
		# ax2.plot(np.nanmedian(cosmos_s82x_wave[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0),np.nanmedian(cosmos_s82x_list[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0),marker='v',ms=10,color='#984ea3',lw=3.5)
		# ax2.plot(np.nanmedian(cosmos_s82x_wave[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0),np.nanmedian(cosmos_s82x_list[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0),marker='v',ms=10,color='#4daf4a',lw=3.5)
		# ax2.plot(np.nanmedian(cosmos_s82x_wave[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0),np.nanmedian(cosmos_s82x_list[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0),marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax2.plot(np.nanmedian(cosmos_s82x_wave[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0),np.nanmedian(cosmos_s82x_list[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0),marker='v',ms=10,color='#e41a1c',lw=3.5)

		ax2.plot(np.nanmedian(wfir[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0),np.nanmedian(ffir[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],axis=0),marker='^',ms=10,color='#377eb8',lw=3.5)
		# ax2.plot(np.nanmedian(wfir[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0),np.nanmedian(ffir[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],axis=0),marker='^',ms=10,color='#984ea3',lw=3.5)
		# ax2.plot(np.nanmedian(wfir[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0),np.nanmedian(ffir[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],axis=0),marker='^',ms=10,color='#4daf4a',lw=3.5)
		# ax2.plot(np.nanmedian(wfir[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0),np.nanmedian(ffir[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],axis=0),marker='^',ms=10,color='#ff7f00',lw=3.5)
		ax2.plot(np.nanmedian(wfir[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0),np.nanmedian(ffir[B5[(z5 <= zlim_3)&(z5 > zlim_2)]],axis=0),marker='^',ms=10,color='#e41a1c',lw=3.5)

		ax2.set_yscale('log')
		ax2.set_xscale('log')
		ax2.set_yticks([1E43,5E43,1E44])
		ax2.set_xticks([100,500,1000])
		ax2.set_ylim(5E42,9E44)
		ax2.set_xlim(100,1000)

		# ax2.set_yticklabels([])
		

		ax3 = plt.subplot(gs[2])

		ax3.plot(np.nanmedian(cosmos_s82x_wave[B1[(z1 <= zlim_4) & (z1 > zlim_3)]], axis=0), np.nanmedian(
			cosmos_s82x_list[B1[(z1 <= zlim_4) & (z1 > zlim_3)]], axis=0), marker='v', ms=10, color='#377eb8', lw=3.5)
		# ax3.plot(np.nanmedian(cosmos_s82x_wave[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0),np.nanmedian(cosmos_s82x_list[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0),marker='v',ms=10,color='#984ea3',lw=3.5)
		# ax3.plot(np.nanmedian(cosmos_s82x_wave[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0),np.nanmedian(cosmos_s82x_list[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0),marker='v',ms=10,color='#4daf4a',lw=3.5)
		# ax3.plot(np.nanmedian(cosmos_s82x_wave[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0),np.nanmedian(cosmos_s82x_list[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0),marker='v',ms=10,color='#ff7f00',lw=3.5)
		ax3.plot(np.nanmedian(cosmos_s82x_wave[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0),np.nanmedian(cosmos_s82x_list[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0),marker='v',ms=10,color='#e41a1c',lw=3.5)

		ax3.plot(np.nanmedian(wfir[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0),np.nanmedian(ffir[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],axis=0),marker='^',ms=10,color='#377eb8',lw=3.5)
		# ax3.plot(np.nanmedian(wfir[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0),np.nanmedian(ffir[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],axis=0),marker='^',ms=10,color='#984ea3',lw=3.5)
		# ax3.plot(np.nanmedian(wfir[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0),np.nanmedian(ffir[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],axis=0),marker='^',ms=10,color='#4daf4a',lw=3.5)
		# ax3.plot(np.nanmedian(wfir[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0),np.nanmedian(ffir[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],axis=0),marker='^',ms=10,color='#ff7f00',lw=3.5)
		ax3.plot(np.nanmedian(wfir[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0),np.nanmedian(ffir[B5[(z5 <= zlim_4)&(z5 > zlim_3)]],axis=0),marker='^',ms=10,color='#e41a1c',lw=3.5)
	
		ax3.set_yscale('log')
		ax3.set_xscale('log')
		ax3.set_yticks([1E43,5E43,1E44])
		ax3.set_xticks([100, 500, 1000])
		ax3.set_ylim(5E42,9E44)
		ax3.set_xlim(100,1000)

		# ax3.set_yticklabels([])
		

		plt.show()


	def allison_seds(self,param,param2,Fx1,Fx2,Fx3,emis1,emis2,emis3,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):
		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.0
		clim2 = 46

		L = np.asarray(L)
		x = np.asarray(x)
		y = np.asarray(y)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)

		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		B1 = L > 44
		B2 = L < 44

		# B1 = spec_type == 1
		# B2 = spec_type == 2

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		# norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		# mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		norm1 = norm[B1]
		norm2 = norm[B2]

		# spec_type1 = spec_type[B1]
		# spec_type2 = spec_type[B2]

		mark1 = mark[B1]
		mark2 = mark[B2]

		# cosmos_norm1 = norm1[mark1 == 0]
		# s82X_norm1 = norm1[mark1 == 1]

		# cosmos_norm2 = norm2[mark2 == 0]
		# s82X_norm2 = norm2[mark2 == 1]

		# cosmos_norm3 = norm3[mark3 == 0]
		# s82X_norm3 = norm3[mark3 == 1]

		# cosmos_norm4 = norm4[mark4 == 0]
		# s82X_norm4 = norm4[mark4 == 1]

		# cosmos_norm5 = norm5[mark5 == 0]
		# s82X_norm5 = norm5[mark5 == 1]

		median_wavelength2 = median_wavelength[B2]
		for i in range(len(median_wavelength2)):
			ind = np.where(median_wavelength2[i] < -0.7)
			median_wavelength2[i][ind] = np.nan
	
		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# xticks = [1E-3,1E-2,1E-1,1,10,100]
		# yticks = [1E-2,0.1,1,10]

		xticks = [1E-2,1E-1,1,10,100]
		yticks = [0.1,1,10]


		fig = plt.figure(figsize=(12,12),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=2, ncols=2, left=0.2,right=0.85,wspace=-0.25,hspace=0.2,width_ratios=[3,0.15])


		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=4)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(9E-2,7E2)
		ax1.set_ylim(5E-2,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
		# ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax1.set_title(r'L$_{\mathrm{X}}$ > 10$^{44}$')
		# ax1.set_title('Type 1')

		ax2 = fig.add_subplot(gs1[1,0])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		spec_type2 = spec_type[B2]


		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		ax2.plot(np.nanmedian(10**median_wavelength2,axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=4)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(9E-2,7E2)
		ax2.set_ylim(5E-2,50)
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.8,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.set_title(r'10$^{42}$ < L$_{\mathrm{X}}$ < 10$^{44}$')
		# ax2.set_title('Type 2')
		ax2.text(-0.2,0.625,r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',transform=ax2.transAxes,rotation=90,fontsize=27)

		# ax2.set_ylabel(r'$\lambda$ L$_\lambda$ normalized at 1$\mu$m')
		ax2.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		
		cbar_ax = fig.add_subplot(gs1[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')

		plt.savefig('/Users/connor_auge/Desktop/Allison_Lx_bins.pdf')
		plt.show()

	def NSF_seds_3panel(self,x,y,L,uv_slope,mir_slope1,mir_slope2,median_wavelength,median_flux,F1):	
		# x[y > 5E2] = np.nan
		# y[y > 5E2] = np.nan
		# x[y < 1E-4] = np.nan
		# y[y < 1E-4] = np.nan

		clim1 = 43
		clim2 = 46

		L = np.asarray(L)
		x = np.asarray(x)
		y = np.asarray(y)
		norm = np.asarray(F1)

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]
		print('CHECK: ',len(B3))
		scale1 = np.nanmedian(norm[B1],axis=0)/np.nanmedian(norm[B3],axis=0)
		scale2 = np.nanmedian(norm[B2],axis=0)/np.nanmedian(norm[B3],axis=0)
		scale4 = np.nanmedian(norm[B4],axis=0)/np.nanmedian(norm[B3],axis=0)
		scale5 = np.nanmedian(norm[B5],axis=0)/np.nanmedian(norm[B3],axis=0)
		# scale_s82x = np.nanmedian(norm[B3],axis=0)/np.nanmedian(norm[B2],axis=0)
		# scale_goods = np.nanmedian(norm[B1],axis=0)/np.nanmedian(norm[B2],axis=0)

		plt.rcParams['font.size'] = 40
		plt.rcParams['axes.linewidth'] = 4
		plt.rcParams['xtick.major.size'] = 6
		plt.rcParams['xtick.major.width'] = 5
		plt.rcParams['ytick.major.size'] = 6
		plt.rcParams['ytick.major.width'] = 5

		xticks = [1E-4,1E-3,1E-2, 1E-1, 1, 10, 100]
		yticks = [1E-2, 0.1, 1, 10]

		xticklabels = ['-4',' ','-2',' ','0',' ','2']
		yticklabels = [-2, -1, 0, 1]

		fig = plt.figure(figsize=(50,11))
		gs = fig.add_gridspec(nrows=2, ncols=5, left=0.1,right=0.9,wspace=0.05,hspace=0.01,height_ratios=[0.2,3])

		ax1 = fig.add_subplot(gs[1, 4])#, aspect='equal', adjustable='box')
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]

		y1[(y1 > 5) & (x1 > 1)] = np.nan
		y1[(y1 < 0.3) & (x1 > 0.1)] = np.nan


		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow_r')
		
		lc1 = self.multilines(x1,y1*scale1,L1,cmap='rainbow_r',lw=1.5,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0)*scale1,color='k',lw=4)
		ax1.plot(np.nanmedian(x1,axis=0)[0:2],np.nanmedian(y1,axis=0)[0:2]*scale1,color='k',lw=4)
		axcb1 = fig.colorbar(lc1,orientation='horizontal',pad=-0.1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(1E-3,75)
		ax1.set_xticklabels(xticklabels)
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_yticklabels([])
		ax1.text(0.05,0.85,'1',transform=ax1.transAxes,fontsize=45,weight='bold')
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.text(0.725,0.08,str((len(x1)/len(x))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		# ax1.set_ylabel(r'Log $\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=20)
		# ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax2 = fig.add_subplot(gs[1, 3])#, aspect='equal', adjustable='box')
		x3 = x[B2]
		y3 = y[B2]
		L3 = L[B2]

		lc2 = self.multilines(x3,y3*scale2,L3,cmap='rainbow_r',lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0)*scale2,color='k',lw=4)
		ax2.plot(np.nanmedian(x3,axis=0)[0:2],np.nanmedian(y3,axis=0)[0:2]*scale2,color='k',lw=4)
		axcb2 = fig.colorbar(lc2,orientation='horizontal',pad=-0.1)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(1E-3,75)
		ax2.set_yticklabels([])
		ax2.set_xticklabels(xticklabels)
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.85,'2',transform=ax2.transAxes,fontsize=45,weight='bold')
		ax2.text(0.05,0.7,f'n = {len(x3)}',transform=ax2.transAxes)
		ax2.text(0.725,0.08,str((len(x3)/len(x))*100)[0:4]+'%',transform=ax2.transAxes,weight='bold')
		# ax2.set_xlabel(r'Log$_{10}$ Rest Wavelength [$\mu$m]',fontsize=22)

		ax3 = fig.add_subplot(gs[1, 2])  # , aspect='equal', adjustable='box')
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]

		# y3[(y3 < 0.3) & (x3 > 10)] = np.nan
		
		lc3 = self.multilines(x3,y3,L3,cmap='rainbow_r',lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=4)
		ax3.plot(np.nanmedian(x3,axis=0)[0:2],np.nanmedian(y3,axis=0)[0:2],color='k',lw=4)	
		axcb3 = fig.colorbar(lc3,orientation='horizontal',pad=-0.1)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(1E-3,75)
		ax3.set_xticklabels(xticklabels)
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_yticklabels(ytick_labels)
		ax3.set_yticklabels([])
		ax3.text(0.05,0.85,'3',transform=ax3.transAxes,fontsize=45,weight='bold')
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		ax3.text(0.725,0.08,str((len(x3)/len(x))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		# ax3.set_ylabel(r'Log$_{10}$ $\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=20)
		ax3.set_xlabel(r'Log Rest Wavelength [$\mu$m]')

		ax4 = fig.add_subplot(gs[1,1])#, aspect='equal', adjustable='box')
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]

		# y4[(y4 < 0.3) & (x4 > 10)] = np.nan
		
		lc4 = self.multilines(x4,y4*scale4,L4,cmap='rainbow_r',lw=1.5,rasterized=True)
		ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0)*scale4,color='k',lw=4)
		ax4.plot(np.nanmedian(x4,axis=0)[0:2],np.nanmedian(y4,axis=0)[0:2]*scale4,color='k',lw=4)	
		axcb4 = fig.colorbar(lc4,orientation='horizontal',pad=-0.1)
		axcb4.mappable.set_clim(clim1,clim2)
		axcb4.remove()

		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(1E-3,75)
		ax4.set_xticklabels(xticklabels)
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.set_yticklabels([])
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		ax4.text(0.05,0.85,'4',transform=ax4.transAxes,fontsize=45,weight='bold')
		ax4.text(0.725,0.08,str((len(x4)/len(x))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		# ax4.set_ylabel(r'Log$_{10}$ $\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=20)
		# ax4.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax5 = fig.add_subplot(gs[1, 0])  # , aspect='equal', adjustable='box')
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]

		# y5[(y5 < 0.3) & (x5 > 10)] = np.nan
		
		lc5 = self.multilines(x5,y5*scale5,L5,cmap='rainbow_r',lw=1.5,rasterized=True)
		ax5.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0)*scale5,color='k',lw=4)
		ax5.plot(np.nanmedian(x4,axis=0)[0:2],np.nanmedian(y4,axis=0)[0:2]*scale5,color='k',lw=4)	
		axcb5 = fig.colorbar(lc5,orientation='horizontal',pad=-0.1)
		axcb5.mappable.set_clim(clim1,clim2)
		axcb5.remove()

		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(1E-3,75)
		ax5.set_xticklabels(xticklabels)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)
		ax5.set_yticklabels(yticklabels)
		ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		ax5.text(0.05,0.85,'5',transform=ax5.transAxes,fontsize=45,weight='bold')
		ax5.text(0.725,0.08,str((len(x5)/len(x))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		ax5.set_ylabel(r'Normalized Log $\lambda$L$_\lambda$')
		# ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()
		
		cbar_ax = fig.add_subplot(gs[:-1, :])
		cb = fig.colorbar(test, cax=cbar_ax, orientation='horizontal')
		cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')
		cb.ax.xaxis.set_ticks_position('top')
		cb.ax.xaxis.set_label_position('top')



		# plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/SEDs_horizontal_5panel.pdf')
		plt.show()



	def SEDs_morph(self,x,y,Lx,BT,DT,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):
		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.5
		clim2 = 46

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(Lx)
		BT = np.asarray(BT)
		DT = np.asarray(DT)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		norm1, norm2, norm3, norm4, norm5 = norm[B1], norm[B2], norm[B3], norm[B4], norm[B5]
		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		cosmos_norm1 = norm1[mark1 == 0]
		s82X_norm1 = norm1[mark1 == 1]

		cosmos_norm2 = norm2[mark2 == 0]
		s82X_norm2 = norm2[mark2 == 1]

		cosmos_norm3 = norm3[mark3 == 0]
		s82X_norm3 = norm3[mark3 == 1]

		cosmos_norm4 = norm4[mark4 == 0]
		s82X_norm4 = norm4[mark4 == 1]

		cosmos_norm5 = norm5[mark5 == 0]
		s82X_norm5 = norm5[mark5 == 1]
	
		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]

		fig = plt.figure(figsize=(20,15),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=5, ncols=2, left=0.08,right=0.38,wspace=-0.3,hspace=0.1,width_ratios=[3,0.25])
		gs2 = fig.add_gridspec(nrows=5, ncols=2, left=0.5,right=0.99,wspace=0.15,hspace=0.1)

		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		BT1 = BT[B1]
		DT1 = DT[B1]
		spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax2 = fig.add_subplot(gs1[1,0])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		BT2 = BT[B2]
		DT2 = DT[B2]
		spec_type2 = spec_type[B2]


		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax3 = fig.add_subplot(gs1[2,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		BT3 = BT[B3]
		DT3 = DT[B3]
		spec_type3 = spec_type[B3]

		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')


		ax4 = fig.add_subplot(gs1[3,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		BT4 = BT[B4]
		DT4 = DT[B4]
		spec_type4 = spec_type[B4]

		lc4 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5)
		ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		axcb4.remove()

		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(5E-3,50)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax5 = fig.add_subplot(gs1[4,0])
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		BT5 = BT[B5]
		DT5 = DT[B5]
		spec_type5 = spec_type[B5]

		x6 = x5
		y6 = y5

		lc5 = self.multilines(x6,y6,L5,cmap='rainbow',lw=1.5)
		ax5.plot(np.nanmedian(10**median_wavelength[B5],axis=0),np.nanmedian(10**median_flux[B5],axis=0),color='k',lw=3.5)
		axcb5 = fig.colorbar(lc5)
		axcb5.mappable.set_clim(clim1,clim2)
		axcb5.remove()

		ax5.set_xscale('log')
		ax5.set_yscale('log')
		ax5.set_xlim(8E-5,7E2)
		ax5.set_ylim(5E-3,50)
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)

		ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		yticks = [0,10,20,30]
		
		cbar_ax = fig.add_subplot(gs1[:,-1:])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')

		# print(BT1)
		# print(BT2)
		# print(BT3)
		# print(BT4)
		# print(BT5)

		# print(DT1)
		# print(DT2)
		# print(DT3)
		# print(DT4)
		# print(DT5)

		ax6 = fig.add_subplot(gs2[0,0])
		ax6.hist(BT1,color='gray')
		# ax6.axvline(np.median(BT1),ls='--',color='k',lw=3)
		# ax6.set_xlim(42,46)
		# ax6.set_ylim(0,35)
		# ax6.set_yticks(yticks)
		ax6.set_xticklabels([])
		ax6.grid()

		ax7 = fig.add_subplot(gs2[1,0])
		ax7.hist(BT2,color='gray')
		# ax7.axvline(np.median(BT2),ls='--',color='k',lw=3)
		# ax7.set_xlim(42,46)
		# ax7.set_ylim(0,35)
		# ax7.set_yticks(yticks)
		ax7.set_xticklabels([])
		ax7.grid()

		ax8 = fig.add_subplot(gs2[2,0])
		ax8.hist(BT3,color='gray')
		# ax8.axvline(np.median(BT3),ls='--',color='k',lw=3)
		# ax8.set_xlim(42,46)
		# ax8.set_ylim(0,35)
		# ax8.set_yticks(yticks)
		ax8.set_xticklabels([])
		ax8.grid()

		ax9 = fig.add_subplot(gs2[3,0])
		ax9.hist(BT4,color='gray')
		# ax9.axvline(np.median(BT4),ls='--',color='k',lw=3)
		# ax9.set_xlim(42,46)
		# ax9.set_ylim(0,35)
		# ax9.set_yticks(yticks)
		ax9.set_xticklabels([])
		ax9.grid()

		ax10 = fig.add_subplot(gs2[4,0])
		ax10.hist(BT5,color='gray')
		# ax10.axvline(np.median(BT5),ls='--',color='k',lw=3)
		# ax10.set_xlim(42,46)
		# ax10.set_ylim(0,35)
		# ax10.set_yticks(yticks)
		ax10.set_xlabel(r'B/T')
		ax10.grid()

		ax11 = fig.add_subplot(gs2[0,1])
		ax11.hist(DT1,color='gray')
		# ax11.axvline(np.median(BT1),ls='--',color='k',lw=3)
		# ax11.set_xlim(42,46)
		# ax11.set_ylim(0,35)
		# ax11.set_yticks(yticks)
		ax11.set_xticklabels([])
		ax11.grid()

		ax12 = fig.add_subplot(gs2[1,1])
		ax12.hist(DT2,color='gray')
		# ax12.axvline(np.median(BT2),ls='--',color='k',lw=3)
		# ax12.set_xlim(42,46)
		# ax12.set_ylim(0,35)
		# ax12.set_yticks(yticks)
		ax12.set_xticklabels([])
		ax12.grid()

		ax13 = fig.add_subplot(gs2[2,1])
		ax13.hist(DT3,color='gray')
		# ax13.axvline(np.median(BT3),ls='--',color='k',lw=3)
		# ax13.set_xlim(42,46)
		# ax13.set_ylim(0,35)
		# ax13.set_yticks(yticks)
		ax13.set_xticklabels([])
		ax13.grid()

		ax14 = fig.add_subplot(gs2[3,1])
		ax14.hist(DT4,color='gray')
		# ax14.axvline(np.median(BT4),ls='--',color='k',lw=3)
		# ax14.set_xlim(42,46)
		# ax14.set_ylim(0,35)
		# ax14.set_yticks(yticks)
		ax14.set_xticklabels([])
		ax14.grid()

		ax15 = fig.add_subplot(gs2[4,1])
		ax15.hist(DT5,color='gray')
		# ax15.axvline(np.median(BT5),ls='--',color='k',lw=3)
		# ax15.set_xlim(42,46)
		# ax15.set_ylim(0,35)
		# ax15.set_yticks(yticks)
		ax15.set_xlabel(r'D/T')
		ax15.grid()

		plt.savefig('/Users/connor_auge/Desktop/LetterFigures/SED_Morph_COSMOS2.pdf')
		plt.show()


	def MIR_Lx_scatter(self,savestring,Lx,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None):


		'''Read in Asmus 2015 Lx - MIR data'''
		asmus = ascii.read('/Users/connor_auge/Research/Disertation/catalogs/Asmus2015.csv')
		asmus_mir = asmus['log_Lnuc_12']
		asmus_Lx = asmus['log_Lx_int_2_10']

		# stern = fits.open('/Users/connor_auge/Research/Disertation/catalogs/Stern_Quasars.fit')
		# stern_data = stern[1].data
		# stern_id = np.asarray(stern_data['SDSS'])
		# stern_z = stern_data['z'][ix]
		# stern_L6 = stern_data['logL6'][ix]
		# stern_Lx = stern_data['logL2-10'][ix]

		stern = ascii.read('/Users/connor_auge/Research/Disertation/catalogs/Stern_sample.txt')
		stern_L6 = np.asarray(stern['L6'])
		stern_Lx = np.asarray(stern['L2-10'])


		plt.rcParams['font.size']=16
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3


		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		# B1 = np.where(spec_z < 0.5)[0]
		# B2 = np.where(np.logical_and(spec_z > 0.5, spec_z < 1.0))[0]
		# B3 = np.where(np.logical_and(spec_z > 1.0, spec_z < 1.5))[0]
		# B4 = np.where(np.logical_and(spec_z > 1.5, spec_z < 2.0))[0]
		# B5 = np.where(np.logical_and(spec_z > 2.0, spec_z < 2.5))[0]

		MIR = np.log10(np.asarray(emis2)*np.asarray(F1))
		MIR2 = np.log10(np.asarray(emis3)*np.asarray(F12))
		Lx2 = np.asarray(emis4)
		MIR3 = np.log10(np.asarray(emis5)*np.asarray(F13))
		Lx3 = np.asarray(emis6)

		# Lx = np.log10(np.asarray(Fx1)*np.asarray(F1))
		# Lx = np.log10(Lx)

		MIR_fit = np.append(MIR[B1],MIR[B2])
		MIR_fit = np.append(MIR_fit,MIR[B3])
		MIR_fit = np.append(MIR_fit,MIR[B4])
		MIR_fit = np.append(MIR_fit,MIR[B5])

		Lx_fit = np.append(Lx[B1],Lx[B2])
		Lx_fit = np.append(Lx_fit,Lx[B3])
		Lx_fit = np.append(Lx_fit,Lx[B4])
		Lx_fit = np.append(Lx_fit,Lx[B5])

		z = np.polyfit(Lx_fit,MIR_fit,1)
		p = np.poly1d(z)
		xp = np.linspace(41,48,10)

		z2 = np.polyfit(Lx_fit,MIR_fit,2)
		p2 = np.poly1d(z2)

		fig = plt.figure(figsize=(8,8))
		plt.title('2.5 < z < 2.8')
		plt.scatter(Lx[B1],MIR[B1],color='b',label='Panel 1')
		plt.scatter(Lx[B2],MIR[B2],color='purple',label='Panel 2')
		plt.scatter(Lx[B3],MIR[B3],color='green',label='Panel 3')
		plt.scatter(Lx[B4],MIR[B4],color='orange',label='Panel 4')
		plt.scatter(Lx[B5],MIR[B5],color='red',label='Panel 5')
		# plt.scatter(Lx[B1],MIR[B1],color='b',label='z < 0.5')
		# plt.scatter(Lx[B2],MIR[B2],color='purple',label='0.5 < z < 1.0')
		# plt.scatter(Lx[B3],MIR[B3],color='green',label='1.0 < z < 1.5')
		# plt.scatter(Lx[B4],MIR[B4],color='orange',label='1.5 < z < 2.0')
		# plt.scatter(Lx[B5],MIR[B5],color='red',label='2.0 < z < 2.5')
		# plt.plot(self.stern_Lx_2_10(np.linspace(41,48,10)),np.linspace(41,48,10),ls='--',color='k',label='Stern 2015')
		plt.scatter(asmus_Lx,asmus_mir,color='gray',edgecolors='k',label='Asmus 2015 data')
		plt.plot(self.amus_Lx_2_10_MIR12(np.linspace(41,48,10)),np.linspace(41,48,10),ls='--',color='k',label='Asmus 2015')
		# plt.plot(np.linspace(41,48,10),self.ichikawa_MIR_12(np.linspace(41,48,10)),ls='--',color='green',label='Ichikawa 2015')
		# plt.scatter(Lx2,MIR2,color='gray',edgecolors='k',marker='s',label='Starburst')
		# plt.scatter(stern_Lx,stern_L6,color='gray',edgecolors='k',label='Stern Data')
		plt.axvline(42.5,color='k',lw=3)
		plt.plot(xp,p(xp),color='k',label='Auge')
		# plt.plot(xp,p2(xp),color='b',label='Auge 2 degree fit')
		plt.xlabel(r'log $\lambda\mathrm{L}_{\lambda}({2-10\mathrm{kev}})$')
		plt.ylabel(r'log $\lambda\mathrm{L}_{\lambda}(12\mu\mathrm{m})$')
		# plt.xlim(-3,3)
		# plt.ylim(-3,3)
		plt.xlim(40,48)
		plt.ylim(40,48)
		plt.legend()
		plt.grid()
		plt.savefig('/Users/connor_auge/Desktop/MIR_Lx_scatter'+savestring+'.png')
		plt.show()


	def FIR_Lx_scatter(self, savestring, Lx, Fx1, Fx2, Fx3, emis1, emis2, f1, f2, f3, f4, spec_type, F1=None, F12=None, F13=None, F2=None, emis3=None, emis4=None, emis5=None, emis6=None, spec_z=None, uv_slope=None, mir_slope1=None, mir_slope2=None, upper_check=None):

		plt.rcParams['font.size']=16
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'


		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <= 0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3,np.logical_and(mir_slope1 < -0.2, mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3,np.logical_and(mir_slope1 < -0.2, mir_slope2 <= 0.0)))[0]


		f_3 = np.asarray([10**i for i in f3])
		FIR = np.log10(f_3*F1)
		# FIR = f3


		FIR_fit1 = np.append(FIR[B1],FIR[B2])
		FIR_fit1 = np.append(FIR_fit1,FIR[B3])

		FIR_fit2 = np.append(FIR[B4],FIR[B5])

		Lx_fit1 = np.append(Lx[B1],Lx[B2])
		Lx_fit1 = np.append(Lx_fit1,Lx[B3])

		Lx_fit2 = np.append(Lx[B4],Lx[B5])

		z1 = np.polyfit(FIR_fit1[~np.isnan(FIR_fit1)],Lx_fit1[~np.isnan(FIR_fit1)],1)
		p1 = np.poly1d(z1)

		z2 = np.polyfit(FIR_fit2[~np.isnan(FIR_fit2)],Lx_fit2[~np.isnan(FIR_fit2)],1)
		p2 = np.poly1d(z2)

		xp = np.linspace(-3,3,10)

		fig = plt.figure(figsize=(10,10))
		# plt.title('0.9 < z < 1.1')
		plt.scatter(FIR[B1][upper_check[B1] == 0],Lx[B1][upper_check[B1] == 0],label='Panel 1',color=c1)
		plt.scatter(FIR[B2][upper_check[B2] == 0],Lx[B2][upper_check[B2] == 0],label='Panel 2',color=c2)
		plt.scatter(FIR[B3][upper_check[B3] == 0],Lx[B3][upper_check[B3] == 0],label='Panel 3',color=c3)
		plt.scatter(FIR[B4][upper_check[B4] == 0],Lx[B4][upper_check[B4] == 0],label='Panel 4',color=c4)
		plt.scatter(FIR[B5][upper_check[B5] == 0],Lx[B5][upper_check[B5] == 0],label='Panel 5',color=c5)

		plt.scatter(FIR[B1][upper_check[B1] == 1], Lx[B1][upper_check[B1] == 1],marker='<', color=c1)
		plt.scatter(FIR[B2][upper_check[B2] == 1], Lx[B2][upper_check[B2] == 1],marker='<', color=c2)
		plt.scatter(FIR[B3][upper_check[B3] == 1], Lx[B3][upper_check[B3] == 1],marker='<', color=c3)
		plt.scatter(FIR[B4][upper_check[B4] == 1], Lx[B4][upper_check[B4] == 1],marker='<', color=c4)
		plt.scatter(FIR[B5][upper_check[B5] == 1], Lx[B5][upper_check[B5] == 1],marker='<', color=c5)

		print('bin 5 length: ',len(FIR[B5][upper_check[B5] == 0])+len(FIR[B5][upper_check[B5] == 1]))


		# plt.plot(xp,p1(xp),color='k',label='Panels 1 + 2 + 3')
		# plt.plot(xp,p2(xp),color='k',ls='--',label='Panels 4 + 5')

		# plt.axvline(42.5,color='k',lw=3)
		# plt.plot(xp,p(xp),color='k',label='Auge')
		plt.ylabel(r'log $\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})$')
		# plt.xlabel(r'log $\lambda\mathrm{L}_{\lambda}(100\mu\mathrm{m})/\lambda\mathrm{L}_{\lambda}(1\mu\mathrm{m})$')
		plt.xlabel(r'log $\lambda\mathrm{L}_{\lambda}(100\mu\mathrm{m}) $')

		# plt.xlim(-2.1,2.1)
		plt.xlim(42,47)
		plt.ylim(41,46)
		plt.legend()
		plt.grid()
		plt.savefig('/Users/connor_auge/Desktop/Paper/Check_WISE/FIR_Lx_scatter'+savestring+'.png')
		plt.show()


	def L_Lx_scatter(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None,emis7=None,emis8=None,emis9=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None,fir_frac=None):

		plt.rcParams['font.size']=16
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		L[L > 100] = np.nan
		# Lx[Lx > 45.5] = np.nan


		l = np.asarray([10**i for i in L])
		# l -= (10**44.66)

		# print(L - np.log10(l))
		L = np.log10(l)
		L -= np.log10(3.8E33)

		emis5=[]
		emis6=[]
		emis9=[]
		F13=[]

		e5 = np.asarray([10**i for i in emis5])
		# e5 -= (10**44.66)
		# emis5 = np.log10(e5)
		emis5 = e5
		emis5 /= 3.8E33

		# print(emis6,emis5)
		# print(emis7,emis8)


		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		up_check1,up_check2,up_check3,up_check4,up_check5 = up_check[B1],up_check[B2],up_check[B3],up_check[B4],up_check[B5]


		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one
		scale_GOALS = np.asarray(F13)/temp_L_one

		scale = (np.nanmedian(F1[B4])-np.nanmedian(F1[B4])*0.5)/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5
		temp_L_bol_GOALS = temp_L_bol*scale_GOALS

		temp_L_bol_all = temp_L_bol*scale
		print('temp L bol :',temp_L_bol_all)
		print('temp L bol :',np.log10(temp_L_bol_all))

		f_3 = np.asarray([10**i for i in f3])
		f_3 *= F1
		f3 = np.log10(f_3)

		f_4 = np.asarray([10**i for i in f4])
		f_4 *= F1
		f4 = np.log10(f_4)

		f_1 = np.asarray([10**i for i in f1])
		f_1 *= F1
		f1 = np.log10(f_1)


		# e9 = np.asarray([10**i for i in emis9])
		# emis9 *= F13
		emis9 = np.log10(emis9)

		F1 = np.log10(F1)


		# l -= (tem_L_bol*)

		# temp_L_bol3 = 0
		# temp_L_bol4 = 0
		# temp_L_bol5 = 0
		# temp_L_bol_GOALS = 0
		# temp_L_bol_all = 0

		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)
		emis6 -= np.log10(3.8E33)
		# B1 = np.where(spec_z < 0.5)[0]
		# B2 = np.where(np.logical_and(spec_z > 0.5, spec_z < 1.0))[0]
		# B3 = np.where(np.logical_and(spec_z > 1.0, spec_z < 1.5))[0]
		# B4 = np.where(np.logical_and(spec_z > 1.5, spec_z < 2.0))[0]
		# B5 = np.where(np.logical_and(spec_z > 2.0, spec_z < 2.5))[0]
		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		fir_frac1 = fir_frac[B1]
		fir_frac2 = fir_frac[B2]
		fir_frac3 = fir_frac[B3]
		fir_frac4 = fir_frac[B4]
		fir_frac5 = fir_frac[B5]

		# L1 = L[B1]
		# L2 = L[B2]
		# L3 = L[B3] - temp_L_bol3
		# L4 = L[B4] - temp_L_bol4
		# L5 = L[B5] - temp_L_bol5

		print(L[B4])
		print(L[B4] - temp_L_bol_all)


		L11 = L[B1] - temp_L_bol_all
		L21 = L[B2] - temp_L_bol_all
		L31 = L[B3] - temp_L_bol_all
		L41 = L[B4] - temp_L_bol_all
		L51 = L[B5] - temp_L_bol_all

		L1 = L[B1] - temp_L_bol_all
		L2 = L[B2] - temp_L_bol_all
		L3 = L[B3] - temp_L_bol_all
		L4 = L[B4] - temp_L_bol_all
		L5 = L[B5] - temp_L_bol_all

		# print(np.log10(L[B4]))
		# print(L[B4]-temp_L_bol_all)
		# print(np.log10(L[B4]-temp_L_bol_all))
		# print(len(np.log10(L[B4])))
		# print(len(L[B4]-temp_L_bol_all))
		# print(len(np.log10(L[B4]-temp_L_bol_all)))

		L_sub = np.append(L1,L2)
		L_sub = np.append(L_sub,L3)
		L_sub = np.append(L_sub,L4)
		L_sub = np.append(L_sub,L5)
		# L_sub = np.append(L_sub,emis5 - temp_L_bol_GOALS)

		lx = np.asarray([10**i for i in Lx])
		Lx = lx
		e6 = np.asarray([10**i for i in emis6])
		emis6 = e6
		
		Lx1 = Lx[B1]
		Lx2 = Lx[B2]
		Lx3 = Lx[B3]
		Lx4 = Lx[B4]
		Lx5 = Lx[B5]
		Lx_sub = np.append(Lx1,Lx2)
		Lx_sub = np.append(Lx_sub,Lx3)
		Lx_sub = np.append(Lx_sub,Lx4)
		Lx_sub = np.append(Lx_sub,Lx5)
		# Lx_sub = np.append(Lx_sub,emis6)

		fc1 = F1[B1]
		fc2 = F1[B2]
		fc3 = F1[B3]
		fc4 = F1[B4]
		fc5 = F1[B5]
		fc_sub = np.append(fc1,fc2)
		fc_sub = np.append(fc_sub,fc3)
		fc_sub = np.append(fc_sub,fc4)
		fc_sub = np.append(fc_sub,fc5)

		sort = np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]).argsort()

		z = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],1)
		# z = np.polyfit(Lx_sub[np.isfinite(np.log10(L_sub))],L_sub[np.isfinite(np.log10(L_sub))],1)
		# z = np.polyfit(Lx_sub[np.isfinite(L_sub)],L_sub[np.isfinite(L_sub)],1)		
		p = np.poly1d(z)


		z2 = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],2)
		# z2 = np.polyfit(Lx_sub[np.isfinite(np.log10(L_sub))],L_sub[np.isfinite(np.log10(L_sub))],2)
		p2 = np.poly1d(z2)

		


		x = np.linspace(8,15,20)
		# x = np.arange(8,14)
		y = p(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort])
		# y2 = p2(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort])
		# print(y)
		# print(z,np.log10(z))

		# for i in range(len(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))][sort]))):
			# print(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))][sort][i]),np.log10(L_sub[np.isfinite(np.log10(L_sub))][sort][i]))

		U = np.zeros(np.shape(Lx1[up_check1==1]))
		V = np.ones(np.shape(Lx1[up_check1==1]))*-1
		
		# print(len(Lx_sub),len(L_sub))
		# L_sub = L_sub[Lx_sub < 1E11]
		# Lx_sub = Lx_sub[Lx_sub < 1E11]

		# L_sub = L_sub[Lx_sub > 1E10]
		# Lx_sub = Lx_sub[Lx_sub > 1E10]
		# print(len(Lx_sub),len(L_sub))
		# sort = np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]).argsort()



		xp, yp = bootstrap2D(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],0.85,100)
		
		# xp, yp = bootstrap2D(np.log10(Lx_sub[np.isfinite(np.log10(L_sub[Lx_sub < 1E11]))])[Lx_sub < 1E11],np.log10(L_sub[np.isfinite(np.log10(L_sub[Lx_sub < 1E11]))])[Lx_sub < 1E11],0.85,100)

		seg = np.stack((xp,yp),axis=2)


		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'


		fig = plt.figure(figsize=(8,8))
		plt.title('0.3 < z < 1.1')
		ax = plt.subplot(111)
		plt.scatter(np.log10(Lx1[up_check1==0]),np.log10(L1[up_check1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		plt.scatter(np.log10(Lx2[up_check2==0]),np.log10(L2[up_check2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		plt.scatter(np.log10(Lx3[up_check3==0]),np.log10(L3[up_check3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		plt.scatter(np.log10(Lx4[up_check4==0]),np.log10(L4[up_check4==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		plt.scatter(np.log10(Lx5[up_check5==0]),np.log10(L5[up_check5==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		# plt.quiver(np.log10(Lx1[up_check1==1]),np.log10(L1[up_check1==1]),U,V,color='b',scale=np.log10(fir_frac1[up_check1==1]*L1[up_check1==1]))
		# plt.plot([np.log10(Lx1[up_check1==1]),np.log10(Lx1[up_check1==1])],[np.log10(L1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1])],color='k',lw=3)		
		plt.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		plt.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		# plt.scatter(np.log10(Lx1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx2[up_check2==1]),np.log10(Lx2[up_check2==1])],[np.log10(L2[up_check2==1]),np.log10(L2[up_check2==1]-fir_frac2[up_check2==1])],color='k',lw=3)		
		plt.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		plt.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		
		# plt.scatter(np.log10(Lx2[up_check2==1]),np.log10(L2[up_check2==1]-fir_frac2[up_check2==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx3[up_check3==1]),np.log10(Lx3[up_check3==1])],[np.log10(L3[up_check3==1]),np.log10(L3[up_check3==1]-fir_frac3[up_check3==1])],color='k',lw=3)		
		plt.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		plt.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		# plt.scatter(np.log10(Lx3[up_check3==1]),np.log10(L3[up_check3==1]-fir_frac3[up_check3==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx4[up_check4==1]),np.log10(Lx4[up_check4==1])],[np.log10(L4[up_check4==1]),np.log10(L4[up_check4==1]-fir_frac4[up_check4==1])],color='k',lw=3)		
		plt.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		plt.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		# plt.scatter(np.log10(Lx4[up_check4==1]),np.log10(L4[up_check4==1]-fir_frac4[up_check4==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx5[up_check5==1]),np.log10(Lx5[up_check5==1])],[np.log10(L5[up_check5==1]),np.log10(L5[up_check5==1]-fir_frac5[up_check5==1])],color='k',lw=3)		
		plt.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		plt.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		# plt.scatter(np.log10(Lx5[up_check5==1]),np.log10(L5[up_check5==1]-fir_frac5[up_check5==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.scatter(np.log10(emis6),np.log10(emis5 - temp_L_bol_GOALS),color='gray',edgecolors='k',s=80,alpha=0.9,label='ULIRGs')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]),np.log10(L_sub[np.isfinite(np.log10(L_sub))]),'.',color='k')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]),y,color='k')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],y,color='cyan')

		# ls = LineCollection(seg,color='gray',alpha=0.3)
		# ax.add_collection(ls)
		# plt.plot(np.nanmedian(xp,axis=0),np.nanmedian(yp,axis=0),color='k',ls='--')
		# ax.autoscale()

		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],y2,color='k',ls='--')
		# print([np.log10(Lx1[up_check1==1]),np.log10(Lx1[up_check1==1])],[np.log10(L1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1])])
		# print(L1[up_check1==1])
		# print('FIR_L:',fir_frac1[up_check1==1])
		# print('FIR_frac1:',fir_frac1[up_check1==1]/L1[up_check1==1])
		# print('FIR_frac2:',fir_frac2[up_check2==1]/L2[up_check2==1])
		# print('FIR_frac3:',fir_frac3[up_check3==1]/L3[up_check3==1])
		# print('FIR_frac4:',fir_frac4[up_check4==1]/L4[up_check4==1])
		# print('FIR_frac5:',fir_frac5[up_check5==1]/L5[up_check5==1])
		# print('FIR_frac1:',max(fir_frac1[up_check1==1]/L1[up_check1==1]))
		# print('FIR_frac2:',max(fir_frac2[up_check2==1]/L2[up_check2==1]))
		# print('FIR_frac3:',max(fir_frac3[up_check3==1]/L3[up_check3==1]))
		# print('FIR_frac4:',max(fir_frac4[up_check4==1]/L4[up_check4==1]))
		# print('FIR_frac5:',max(fir_frac5[up_check5==1]/L5[up_check5==1]))
		# print(L1[up_check1==1]-fir_frac1[up_check1==1])
		# plt.scatter(Lx[B1],L[B1],color='b',s=80,alpha=0.9,label='Panel 1')
		# plt.scatter(Lx[B2],L[B2],color='purple',s=80,alpha=0.9,label='Panel 2')
		# plt.scatter(Lx[B3],L[B3] - temp_L_bol3,color='green',s=80,alpha=0.9,label='Panel 3')
		# plt.scatter(Lx[B4],L[B4] - temp_L_bol4,color='orange',s=80,alpha=0.9,label='Panel 4')
		# plt.scatter(Lx[B5],L[B5] - temp_L_bol5,color='red',s=80,alpha=0.9,label='Panel 5')
		# plt.scatter(emis6,emis5 - temp_L_bol_GOALS,color='gray',edgecolors='k',s=80,alpha=0.9,label='ULIRGs')
		# plt.plot(10**x,y,color='k')


		# plt.scatter(Lx[B1],np.log10(L[B1]),c=f3[B1],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='s',label='Panel 1')
		# plt.scatter(Lx[B2],np.log10(L[B2]),c=f3[B2],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='o',label='Panel 2')
		# plt.scatter(Lx[B3],np.log10(L[B3] - temp_L_bol3),c=f3[B3],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='P',label='Panel 3')
		# plt.scatter(Lx[B4],np.log10(L[B4] - temp_L_bol4),c=f3[B4],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='X',label='Panel 4')
		# plt.scatter(Lx[B5],np.log10(L[B5] - temp_L_bol5),c=f3[B5],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='D',label='Panel 5')
		# plt.scatter(emis6,np.log10(emis5 - temp_L_bol_GOALS),c=emis9,cmap='viridis',edgecolors='k',s=100,alpha=1,marker='*',label='ULIRGs')
		
		# plt.scatter(Lx,np.log10(L),c=f3,cmap='viridis')
		# plt.plot(Lx,np.log10(L),'.',color='k')
		# print(F1)
		# print(f3)

		# plt.scatter(Lx_sub,np.log10(L_sub),c=fc_sub,cmap='viridis',edgecolors='k',s=80)


		plt.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		# plt.scatter(np.nanmean(Lx[B1]),np.nanmean(np.log10(L[B1])),color='b',marker='s',s=120,label='Panel 1')
		# plt.errorbar(np.nanmean(Lx[B1]),np.nanmean(np.log10(L[B1])),np.nanstd(Lx[B1]),np.nanstd(np.log10(L[B1])),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B2]),np.nanmean(np.log10(L[B2])),color='purple',marker='s',s=120,label='Panel 2')
		# plt.errorbar(np.nanmean(Lx[B2]),np.nanmean(np.log10(L[B2])),np.nanstd(Lx[B2]),np.nanstd(np.log10(L[B2])),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B3]),np.nanmean(np.log10(L[B3] - temp_L_bol3)),color='green',marker='s',s=120,label='Panel 3')
		# plt.errorbar(np.nanmean(Lx[B3]),np.nanmean(np.log10(L[B3] - temp_L_bol3)),np.nanstd(Lx[B3]),np.nanstd(np.log10(L[B3] - temp_L_bol3)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B4]),np.nanmean(np.log10(L[B4] - temp_L_bol4)),color='orange',marker='s',s=120,label='Panel 4')
		# plt.errorbar(np.nanmean(Lx[B4]),np.nanmean(np.log10(L[B4] - temp_L_bol4)),np.nanstd(Lx[B4]),np.nanstd(np.log10(L[B4] - temp_L_bol4)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B5]),np.nanmean(np.log10(L[B5] - temp_L_bol5)),color='red',marker='s',s=120,label='Panel 5')
		# plt.errorbar(np.nanmean(Lx[B5]),np.nanmean(np.log10(L[B5] - temp_L_bol5)),np.nanstd(Lx[B5]),np.nanstd(np.log10(L[B5] - temp_L_bol5)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(emis6),np.nanmean(np.log10(emis5 - temp_L_bol_GOALS)),color='gray',marker='s',edgecolors='k',s=120,label='ULIRGs')
		# plt.errorbar(np.nanmean(emis6),np.nanmean(np.log10(emis5 - temp_L_bol_GOALS)),np.nanstd(emis6),np.nanstd(np.log10(emis5 - temp_L_bol_GOALS)),fmt='none',ecolor='gray',zorder=0.)

		# plt.scatter(emis7,np.log10(emis8)-np.log10(3.8E33),color='k',edgecolors='k',s=80,label='Swift/BAT')

		# plt.axvline(42.5,color='k',lw=3)
		# plt.colorbar(label=r'log L$_{1\mu\mathrm{m}}$ erg/s')
		secax = ax.secondary_xaxis('top',functions=(ergs, solar))
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		ax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}({0.5-10\mathrm{kev}})$ [L$_{\odot}$]')
		ax.set_ylabel(r'log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# ax.set_xlim(42,47)
		ax.set_xlim(8,13)
		ax.set_ylim(9,14)
		# plt.ylim(43,48)
		plt.legend(fontsize=14)
		# plt.xscale('log')
		# plt.yscale('log')
		plt.grid()
		plt.savefig('/Users/connor_auge/Desktop/Paper/Lbol_Lx_scatter'+savestring+'.pdf')
		plt.show()

		plt.figure(figsize=(8,8))
		ax = plt.subplot(111)
		
		plt.scatter(np.log10(Lx1),np.log10(L[B1]),color=c1,alpha=0.9,rasterized=True)
		plt.scatter(np.log10(Lx2),np.log10(L[B2]),color=c2,alpha=0.9,rasterized=True)
		plt.scatter(np.log10(Lx3),np.log10(L[B3]),color=c3,alpha=0.9,rasterized=True)
		plt.scatter(np.log10(Lx4),np.log10(L[B4]),color=c4,alpha=0.9,rasterized=True)
		plt.scatter(np.log10(Lx5),np.log10(L[B5]),color=c5,alpha=0.9,rasterized=True)
		
		plt.plot([np.log10(Lx1),np.log10(Lx1)],[np.log10(L[B1]),np.log10(L11)],color='k',alpha=0.9,rasterized=True)
		plt.plot([np.log10(Lx2),np.log10(Lx2)],[np.log10(L[B2]),np.log10(L21)],color='k',alpha=0.9,rasterized=True)
		plt.plot([np.log10(Lx3),np.log10(Lx3)],[np.log10(L[B3]),np.log10(L31)],color='k',alpha=0.9,rasterized=True)
		plt.plot([np.log10(Lx4),np.log10(Lx4)],[np.log10(L[B4]),np.log10(L41)],color='k',alpha=0.9,rasterized=True)
		plt.plot([np.log10(Lx5),np.log10(Lx5)],[np.log10(L[B5]),np.log10(L51)],color='k',alpha=0.9,rasterized=True)

		secax = ax.secondary_xaxis('top',functions=(ergs, solar))
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		ax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}({0.5-10\mathrm{kev}})$ [L$_{\odot}$]')
		ax.set_ylabel(r'log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# ax.set_xlim(42,47)
		ax.set_xlim(8,13)
		ax.set_ylim(9,14)
		# plt.ylim(43,48)
		plt.legend(fontsize=14)
		# plt.xscale('log')
		# plt.yscale('log')
		plt.grid()
		# plt.show()		
		plt.close()


	def L_Lx_scatter_5bins(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None,emis7=None,emis8=None,emis9=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None,fir_frac=None):

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3
		
		L[L > 70] = np.nan
		# L[Lx > 45.5] = np.nan


		l = np.asarray([10**i for i in L])
		# l -= (10**44.66)

		L = np.log10(l)
		L -= np.log10(3.8E33)

		emis5=[]
		emis6=[]
		emis9=[]
		F13=[]

		e5 = np.asarray([10**i for i in emis5])
		emis5 = e5
		emis5 /= 3.8E33

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		up_check1,up_check2,up_check3,up_check4,up_check5 = up_check[B1],up_check[B2],up_check[B3],up_check[B4],up_check[B5]


		# Templete scaling
		temp_L_one = 1.1963983003219803E43
		print(np.log10(temp_L_one))

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one
		scale_GOALS = np.asarray(F13)/temp_L_one

		scale = np.nanmedian(F1[B4])/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol1 = temp_L_bol*np.median(scale_3)
		temp_L_bol2 = temp_L_bol*np.median(scale_3)
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5
		temp_L_bol_GOALS = temp_L_bol*scale_GOALS

		temp_L_bol_all = temp_L_bol*scale
		temp_L_bol_all = np.nanmedian(temp_L_bol3)
		print(np.log10(temp_L_bol_all))
		# temp_L_bol_all = 10**11.548133

		f_3 = np.asarray([10**i for i in f3])
		f_3 *= F1
		f3 = np.log10(f_3)

		f_4 = np.asarray([10**i for i in f4])
		f_4 *= F1
		f4 = np.log10(f_4)

		f_1 = np.asarray([10**i for i in f1])
		f_1 *= F1
		f1 = np.log10(f_1)

		emis9 = np.log10(emis9)

		F1 = np.log10(F1)


		# l -= (tem_L_bol*)

		# temp_L_bol1 = 0
		# temp_L_bol2 = 0
		# temp_L_bol3 = 0
		# temp_L_bol4 = 0
		# temp_L_bol5 = 0
		# temp_L_bol_GOALS = 0
		# temp_L_bol_all = 0

		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)
		emis6 -= np.log10(3.8E33)
		
		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		fir_frac1 = fir_frac[B1]
		fir_frac2 = fir_frac[B2]
		fir_frac3 = fir_frac[B3]
		fir_frac4 = fir_frac[B4]
		fir_frac5 = fir_frac[B5]

		L1 = L[B1] - temp_L_bol1
		L2 = L[B2] - temp_L_bol2
		L3 = L[B3] - temp_L_bol3
		L4 = L[B4] - temp_L_bol4
		L5 = L[B5] - temp_L_bol5

		# L1 = L[B1] - temp_L_bol_all
		# L2 = L[B2] - temp_L_bol_all
		# L3 = L[B3] - temp_L_bol_all
		# L4 = L[B4] - temp_L_bol_all
		# L5 = L[B5] - temp_L_bol_all

		L_sub = np.append(L1,L2)
		L_sub = np.append(L_sub,L3)
		L_sub = np.append(L_sub,L4)
		L_sub = np.append(L_sub,L5)
		# L_sub = np.append(L_sub,emis5 - temp_L_bol_GOALS)

		lx = np.asarray([10**i for i in Lx])
		Lx = lx
		e6 = np.asarray([10**i for i in emis6])
		emis6 = e6
		
		Lx1 = Lx[B1]
		Lx2 = Lx[B2]
		Lx3 = Lx[B3]
		Lx4 = Lx[B4]
		Lx5 = Lx[B5]
		Lx_sub = np.append(Lx1,Lx2)
		Lx_sub = np.append(Lx_sub,Lx3)
		Lx_sub = np.append(Lx_sub,Lx4)
		Lx_sub = np.append(Lx_sub,Lx5)
		# Lx_sub = np.append(Lx_sub,emis6)

		fc1 = F1[B1]
		fc2 = F1[B2]
		fc3 = F1[B3]
		fc4 = F1[B4]
		fc5 = F1[B5]
		fc_sub = np.append(fc1,fc2)
		fc_sub = np.append(fc_sub,fc3)
		fc_sub = np.append(fc_sub,fc4)
		fc_sub = np.append(fc_sub,fc5)

		sort = np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]).argsort()




		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		print(np.log10(Lx1))
		print(np.log10(L1))

		xticks = [8,9,10,11,12,13]
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		fig = plt.figure(figsize=(7,21))
		gs = fig.add_gridspec(nrows=5, ncols=1)
		gs.update(hspace=0.1)
		gs.update(left=0.2,right=0.8,top=0.93,bottom=0.08)

		ax1 = plt.subplot(gs[0])
		xp1, yp1 = bootstrap2D(np.log10(Lx1[np.isfinite(np.log10(L1))]),np.log10(L1[np.isfinite(np.log10(L1))]),0.85,100)
		seg1 = np.stack((xp1,yp1),axis=2)

		ax1.scatter(np.log10(Lx1[up_check1==0]),np.log10(L1[up_check1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax1.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)
		ax1.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')
		ls1 = LineCollection(seg1,color='gray',alpha=0.3)
		ax1.add_collection(ls1)
		ax1.plot(np.nanmedian(xp1,axis=0),np.nanmedian(yp1,axis=0),color='k')
		ax1.set_xlim(8,13)
		ax1.set_ylim(10,15)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.grid()

		ax2 = plt.subplot(gs[1])
		xp2, yp2 = bootstrap2D(np.log10(Lx2[np.isfinite(np.log10(L2))]),np.log10(L2[np.isfinite(np.log10(L2))]),0.85,100)
		seg2 = np.stack((xp2,yp2),axis=2)

		ax2.scatter(np.log10(Lx2[up_check2==0]),np.log10(L2[up_check2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax2.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax2.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		
		ax2.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')
		ls2 = LineCollection(seg2,color='gray',alpha=0.3)
		ax2.add_collection(ls2)
		ax2.plot(np.nanmedian(xp2,axis=0),np.nanmedian(yp2,axis=0),color='k')
		ax2.set_xlim(8,13)
		ax2.set_ylim(10,15)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.grid()

		ax3 = plt.subplot(gs[2])
		xp3, yp3 = bootstrap2D(np.log10(Lx3[np.isfinite(np.log10(L3))]),np.log10(L3[np.isfinite(np.log10(L3))]),0.85,100)
		seg3 = np.stack((xp3,yp3),axis=2)

		ax3.scatter(np.log10(Lx3[up_check3==0]),np.log10(L3[up_check3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax3.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax3.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')
		ls3 = LineCollection(seg3,color='gray',alpha=0.3)
		ax3.add_collection(ls3)
		ax3.plot(np.nanmedian(xp3,axis=0),np.nanmedian(yp3,axis=0),color='k')
		ax3.set_xlim(8,13)
		ax3.set_ylim(10,15)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.grid()

		ax4 = plt.subplot(gs[3])
		xp4, yp4 = bootstrap2D(np.log10(Lx4[np.isfinite(np.log10(L4))]),np.log10(L4[np.isfinite(np.log10(L4))]),0.85,100)
		seg4 = np.stack((xp4,yp4),axis=2)

		ax4.scatter(np.log10(Lx4[up_check4==0]),np.log10(L4[up_check4==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax4.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		# plt.scatter(np.log10(Lx4[up_check4==1]),np.log10(L4[up_check4==1]-fir_frac4[up_check4==1]),marker='v',color='k',s=60,alpha=0.9)
		ax4.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')
		ls4 = LineCollection(seg4,color='gray',alpha=0.3)
		ax4.add_collection(ls4)
		ax4.plot(np.nanmedian(xp4,axis=0),np.nanmedian(yp4,axis=0),color='k')
		ax4.set_xlim(8,13)
		ax4.set_ylim(10,15)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.grid()

		ax5 = plt.subplot(gs[4])
		xp5, yp5 = bootstrap2D(np.log10(Lx5[np.isfinite(np.log10(L5))]),np.log10(L5[np.isfinite(np.log10(L5))]),0.85,100)
		seg5 = np.stack((xp5,yp5),axis=2)

		ax5.scatter(np.log10(Lx5[up_check5==0]),np.log10(L5[up_check5==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)
		ax5.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		ax5.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')
		ls5 = LineCollection(seg5,color='gray',alpha=0.3)
		ax5.add_collection(ls5)
		ax5.plot(np.nanmedian(xp5,axis=0),np.nanmedian(yp5,axis=0),color='k')
		ax5.set_xlim(8,13)
		ax5.set_ylim(10,15)
		ax5.set_xticks(xticks)
		ax5.grid()

		secax = ax1.secondary_xaxis('top',functions=(ergs, solar))
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		ax5.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}({0.5-10\mathrm{kev}})$ [L$_{\odot}$]')
		ax3.set_ylabel(r'log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		plt.savefig('/Users/connor_auge/Desktop/Paper/XX_5panel_Lbol_Lx_scatter'+savestring+'.pdf')
		plt.show()

	def L_Lx_scatter_6zbins(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None,emis7=None,emis8=None,emis9=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None,fir_frac=None):
		
		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		L[L > 100] = np.nan
		# Lx[Lx > 45.5] = np.nan


		l = np.asarray([10**i for i in L])

		L = np.log10(l)
		L -= np.log10(3.8E33)

		emis5=[]
		emis6=[]
		emis9=[]
		F13=[]

		e5 = np.asarray([10**i for i in emis5])
		emis5 = e5
		emis5 /= 3.8E33
		

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one
		scale_GOALS = np.asarray(F13)/temp_L_one

		scale = (np.nanmedian(F1[B4])-np.nanmedian(F1[B4])*0.5)/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5
		temp_L_bol_GOALS = temp_L_bol*scale_GOALS

		temp_L_bol_all = temp_L_bol*scale
		# temp_L_bol_all = np.nanmedian(temp_L_bol3)
		

		f_3 = np.asarray([10**i for i in f3])
		f_3 *= F1
		f3 = np.log10(f_3)

		f_4 = np.asarray([10**i for i in f4])
		f_4 *= F1
		f4 = np.log10(f_4)

		f_1 = np.asarray([10**i for i in f1])
		f_1 *= F1
		f1 = np.log10(f_1)

		emis9 = np.log10(emis9)

		F1 = np.log10(F1)

		# temp_L_bol3 = 0
		# temp_L_bol4 = 0
		# temp_L_bol5 = 0
		# temp_L_bol_GOALS = 0
		# temp_L_bol_all = 0

		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)
		emis6 -= np.log10(3.8E33)

		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		fir_frac1 = fir_frac[B1]
		fir_frac2 = fir_frac[B2]
		fir_frac3 = fir_frac[B3]
		fir_frac4 = fir_frac[B4]
		fir_frac5 = fir_frac[B5]

		# L11 = L[B1] - temp_L_bol_all
		# L21 = L[B2] - temp_L_bol_all
		# L31 = L[B3] - temp_L_bol_all
		# L41 = L[B4] - temp_L_bol_all
		# L51 = L[B5] - temp_L_bol_all
		# print(np.log10(temp_L_bol_all))

		temp_L_bol_all = 1.44124431E11


		print(np.log10(temp_L_bol_all))


		lx = np.asarray([10**i for i in Lx])
		Lx = lx

		L1_1 = L[B1[(z1 < 0.5)&(z1 > 0.3)]]
		L2_1 = L[B2[(z2 < 0.5)&(z2 > 0.3)]]
		L3_1 = L[B3[(z3 < 0.5)&(z3 > 0.3)]]
		L4_1 = L[B4[(z4 < 0.5)&(z4 > 0.3)]]
		L5_1 = L[B5[(z5 < 0.5)&(z5 > 0.3)]]

		Lx1_1 = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]]
		Lx2_1 = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]]
		Lx3_1 = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]]
		Lx4_1 = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]]
		Lx5_1 = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]]

		fir_frac1_1 = fir_frac[B1[(z1 < 0.5)&(z1 > 0.3)]]
		fir_frac2_1 = fir_frac[B2[(z2 < 0.5)&(z2 > 0.3)]]
		fir_frac3_1 = fir_frac[B3[(z3 < 0.5)&(z3 > 0.3)]]
		fir_frac4_1 = fir_frac[B4[(z4 < 0.5)&(z4 > 0.3)]]
		fir_frac5_1 = fir_frac[B5[(z5 < 0.5)&(z5 > 0.3)]]

		up_check1_1,up_check2_1,up_check3_1,up_check4_1,up_check5_1 = up_check[B1[(z1 < 0.5)&(z1 > 0.3)]],up_check[B2[(z2 < 0.5)&(z2 > 0.3)]],up_check[B3[(z3 < 0.5)&(z3 > 0.3)]],up_check[B4[(z4 < 0.5)&(z4 > 0.3)]],up_check[B5[(z5 < 0.5)&(z5 > 0.3)]]


		L1_2 = L[B1[(z1 < 0.8)&(z1 > 0.6)]]
		L2_2 = L[B2[(z2 < 0.8)&(z2 > 0.6)]]
		L3_2 = L[B3[(z3 < 0.8)&(z3 > 0.6)]]
		L4_2 = L[B4[(z4 < 0.8)&(z4 > 0.6)]]
		L5_2 = L[B5[(z5 < 0.8)&(z5 > 0.6)]]

		Lx1_2 = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]]
		Lx2_2 = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]]
		Lx3_2 = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]]
		Lx4_2 = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]]
		Lx5_2 = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]]

		fir_frac1_2 = fir_frac[B1[(z1 < 0.8)&(z1 > 0.6)]]
		fir_frac2_2 = fir_frac[B2[(z2 < 0.8)&(z2 > 0.6)]]
		fir_frac3_2 = fir_frac[B3[(z3 < 0.8)&(z3 > 0.6)]]
		fir_frac4_2 = fir_frac[B4[(z4 < 0.8)&(z4 > 0.6)]]
		fir_frac5_2 = fir_frac[B5[(z5 < 0.8)&(z5 > 0.6)]]

		up_check1_2,up_check2_2,up_check3_2,up_check4_2,up_check5_2 = up_check[B1[(z1 < 0.8)&(z1 > 0.6)]],up_check[B2[(z2 < 0.8)&(z2 > 0.6)]],up_check[B3[(z3 < 0.8)&(z3 > 0.6)]],up_check[B4[(z4 < 0.8)&(z4 > 0.6)]],up_check[B5[(z5 < 0.8)&(z5 > 0.6)]]


		L1_3 = L[B1[(z1 < 1.2)&(z1 > 0.9)]]
		L2_3 = L[B2[(z2 < 1.2)&(z2 > 0.9)]]
		L3_3 = L[B3[(z3 < 1.2)&(z3 > 0.9)]]
		L4_3 = L[B4[(z4 < 1.2)&(z4 > 0.9)]]
		L5_3 = L[B5[(z5 < 1.2)&(z5 > 0.9)]]

		Lx1_3 = Lx[B1[(z1 < 1.2)&(z1 > 0.9)]]
		Lx2_3 = Lx[B2[(z2 < 1.2)&(z2 > 0.9)]]
		Lx3_3 = Lx[B3[(z3 < 1.2)&(z3 > 0.9)]]
		Lx4_3 = Lx[B4[(z4 < 1.2)&(z4 > 0.9)]]
		Lx5_3 = Lx[B5[(z5 < 1.2)&(z5 > 0.9)]]

		fir_frac1_3 = fir_frac[B1[(z1 < 1.2)&(z1 > 0.9)]]
		fir_frac2_3 = fir_frac[B2[(z2 < 1.2)&(z2 > 0.9)]]
		fir_frac3_3 = fir_frac[B3[(z3 < 1.2)&(z3 > 0.9)]]
		fir_frac4_3 = fir_frac[B4[(z4 < 1.2)&(z4 > 0.9)]]
		fir_frac5_3 = fir_frac[B5[(z5 < 1.2)&(z5 > 0.9)]]


		up_check1_3,up_check2_3,up_check3_3,up_check4_3,up_check5_3 = up_check[B1[(z1 < 1.2)&(z1 > 0.9)]],up_check[B2[(z2 < 1.2)&(z2 > 0.9)]],up_check[B3[(z3 < 1.2)&(z3 > 0.9)]],up_check[B4[(z4 < 1.2)&(z4 > 0.9)]],up_check[B5[(z5 < 1.2)&(z5 > 0.9)]]


		L1_1_sub = L[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		L2_1_sub = L[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		L3_1_sub = L[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		L4_1_sub = L[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		L5_1_sub = L[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		# Lx1_1_sub = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		# Lx2_1_sub = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		# Lx3_1_sub = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		# Lx4_1_sub = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		# Lx5_1_sub = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		L1_2_sub = L[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		L2_2_sub = L[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		L3_2_sub = L[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		L4_2_sub = L[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		L5_2_sub = L[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		# Lx1_2_sub = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		# Lx2_2_sub = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		# Lx3_2_sub = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		# Lx4_2_sub = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		# Lx5_2_sub = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		L1_3_sub = L[B1[(z1 < 1.2)&(z1 > 0.9)]] - temp_L_bol_all
		L2_3_sub = L[B2[(z2 < 1.2)&(z2 > 0.9)]] - temp_L_bol_all
		L3_3_sub = L[B3[(z3 < 1.2)&(z3 > 0.9)]] - temp_L_bol_all
		L4_3_sub = L[B4[(z4 < 1.2)&(z4 > 0.9)]] - temp_L_bol_all
		L5_3_sub = L[B5[(z5 < 1.2)&(z5 > 0.9)]] - temp_L_bol_all

		# Lx1_3_sub = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		# Lx2_3_sub = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		# Lx3_3_sub = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		# Lx4_3_sub = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		# Lx5_3_sub = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all



		# L_sub = np.append(L1,L2)
		# L_sub = np.append(L_sub,L3)
		# L_sub = np.append(L_sub,L4)
		# L_sub = np.append(L_sub,L5)

		# lx = np.asarray([10**i for i in Lx])
		# Lx = lx
		# e6 = np.asarray([10**i for i in emis6])
		# emis6 = e6
		
		# Lx1 = Lx[B1]
		# Lx2 = Lx[B2]
		# Lx3 = Lx[B3]
		# Lx4 = Lx[B4]
		# Lx5 = Lx[B5]
		# Lx_sub = np.append(Lx1,Lx2)
		# Lx_sub = np.append(Lx_sub,Lx3)
		# Lx_sub = np.append(Lx_sub,Lx4)
		# Lx_sub = np.append(Lx_sub,Lx5)

		# fc1 = F1[B1]
		# fc2 = F1[B2]
		# fc3 = F1[B3]
		# fc4 = F1[B4]
		# fc5 = F1[B5]
		# fc_sub = np.append(fc1,fc2)
		# fc_sub = np.append(fc_sub,fc3)
		# fc_sub = np.append(fc_sub,fc4)
		# fc_sub = np.append(fc_sub,fc5)

		# sort = np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]).argsort()

		# z = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],1)
		# p = np.poly1d(z)

		# z2 = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],2)
		# p2 = np.poly1d(z2)

		# x = np.linspace(8,15,20)
		# y = p(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort])

		# U = np.zeros(np.shape(Lx1[up_check1==1]))
		# V = np.ones(np.shape(Lx1[up_check1==1]))*-1



		# xp, yp = bootstrap2D(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],0.85,100)
		# seg = np.stack((xp,yp),axis=2)


		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'


		fig = plt.figure(figsize=(20,12))
		gs = fig.add_gridspec(nrows=2, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.05,right=0.95,top=0.9,bottom=0.15)

		ax1 = plt.subplot(gs[0,0])
		ax1.set_title('0.3 < z < 0.5')
		ax1.scatter(np.log10(Lx1_1[up_check1_1==0]),np.log10(L1_1[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax1.scatter(np.log10(Lx2_1[up_check2_1==0]),np.log10(L2_1[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax1.scatter(np.log10(Lx3_1[up_check3_1==0]),np.log10(L3_1[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax1.scatter(np.log10(Lx4_1[up_check4_1==0]),np.log10(L4_1[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax1.scatter(np.log10(Lx5_1[up_check5_1==0]),np.log10(L5_1[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax1.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax1.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax1.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax1 = ax1.secondary_xaxis('top',functions=(ergs, solar))
		secax1.set_xlabel(r' ')
		ax1.set_ylabel(r'Total log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		ax1.set_xticklabels([])
		ax1.set_xlim(8,13)
		ax1.set_ylim(9,14)

		ax2 = plt.subplot(gs[1,0])
		ax2.scatter(np.log10(Lx1_1[up_check1_1==0]),np.log10(L1_1_sub[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax2.scatter(np.log10(Lx2_1[up_check2_1==0]),np.log10(L2_1_sub[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax2.scatter(np.log10(Lx3_1[up_check3_1==0]),np.log10(L3_1_sub[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax2.scatter(np.log10(Lx4_1[up_check4_1==0]),np.log10(L4_1_sub[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax2.scatter(np.log10(Lx5_1[up_check5_1==0]),np.log10(L5_1_sub[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax2.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),np.log10(L1_1_sub[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax2.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),np.log10(L1_1_sub[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax2.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),np.log10(L2_1_sub[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax2.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),np.log10(L2_1_sub[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),np.log10(L3_1_sub[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),np.log10(L3_1_sub[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),np.log10(L4_1_sub[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),np.log10(L4_1_sub[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),np.log10(L5_1_sub[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),np.log10(L5_1_sub[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax2.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax2 = ax2.secondary_xaxis('top',functions=(ergs, solar))
		ax2.set_ylabel(r'Galaxy Subtracted log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		secax2.set_xticklabels([])
		ax2.set_xlim(8,13)
		ax2.set_ylim(9,14)


		ax3 = plt.subplot(gs[0,1])
		ax3.set_title('0.6 < z < 0.8')
		ax3.scatter(np.log10(Lx1_2[up_check1_2==0]),np.log10(L1_2[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax3.scatter(np.log10(Lx2_2[up_check2_2==0]),np.log10(L2_2[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax3.scatter(np.log10(Lx3_2[up_check3_2==0]),np.log10(L3_2[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax3.scatter(np.log10(Lx4_2[up_check4_2==0]),np.log10(L4_2[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax3.scatter(np.log10(Lx5_2[up_check5_2==0]),np.log10(L5_2[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax3.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax3.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax3.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax3 = ax3.secondary_xaxis('top',functions=(ergs, solar))
		secax3.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		ax3.set_xticklabels([])
		ax3.set_yticklabels([])
		ax3.set_xlim(8,13)
		ax3.set_ylim(9,14)


		ax4 = plt.subplot(gs[1,1])
		ax4.scatter(np.log10(Lx1_2[up_check1_2==0]),np.log10(L1_2_sub[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax4.scatter(np.log10(Lx2_2[up_check2_2==0]),np.log10(L2_2_sub[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax4.scatter(np.log10(Lx3_2[up_check3_2==0]),np.log10(L3_2_sub[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax4.scatter(np.log10(Lx4_2[up_check4_2==0]),np.log10(L4_2_sub[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax4.scatter(np.log10(Lx5_2[up_check5_2==0]),np.log10(L5_2_sub[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax4.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),np.log10(L1_2_sub[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax4.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),np.log10(L1_2_sub[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax4.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),np.log10(L2_2_sub[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax4.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),np.log10(L2_2_sub[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),np.log10(L3_2_sub[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),np.log10(L3_2_sub[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),np.log10(L4_2_sub[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),np.log10(L4_2_sub[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),np.log10(L5_2_sub[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),np.log10(L5_2_sub[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax4.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax4 = ax4.secondary_xaxis('top',functions=(ergs, solar))
		ax4.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}({0.5-10\mathrm{kev}})$ [L$_{\odot}$]')
		ax4.set_yticklabels([])
		secax4.set_xticklabels([])
		ax4.set_xlim(8,13)
		ax4.set_ylim(9,14)


		ax5 = plt.subplot(gs[0,2])
		ax5.set_title('0.9 < z < 1.1')
		ax5.scatter(np.log10(Lx1_3[up_check1_3==0]),np.log10(L1_3[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax5.scatter(np.log10(Lx2_3[up_check2_3==0]),np.log10(L2_3[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax5.scatter(np.log10(Lx3_3[up_check3_3==0]),np.log10(L3_3[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax5.scatter(np.log10(Lx4_3[up_check4_3==0]),np.log10(L4_3[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax5.scatter(np.log10(Lx5_3[up_check5_3==0]),np.log10(L5_3[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax5.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax5.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax5.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax5 = ax5.secondary_xaxis('top',functions=(ergs, solar))
		secax5.set_xlabel(r' ')
		ax5.set_xticklabels([])
		ax5.set_yticklabels([])
		ax5.set_xlim(8,13)
		ax5.set_ylim(9,14)

		ax6 = plt.subplot(gs[1,2])
		ax6.scatter(np.log10(Lx1_3[up_check1_3==0]),np.log10(L1_3_sub[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax6.scatter(np.log10(Lx2_3[up_check2_3==0]),np.log10(L2_3_sub[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax6.scatter(np.log10(Lx3_3[up_check3_3==0]),np.log10(L3_3_sub[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax6.scatter(np.log10(Lx4_3[up_check4_3==0]),np.log10(L4_3_sub[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax6.scatter(np.log10(Lx5_3[up_check5_3==0]),np.log10(L5_3_sub[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax6.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),np.log10(L1_3_sub[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax6.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),np.log10(L1_3_sub[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax6.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),np.log10(L2_3_sub[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax6.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),np.log10(L2_3_sub[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),np.log10(L3_3_sub[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),np.log10(L3_3_sub[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),np.log10(L4_3_sub[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),np.log10(L4_3_sub[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),np.log10(L5_3_sub[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),np.log10(L5_3_sub[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax6.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')

		secax6 = ax6.secondary_xaxis('top',functions=(ergs, solar))
		ax6.set_yticklabels([])
		secax6.set_xticklabels([])
		ax6.set_xlim(8,13)
		ax6.set_ylim(9,14)

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()
		ax6.grid()
		plt.savefig('/Users/connor_auge/Desktop/Paper/Lx_425/final/Lbol_Lx_scatter_6zbins'+savestring+'.pdf')
		plt.show()


		
	def L_Lx_scatter_3zbins(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None,emis7=None,emis8=None,emis9=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None,fir_frac=None):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3


		# Lx[Lx > 45.5] = np.nan


		# L = np.log10(l)
		L -= np.log10(3.8E33)
		L[L > 22] = np.nan



		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.1

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one

		scale = (np.nanmedian(F1[B4])-np.nanmedian(F1[B4])*0.5)/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5

		temp_L_bol_all = temp_L_bol*scale
		# temp_L_bol_all = np.nanmedian(temp_L_bol3)
		
		F1 = np.log10(F1)

		temp_L_bol3 = 0
		temp_L_bol4 = 0
		temp_L_bol5 = 0
		temp_L_bol_GOALS = 0
		temp_L_bol_all = 0


		l = np.asarray([10**i for i in L])
		L = l

		Lx -= np.log10(3.8E33)


		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		# fir_frac1 = fir_frac[B1]
		# fir_frac2 = fir_frac[B2]
		# fir_frac3 = fir_frac[B3]
		# fir_frac4 = fir_frac[B4]
		# fir_frac5 = fir_frac[B5]

		# L11 = L[B1] - temp_L_bol_all
		# L21 = L[B2] - temp_L_bol_all
		# L31 = L[B3] - temp_L_bol_all
		# L41 = L[B4] - temp_L_bol_all
		# L51 = L[B5] - temp_L_bol_all
		# print(np.log10(temp_L_bol_all))

		# temp_L_bol_all = 1.44124431E11


		# print(np.log10(temp_L_bol_all))


		lx = np.asarray([10**i for i in Lx])
		Lx = lx

		L1_1 = L[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]]
		L2_1 = L[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]]
		L3_1 = L[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]]
		L4_1 = L[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]]
		L5_1 = L[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]]

		Lx1_1 = Lx[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]]
		Lx2_1 = Lx[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]]
		Lx3_1 = Lx[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]]
		Lx4_1 = Lx[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]]
		Lx5_1 = Lx[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]]

		fir_frac1_1 = fir_frac[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]]
		fir_frac2_1 = fir_frac[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]]
		fir_frac3_1 = fir_frac[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]]
		fir_frac4_1 = fir_frac[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]]
		fir_frac5_1 = fir_frac[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]]

		up_check1_1,up_check2_1,up_check3_1,up_check4_1,up_check5_1 = up_check[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]],up_check[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]],up_check[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]],up_check[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]],up_check[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]]


		L1_2 = L[B1[(z1 <= zlim_3)&(z1 > zlim_2)]]
		L2_2 = L[B2[(z2 <= zlim_3)&(z2 > zlim_2)]]
		L3_2 = L[B3[(z3 <= zlim_3)&(z3 > zlim_2)]]
		L4_2 = L[B4[(z4 <= zlim_3)&(z4 > zlim_2)]]
		L5_2 = L[B5[(z5 <= zlim_3)&(z5 > zlim_2)]]

		Lx1_2 = Lx[B1[(z1 <= zlim_3)&(z1 > zlim_2)]]
		Lx2_2 = Lx[B2[(z2 <= zlim_3)&(z2 > zlim_2)]]
		Lx3_2 = Lx[B3[(z3 <= zlim_3)&(z3 > zlim_2)]]
		Lx4_2 = Lx[B4[(z4 <= zlim_3)&(z4 > zlim_2)]]
		Lx5_2 = Lx[B5[(z5 <= zlim_3)&(z5 > zlim_2)]]

		fir_frac1_2 = fir_frac[B1[(z1 <= zlim_3)&(z1 > zlim_2)]]
		fir_frac2_2 = fir_frac[B2[(z2 <= zlim_3)&(z2 > zlim_2)]]
		fir_frac3_2 = fir_frac[B3[(z3 <= zlim_3)&(z3 > zlim_2)]]
		fir_frac4_2 = fir_frac[B4[(z4 <= zlim_3)&(z4 > zlim_2)]]
		fir_frac5_2 = fir_frac[B5[(z5 <= zlim_3)&(z5 > zlim_2)]]

		up_check1_2,up_check2_2,up_check3_2,up_check4_2,up_check5_2 = up_check[B1[(z1 <= zlim_3)&(z1 > zlim_2)]],up_check[B2[(z2 <= zlim_3)&(z2 > zlim_2)]],up_check[B3[(z3 <= zlim_3)&(z3 > zlim_2)]],up_check[B4[(z4 <= zlim_3)&(z4 > zlim_2)]],up_check[B5[(z5 <= zlim_3)&(z5 > zlim_2)]]


		L1_3 = L[B1[(z1 <= zlim_4)&(z1 > zlim_3)]]
		L2_3 = L[B2[(z2 <= zlim_4)&(z2 > zlim_3)]]
		L3_3 = L[B3[(z3 <= zlim_4)&(z3 > zlim_3)]]
		L4_3 = L[B4[(z4 <= zlim_4)&(z4 > zlim_3)]]
		L5_3 = L[B5[(z5 <= zlim_4)&(z5 > zlim_3)]]

		Lx1_3 = Lx[B1[(z1 <= zlim_4)&(z1 > zlim_3)]]
		Lx2_3 = Lx[B2[(z2 <= zlim_4)&(z2 > zlim_3)]]
		Lx3_3 = Lx[B3[(z3 <= zlim_4)&(z3 > zlim_3)]]
		Lx4_3 = Lx[B4[(z4 <= zlim_4)&(z4 > zlim_3)]]
		Lx5_3 = Lx[B5[(z5 <= zlim_4)&(z5 > zlim_3)]]

		fir_frac1_3 = fir_frac[B1[(z1 <= zlim_4)&(z1 > zlim_3)]]
		fir_frac2_3 = fir_frac[B2[(z2 <= zlim_4)&(z2 > zlim_3)]]
		fir_frac3_3 = fir_frac[B3[(z3 <= zlim_4)&(z3 > zlim_3)]]
		fir_frac4_3 = fir_frac[B4[(z4 <= zlim_4)&(z4 > zlim_3)]]
		fir_frac5_3 = fir_frac[B5[(z5 <= zlim_4)&(z5 > zlim_3)]]


		up_check1_3,up_check2_3,up_check3_3,up_check4_3,up_check5_3 = up_check[B1[(z1 <= zlim_4)&(z1 > zlim_3)]],up_check[B2[(z2 <= zlim_4)&(z2 > zlim_3)]],up_check[B3[(z3 <= zlim_4)&(z3 > zlim_3)]],up_check[B4[(z4 <= zlim_4)&(z4 > zlim_3)]],up_check[B5[(z5 <= zlim_4)&(z5 > zlim_3)]]


		L1_1_sub = L[B1[(z1 <= zlim_2)&(z1 >= zlim_1)]] - temp_L_bol_all
		L2_1_sub = L[B2[(z2 <= zlim_2)&(z2 >= zlim_1)]] - temp_L_bol_all
		L3_1_sub = L[B3[(z3 <= zlim_2)&(z3 >= zlim_1)]] - temp_L_bol_all
		L4_1_sub = L[B4[(z4 <= zlim_2)&(z4 >= zlim_1)]] - temp_L_bol_all
		L5_1_sub = L[B5[(z5 <= zlim_2)&(z5 >= zlim_1)]] - temp_L_bol_all

		# Lx1_1_sub = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		# Lx2_1_sub = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		# Lx3_1_sub = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		# Lx4_1_sub = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		# Lx5_1_sub = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		L1_2_sub = L[B1[(z1 <= zlim_3)&(z1 > zlim_2)]] - temp_L_bol_all
		L2_2_sub = L[B2[(z2 <= zlim_3)&(z2 > zlim_2)]] - temp_L_bol_all
		L3_2_sub = L[B3[(z3 <= zlim_3)&(z3 > zlim_2)]] - temp_L_bol_all
		L4_2_sub = L[B4[(z4 <= zlim_3)&(z4 > zlim_2)]] - temp_L_bol_all
		L5_2_sub = L[B5[(z5 <= zlim_3)&(z5 > zlim_2)]] - temp_L_bol_all

		# Lx1_2_sub = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		# Lx2_2_sub = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		# Lx3_2_sub = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		# Lx4_2_sub = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		# Lx5_2_sub = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		L1_3_sub = L[B1[(z1 <= zlim_4)&(z1 > zlim_3)]] - temp_L_bol_all
		L2_3_sub = L[B2[(z2 <= zlim_4)&(z2 > zlim_3)]] - temp_L_bol_all
		L3_3_sub = L[B3[(z3 <= zlim_4)&(z3 > zlim_3)]] - temp_L_bol_all
		L4_3_sub = L[B4[(z4 <= zlim_4)&(z4 > zlim_3)]] - temp_L_bol_all
		L5_3_sub = L[B5[(z5 <= zlim_4)&(z5 > zlim_3)]] - temp_L_bol_all

		# Lx1_3_sub = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		# Lx2_3_sub = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		# Lx3_3_sub = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		# Lx4_3_sub = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		# Lx5_3_sub = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all



		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'



		fig = plt.figure(figsize=(18, 6.5))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.07,right=0.92,top=0.83,bottom=0.12)



		ax1 = plt.subplot(gs[0])

		xp = np.linspace(8.5,12)
		fit_z1 = np.polyfit(np.log10(Lx[(spec_z <= zlim_2) & (spec_z >= zlim_1)]),np.log10(L[(spec_z <= zlim_2) & (spec_z >= zlim_1)]),1)
		fit_p1 = np.poly1d(fit_z1)

		fit_z12 = np.polyfit(np.log10(np.asarray([np.nanmedian(Lx1_1),np.nanmedian(Lx2_1),np.nanmedian(Lx3_1),np.nanmedian(Lx4_1),np.nanmedian(Lx5_1)])),
		np.log10(np.asarray([np.nanmedian(L1_1),np.nanmedian(L2_1),np.nanmedian(L3_1),np.nanmedian(L4_1),np.nanmedian(L5_1)])),1)
		fit_p12 = np.poly1d(fit_z12)


		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		ax1.scatter(np.log10(Lx1_1[up_check1_1==0]),np.log10(L1_1[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.65,label='Panel 1',rasterized=True)
		ax1.scatter(np.log10(Lx2_1[up_check2_1==0]),np.log10(L2_1[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.65,label='Panel 2',rasterized=True)
		ax1.scatter(np.log10(Lx3_1[up_check3_1==0]),np.log10(L3_1[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.65,label='Panel 3',rasterized=True)
		ax1.scatter(np.log10(Lx4_1[up_check4_1==0]),np.log10(L4_1[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.65,label='Panel 4',rasterized=True)
		ax1.scatter(np.log10(Lx5_1[up_check5_1==0]),np.log10(L5_1[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.65,label='Panel 5',rasterized=True)

		ax1.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.65)
		ax1.scatter(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.65)

		ax1.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.65)
		ax1.scatter(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.65)		

		ax1.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.65)		
		ax1.scatter(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.65)		

		ax1.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.65)		
		ax1.scatter(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.65)		

		ax1.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.65)		
		ax1.scatter(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.65)

		ax1.scatter(np.nanmedian(np.log10(Lx1_1)),np.nanmedian(np.log10(L1_1)),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(np.log10(Lx2_1)),np.nanmedian(np.log10(L2_1)),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(np.log10(Lx3_1)),np.nanmedian(np.log10(L3_1)),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(np.log10(Lx4_1)),np.nanmedian(np.log10(L4_1)),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(np.log10(Lx5_1)),np.nanmedian(np.log10(L5_1)),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)

		ax1.plot(xp,fit_p1(xp),color='k')
		ax1.plot(xp,fit_p12(xp),'--',color='k')

		# ax1.scatter([np.nanmedian(Lx1_1),np.nanmedian(Lx2_1),np.nanmedian(Lx3_1),np.nanmedian(Lx4_1),np.nanmedian(Lx5_1)],[np.nanmedian(L1_1),np.nanmedian(L2_1),np.nanmedian(L3_1),np.nanmedian(L4_1),np.nanmedian(L5_1))])
		# ax1.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax1.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')

		

		secax1 = ax1.secondary_xaxis('top',functions=(ergs, solar))
		secax1.set_xlabel(r' ')
		ax1.set_ylabel(r'Total log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# # ax1.set_xticklabels([])
		ax1.set_xlim(8,13)
		ax1.set_ylim(10,14)
		ax1.set_yticks([10,11,12,13,14])



		ax3 = plt.subplot(gs[1])


		fit_z2 = np.polyfit(np.log10(Lx[(spec_z <= zlim_3) & (spec_z > zlim_2)]),np.log10(L[(spec_z <= zlim_3) & (spec_z > zlim_2)]),1)
		fit_p2 = np.poly1d(fit_z2)

		fit_z22 = np.polyfit(np.log10(np.asarray([np.nanmedian(Lx1_2),np.nanmedian(Lx2_2),np.nanmedian(Lx3_2),np.nanmedian(Lx4_2),np.nanmedian(Lx5_2)])),
		np.log10(np.asarray([np.nanmedian(L1_2),np.nanmedian(L2_2),np.nanmedian(L3_2),np.nanmedian(L4_2),np.nanmedian(L5_2)])),1)
		fit_p22 = np.poly1d(fit_z22)


		ax3.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		ax3.scatter(np.log10(Lx1_2[up_check1_2==0]),np.log10(L1_2[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.65,label='Panel 1',rasterized=True)
		ax3.scatter(np.log10(Lx2_2[up_check2_2==0]),np.log10(L2_2[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.65,label='Panel 2',rasterized=True)
		ax3.scatter(np.log10(Lx3_2[up_check3_2==0]),np.log10(L3_2[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.65,label='Panel 3',rasterized=True)
		ax3.scatter(np.log10(Lx4_2[up_check4_2==0]),np.log10(L4_2[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.65,label='Panel 4',rasterized=True)
		ax3.scatter(np.log10(Lx5_2[up_check5_2==0]),np.log10(L5_2[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.65,label='Panel 5',rasterized=True)

		ax3.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.65)
		ax3.scatter(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.65)

		ax3.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.65)
		ax3.scatter(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.65)		

		ax3.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.65)		
		ax3.scatter(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.65)		

		ax3.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.65)		
		ax3.scatter(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.65)		

		ax3.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.65)		
		ax3.scatter(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.65)
		

		ax3.scatter(np.nanmedian(np.log10(Lx1_2)),np.nanmedian(np.log10(L1_2)),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(np.log10(Lx2_2)),np.nanmedian(np.log10(L2_2)),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(np.log10(Lx3_2)),np.nanmedian(np.log10(L3_2)),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(np.log10(Lx4_2)),np.nanmedian(np.log10(L4_2)),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(np.log10(Lx5_2)),np.nanmedian(np.log10(L5_2)),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)
		# ax3.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax3.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		ax3.plot(xp,fit_p2(xp),color='k')
		ax3.plot(xp,fit_p22(xp),'--',color='k')

		secax3 = ax3.secondary_xaxis('top',functions=(ergs, solar))
		secax3.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		ax3.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [L$_{\odot}$]')
		# ax3.set_xticklabels([])
		ax3.set_yticklabels([])
		ax3.set_xlim(8,13)
		ax3.set_ylim(10,14)
		ax3.set_yticks([10,11,12,13,14])



		ax5 = plt.subplot(gs[2])

		fit_z3 = np.polyfit(np.log10(Lx[(spec_z <= zlim_4) & (spec_z > zlim_3)]),np.log10(L[(spec_z <= zlim_4) & (spec_z > zlim_3)]),1)
		fit_p3 = np.poly1d(fit_z3)

		fit_z32 = np.polyfit(np.log10(np.asarray([np.nanmedian(Lx1_3),np.nanmedian(Lx2_3),np.nanmedian(Lx3_3),np.nanmedian(Lx4_3),np.nanmedian(Lx5_3)])),
		np.log10(np.asarray([np.nanmedian(L1_3),np.nanmedian(L2_3),np.nanmedian(L3_3),np.nanmedian(L4_3),np.nanmedian(L5_3)])),1)
		fit_p32 = np.poly1d(fit_z32)


		ax5.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		ax5.scatter(np.log10(Lx1_3[up_check1_3==0]),np.log10(L1_3[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.65,label='Panel 1',rasterized=True)
		ax5.scatter(np.log10(Lx2_3[up_check2_3==0]),np.log10(L2_3[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.65,label='Panel 2',rasterized=True)
		ax5.scatter(np.log10(Lx3_3[up_check3_3==0]),np.log10(L3_3[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.65,label='Panel 3',rasterized=True)
		ax5.scatter(np.log10(Lx4_3[up_check4_3==0]),np.log10(L4_3[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.65,label='Panel 4',rasterized=True)
		ax5.scatter(np.log10(Lx5_3[up_check5_3==0]),np.log10(L5_3[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.65,label='Panel 5',rasterized=True)

		ax5.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.65)
		ax5.scatter(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.65)

		ax5.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.65)
		ax5.scatter(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.65)		

		ax5.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.65)		
		ax5.scatter(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.65)		

		ax5.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.65)		
		ax5.scatter(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.65)		

		ax5.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.65)		
		ax5.scatter(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.65)
		
		ax5.scatter(np.nanmedian(np.log10(Lx1_3)),np.nanmedian(np.log10(L1_3)),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax5.scatter(np.nanmedian(np.log10(Lx2_3)),np.nanmedian(np.log10(L2_3)),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax5.scatter(np.nanmedian(np.log10(Lx3_3)),np.nanmedian(np.log10(L3_3)),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax5.scatter(np.nanmedian(np.log10(Lx4_3)),np.nanmedian(np.log10(L4_3)),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax5.scatter(np.nanmedian(np.log10(Lx5_3)),np.nanmedian(np.log10(L5_3)),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)
		# ax5.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax5.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')

		ax5.plot(xp,fit_p3(xp),color='k')
		ax5.plot(xp,fit_p32(xp),'--',color='k')

		secax5 = ax5.secondary_xaxis('top',functions=(ergs, solar))
		secax5.set_xlabel(r' ')
		secax5 = ax5.secondary_yaxis('right',functions=(ergs, solar))
		secax5.set_ylabel(r'log $\mathrm{L}_{\mathrm{bol}}$ [L$_\odot$]')
		# ax5.scatter(-100,100,marker='P',color='k',s=250,label='Median',rasterized=True)
		# ax5.legend()
		secax5.set_yticks([44,45,46,47])
		# ax5.set_xticklabels([])
		ax5.set_yticklabels([])
		ax5.set_yticks([10,11,12,13,14])
		ax5.set_xlim(8,13)
		ax5.set_ylim(10,14)

		ax1.grid()
		ax3.grid()
		ax5.grid()
		plt.savefig('/Users/connor_auge/Desktop/New_runSED/'+savestring+'.pdf')
		plt.show()

	def L_hist_3zbins(self,savestring,L,F1,spec_z,uv_slope,mir_slope1,mir_slope2):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3]
		L4 = L[B4]
		L5 = L[B5]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		bin1 = 42
		bin2 = 46.5
		binsize=0.25

		fig = plt.figure(figsize=(21,7))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.07,right=0.93,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		ax1.set_title('0.3 < z < 0.5')
		ax1.hist(L1[(z1 > 0.3)&(z1 < 0.5)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c1)
		ax1.hist(L2[(z2 > 0.3)&(z2 < 0.5)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c2)
		ax1.hist(L3[(z3 > 0.3)&(z3 < 0.5)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c3)
		ax1.hist(L4[(z4 > 0.3)&(z4 < 0.5)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c4)
		ax1.hist(L5[(z5 > 0.3)&(z5 < 0.5)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c5)
		ax1.axvline(np.nanmedian(L1[(z1 > 0.3)&(z1 < 0.5)]),ls='--',lw=2.5,color=c1)
		ax1.axvline(np.nanmedian(L2[(z2 > 0.3)&(z2 < 0.5)]),ls='--',lw=2.5,color=c2)
		ax1.axvline(np.nanmedian(L3[(z3 > 0.3)&(z3 < 0.5)]),ls='--',lw=2.5,color=c3)
		ax1.axvline(np.nanmedian(L4[(z4 > 0.3)&(z4 < 0.5)]),ls='--',lw=2.5,color=c4)
		ax1.axvline(np.nanmedian(L5[(z5 > 0.3)&(z5 < 0.5)]),ls='--',lw=2.5,color=c5)
		ax1.set_xlim(42,46.5)
		ax1.set_ylim(0,40)
		ax1.set_xlabel(r'log L$_{\mathrm{X}}$')


		ax2 = plt.subplot(gs[1])
		ax2.set_title('0.6 < z < 0.8')
		ax2.hist(L1[(z1 > 0.6)&(z1 < 0.8)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c1)
		ax2.hist(L2[(z2 > 0.6)&(z2 < 0.8)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c2)
		ax2.hist(L3[(z3 > 0.6)&(z3 < 0.8)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c3)
		ax2.hist(L4[(z4 > 0.6)&(z4 < 0.8)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c4)
		ax2.hist(L5[(z5 > 0.6)&(z5 < 0.8)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c5)
		ax2.axvline(np.nanmedian(L1[(z1 > 0.6)&(z1 < 0.8)]),ls='--',lw=2.5,color=c1)
		ax2.axvline(np.nanmedian(L2[(z2 > 0.6)&(z2 < 0.8)]),ls='--',lw=2.5,color=c2)
		ax2.axvline(np.nanmedian(L3[(z3 > 0.6)&(z3 < 0.8)]),ls='--',lw=2.5,color=c3)
		ax2.axvline(np.nanmedian(L4[(z4 > 0.6)&(z4 < 0.8)]),ls='--',lw=2.5,color=c4)
		ax2.axvline(np.nanmedian(L5[(z5 > 0.6)&(z5 < 0.8)]),ls='--',lw=2.5,color=c5)
		ax2.set_xlim(42,46.5)
		ax2.set_ylim(0,40)
		ax2.set_yticklabels([])
		ax2.set_xlabel(r'log L$_{\mathrm{X}}$')


		ax3 = plt.subplot(gs[2])
		ax3.set_title('0.9 < z < 1.1')
		ax3.hist(L1[(z1 > 0.9)&(z1 < 1.1)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c1)
		ax3.hist(L2[(z2 > 0.9)&(z2 < 1.1)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c2)
		ax3.hist(L3[(z3 > 0.9)&(z3 < 1.1)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c3)
		ax3.hist(L4[(z4 > 0.9)&(z4 < 1.1)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c4)
		ax3.hist(L5[(z5 > 0.9)&(z5 < 1.1)],bins=np.arange(bin1,bin2,binsize),histtype='step',lw=2.5,color=c5)
		ax3.axvline(np.nanmedian(L1[(z1 > 0.9)&(z1 < 1.1)]),ls='--',lw=2.5,color=c1)
		ax3.axvline(np.nanmedian(L2[(z2 > 0.9)&(z2 < 1.1)]),ls='--',lw=2.5,color=c2)
		ax3.axvline(np.nanmedian(L3[(z3 > 0.9)&(z3 < 1.1)]),ls='--',lw=2.5,color=c3)
		ax3.axvline(np.nanmedian(L4[(z4 > 0.9)&(z4 < 1.1)]),ls='--',lw=2.5,color=c4)
		ax3.axvline(np.nanmedian(L5[(z5 > 0.9)&(z5 < 1.1)]),ls='--',lw=2.5,color=c5)
		ax3.set_xlim(42,46.5)
		ax3.set_ylim(0,40)
		ax3.set_yticklabels([])
		ax3.set_xlabel(r'log L$_{\mathrm{X}}$')

		plt.savefig('/Users/connor_auge/Desktop/Paper/'+savestring+'.pdf')
		plt.show()

	def L_box_3zbins(self,savestring,L,F1,spec_z,uv_slope,mir_slope1,mir_slope2,goals_L=None,label=None):
		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		L = np.asarray(L)
		# L = np.log10(np.asarray(F1))


		L -= np.log10(3.8E33)
		
		uv_slope = uv_slope[L < 14]
		mir_slope1 = mir_slope1[L < 14]
		mir_slope2 = mir_slope2[L < 14]	
		L = L[L < 14]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]
		
		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3]
		L4 = L[B4]
		L5 = L[B5]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'
		c_list = [c1,c2,c3,c4,c5]
		c_list = ['r','b','g','orange','yellow']

		bin1 = 42
		bin2 = 47.5
		binsize=0.25
		yticks = [5,4,3,2,1]
		xticks = [5,4,3,2,1]

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		fig = plt.figure(figsize=(18,6))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.05,right=0.95,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		# ax1.boxplot([L1[(z1 > 0.3)&(z1 < 0.5)],L2[(z2 > 0.3)&(z2 < 0.5)],L3[(z3 > 0.3)&(z3 < 0.5)],L4[(z4 > 0.3)&(z4 < 0.5)],L5[(z5 > 0.3)&(z5 < 0.5)]],
		# patch_artist=True,boxprops=dict(facecolor='b', color='r'))
		# ax1.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 >= zlim_1)&(z1 <= zlim_2)]),np.nanmedian(L2[(z2 >= zlim_1)&(z2 <= zlim_2)]),np.nanmedian(L3[(z3 >= zlim_1)&(z3 <= zlim_2)]),np.nanmedian(L4[(z4 >= zlim_1)&(z4 <= zlim_2)]),np.nanmedian(L5[(z5 >= zlim_1)&(z5 <= zlim_2)])],color='k')
		ax1.plot([1,2,3,4,5],[np.nanmean(L1[(z1 >= zlim_1)&(z1 <= zlim_2)]),np.nanmean(L2[(z2 >= zlim_1)&(z2 <= zlim_2)]),np.nanmean(L3[(z3 >= zlim_1)&(z3 <= zlim_2)]),np.nanmean(L4[(z4 >= zlim_1)&(z4 <= zlim_2)]),np.nanmean(L5[(z5 >= zlim_1)&(z5 <= zlim_2)])],color='k')		
		ax1.boxplot(L1[(z1 >= zlim_1)&(z1 <= zlim_2)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L2[(z2 >= zlim_1)&(z2 <= zlim_2)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L3[(z3 >= zlim_1)&(z3 <= zlim_2)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L4[(z4 >= zlim_1)&(z4 <= zlim_2)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L5[(z5 >= zlim_1)&(z5 <= zlim_2)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax1.boxplot(goals_L-np.log10(3.8E33),positions=[6],patch_artist=True,boxprops=dict(facecolor='grey', color='grey'),medianprops=dict(color='k',lw=3))
		# ax1.boxplot(goals_L,positions=[6],patch_artist=True,boxprops=dict(facecolor='grey', color='grey'),medianprops=dict(color='k',lw=3))

		# ax1.set_ylim(43,47.5)
		# ax1.set_ylim(42,46.1)
		# ax1.set_ylim(10,14)
		ax1.set_ylim(8.5,12.6)
		plt.gca().invert_xaxis()		
		# ax1.set_ylim(0,40)
		ax1.set_ylabel(r'log L$_{\mathrm{bol}}$ [L$_\odot$]')
		# ax1.set_ylabel(r'log L$_{\mathrm{%s}}$ [L$_\odot$]' % label)
		# ax1.set_xticklabels(xticks)
		# ax1.set_xlabel('Panels')

		print(np.nanmean(L1[(z1 > zlim_2) & (z1 <= zlim_3)]))
		print(L1[(z1 > zlim_2) & (z1 <= zlim_3)])
		print(np.nanmean(L4[(z4 > zlim_2)&(z4 <= zlim_3)]))
		print(L4[(z4 > zlim_2)&(z4 <= zlim_3)])

		ax2 = plt.subplot(gs[1])
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		# ax2.boxplot([L1[(z1 > 0.6)&(z1 < 0.8)],L2[(z2 > 0.6)&(z2 < 0.8)],L3[(z3 > 0.6)&(z3 < 0.8)],L4[(z4 > 0.6)&(z4 < 0.8)],L5[(z5 > 0.6)&(z5 < 0.8)]])
		# ax2.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_2)&(z1 <= zlim_3)]),np.nanmedian(L2[(z2 > zlim_2)&(z2 <= zlim_3)]),np.nanmedian(L3[(z3 > zlim_2)&(z3 <= zlim_3)]),np.nanmedian(L4[(z4 > zlim_2)&(z4 <= zlim_3)]),np.nanmedian(L5[(z5 > zlim_2)&(z5 <= zlim_3)])],color='k')
		ax2.plot([1,2,3,4,5],[np.nanmean(L1[(z1 > zlim_2)&(z1 <= zlim_3)]),np.nanmean(L2[(z2 > zlim_2)&(z2 <= zlim_3)]),np.nanmean(L3[(z3 > zlim_2)&(z3 <= zlim_3)]),np.nanmean(L4[(z4 > zlim_2)&(z4 <= zlim_3)]),np.nanmean(L5[(z5 > zlim_2)&(z5 <= zlim_3)])],color='k')
		ax2.boxplot(L1[(z1 > zlim_2)&(z1 <= zlim_3)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L2[(z2 > zlim_2)&(z2 <= zlim_3)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L3[(z3 > zlim_2)&(z3 <= zlim_3)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L4[(z4 > zlim_2)&(z4 <= zlim_3)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L5[(z5 > zlim_2)&(z5 <= zlim_3)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax2.set_ylim(43,47.5)
		# ax2.set_ylim(42,46.1)
		# ax2.set_ylim(10,14)
		ax2.set_ylim(8.5,12.6)
		# ax2.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax2.set_yticklabels([])
		ax2.set_xlabel('Panel Number')


		ax3 = plt.subplot(gs[2])
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		# ax3.boxplot([L1[(z1 > 0.9)&(z1 < 1.1)],L2[(z2 > 0.9)&(z2 < 1.1)],L3[(z3 > 0.9)&(z3 < 1.1)],L4[(z4 > 0.9)&(z4 < 1.1)],L5[(z5 > 0.9)&(z5 < 1.1)]])
		# ax3.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_3)&(z1 <= zlim_4)]),np.nanmedian(L2[(z2 > zlim_3)&(z2 <= zlim_4)]),np.nanmedian(L3[(z3 > zlim_3)&(z3 <= zlim_4)]),np.nanmedian(L4[(z4 > zlim_3)&(z4 <= zlim_4)]),np.nanmedian(L5[(z5 > zlim_3)&(z5 <= zlim_4)])],color='k')
		ax3.plot([1,2,3,4,5],[np.nanmean(L1[(z1 > zlim_3)&(z1 <= zlim_4)]),np.nanmean(L2[(z2 > zlim_3)&(z2 <= zlim_4)]),np.nanmean(L3[(z3 > zlim_3)&(z3 <= zlim_4)]),np.nanmean(L4[(z4 > zlim_3)&(z4 <= zlim_4)]),np.nanmean(L5[(z5 > zlim_3)&(z5 <= zlim_4)])],color='k')
		ax3.boxplot(L1[(z1 > zlim_3)&(z1 <= zlim_4)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L2[(z2 > zlim_3)&(z2 <= zlim_4)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L3[(z3 > zlim_3)&(z3 <= zlim_4)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L4[(z4 > zlim_3)&(z4 <= zlim_4)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L5[(z5 > zlim_3)&(z5 <= zlim_4)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax3.set_ylim(43,47.5)
		# ax3.set_ylim(42,46.1)
		# ax3.set_ylim(10,14)
		ax3.set_ylim(8.5,12.6)
		# ax3.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax3.set_yticklabels([])
		secax3 = ax3.secondary_yaxis('right', functions=(ergs, solar))
		secax3.set_ylabel(r'log L$_\mathrm{bol}$ [erg/s]')
		# secax3.set_ylabel(r'log L$_\mathrm{%s}$ [erg/s]' % label)

		# ax3.set_xlabel('Panels')

		ax1.grid()
		ax2.grid()
		ax3.grid()

		plt.savefig('/Users/connor_auge/Desktop/New_plots3/'+savestring+'.pdf')
		plt.show()

	def Nh_box_3zbins(self,savestring,L,F1,spec_z,uv_slope,mir_slope1,mir_slope2):
		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		L = np.asarray(L)
		# L[L < 20] = np.nan
		# print(L)

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3]
		L4 = L[B4]
		L5 = L[B5]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'


		fig = plt.figure(figsize=(18,6))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.05,right=0.97,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		# ax1.boxplot([L1[(z1 > 0.3)&(z1 < 0.5)],L2[(z2 > 0.3)&(z2 < 0.5)],L3[(z3 > 0.3)&(z3 < 0.5)],L4[(z4 > 0.3)&(z4 < 0.5)],L5[(z5 > 0.3)&(z5 < 0.5)]],
		# patch_artist=True,boxprops=dict(facecolor='b', color='r'))
		# ax1.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 >= zlim_1)&(z1 <= zlim_2)]),np.nanmedian(L2[(z2 >= zlim_1)&(z2 <= zlim_2)]),np.nanmedian(L3[(z3 >= zlim_1)&(z3 <= zlim_2)]),np.nanmedian(L4[(z4 >= zlim_1)&(z4 <= zlim_2)]),np.nanmedian(L5[(z5 >= zlim_1)&(z5 <= zlim_2)])],color='k')
		ax1.plot([1,2,3,4,5],[np.nanmean(L1[(z1 >= zlim_1)&(z1 <= zlim_2)]),np.nanmean(L2[(z2 >= zlim_1)&(z2 <= zlim_2)]),np.nanmean(L3[(z3 >= zlim_1)&(z3 <= zlim_2)]),np.nanmean(L4[(z4 >= zlim_1)&(z4 <= zlim_2)]),np.nanmean(L5[(z5 >= zlim_1)&(z5 <= zlim_2)])],color='k')
		ax1.boxplot(L1[(z1 >= zlim_1)&(z1 <= zlim_2)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L2[(z2 >= zlim_1)&(z2 <= zlim_2)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L3[(z3 >= zlim_1)&(z3 <= zlim_2)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L4[(z4 >= zlim_1)&(z4 <= zlim_2)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L5[(z5 >= zlim_1)&(z5 <= zlim_2)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.set_ylim(19.5,24.5)
		# ax1.set_ylim(42,46.5)
		plt.gca().invert_xaxis()		
		# ax1.set_ylim(0,40)
		ax1.set_ylabel(r'log N$_{\mathrm{H}}$ [cm$^{-2}$]')
		# ax1.set_xlabel('Panels')

	


		ax2 = plt.subplot(gs[1])
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		# ax2.boxplot([L1[(z1 > 0.6)&(z1 < 0.8)],L2[(z2 > 0.6)&(z2 < 0.8)],L3[(z3 > 0.6)&(z3 < 0.8)],L4[(z4 > 0.6)&(z4 < 0.8)],L5[(z5 > 0.6)&(z5 < 0.8)]])
		# ax2.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_2)&(z1 <= zlim_3)]),np.nanmedian(L2[(z2 > zlim_2)&(z2 <= zlim_3)]),np.nanmedian(L3[(z3 > zlim_2)&(z3 <= zlim_3)]),np.nanmedian(L4[(z4 > zlim_2)&(z4 <= zlim_3)]),np.nanmedian(L5[(z5 > zlim_2)&(z5 <= zlim_3)])],color='k')
		ax2.plot([1,2,3,4,5],[np.nanmean(L1[(z1 > zlim_2)&(z1 <= zlim_3)]),np.nanmean(L2[(z2 > zlim_2)&(z2 <= zlim_3)]),np.nanmean(L3[(z3 > zlim_2)&(z3 <= zlim_3)]),np.nanmean(L4[(z4 > zlim_2)&(z4 <= zlim_3)]),np.nanmean(L5[(z5 > zlim_2)&(z5 <= zlim_3)])],color='k')
		ax2.boxplot(L1[(z1 > zlim_2)&(z1 <= zlim_3)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L2[(z2 > zlim_2)&(z2 <= zlim_3)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L3[(z3 > zlim_2)&(z3 <= zlim_3)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L4[(z4 > zlim_2)&(z4 <= zlim_3)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L5[(z5 > zlim_2)&(z5 <= zlim_3)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.set_ylim(19.5,24.5)
		# ax2.set_ylim(42,46.5)
		# ax2.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax2.set_yticklabels([])
		ax2.set_xlabel('Panel Number')


		ax3 = plt.subplot(gs[2])
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		# ax3.boxplot([L1[(z1 > 0.9)&(z1 < 1.1)],L2[(z2 > 0.9)&(z2 < 1.1)],L3[(z3 > 0.9)&(z3 < 1.1)],L4[(z4 > 0.9)&(z4 < 1.1)],L5[(z5 > 0.9)&(z5 < 1.1)]])
		# ax3.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_3)&(z1 <= zlim_4)]),np.nanmedian(L2[(z2 > zlim_3)&(z2 <= zlim_4)]),np.nanmedian(L3[(z3 > zlim_3)&(z3 <= zlim_4)]),np.nanmedian(L4[(z4 > zlim_3)&(z4 <= zlim_4)]),np.nanmedian(L5[(z5 > zlim_3)&(z5 <= zlim_4)])],color='k')
		ax3.plot([1,2,3,4,5],[np.nanmean(L1[(z1 > zlim_3)&(z1 <= zlim_4)]),np.nanmean(L2[(z2 > zlim_3)&(z2 <= zlim_4)]),np.nanmean(L3[(z3 > zlim_3)&(z3 <= zlim_4)]),np.nanmean(L4[(z4 > zlim_3)&(z4 <= zlim_4)]),np.nanmean(L5[(z5 > zlim_3)&(z5 <= zlim_4)])],color='k')
		ax3.boxplot(L1[(z1 > zlim_3)&(z1 <= zlim_4)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L2[(z2 > zlim_3)&(z2 <= zlim_4)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L3[(z3 > zlim_3)&(z3 <= zlim_4)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L4[(z4 > zlim_3)&(z4 <= zlim_4)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L5[(z5 > zlim_3)&(z5 <= zlim_4)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.set_ylim(19.5,24.5)
		# ax3.set_ylim(42,46.5)
		# ax3.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax3.set_yticklabels([])
		# ax3.set_xlabel('Panels')

		ax1.grid()
		ax2.grid()
		ax3.grid()

		plt.savefig('/Users/connor_auge/Desktop/New_plots3/'+savestring+'.pdf')
		plt.show()

	def L_box_fields(self,savestring,L,F1,spec_z,uv_slope,mir_slope1,mir_slope2,field,goals_L=None,label=None):
		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		L = np.asarray(L)
		# L = np.log10(np.asarray(F1))


		L -= np.log10(3.8E33)
		
		uv_slope = uv_slope[L < 14]
		mir_slope1 = mir_slope1[L < 14]
		mir_slope2 = mir_slope2[L < 14]	
		L = L[L < 14]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]
		
		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3]
		L4 = L[B4]
		L5 = L[B5]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		field1 = field[B1]
		field2 = field[B2]
		field3 = field[B3]
		field4 = field[B4]
		field5 = field[B5]

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'
		c_list = [c1,c2,c3,c4,c5]
		c_list = ['r','b','g','orange','yellow']

		bin1 = 42
		bin2 = 47.5
		binsize=0.25
		yticks = [5,4,3,2,1]
		xticks = [5,4,3,2,1]

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		fig = plt.figure(figsize=(18,6))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.05,right=0.95,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		# ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))
		ax1.set_title('GOODS-N/S')
		# ax1.boxplot([L1[(z1 > 0.3)&(z1 < 0.5)],L2[(z2 > 0.3)&(z2 < 0.5)],L3[(z3 > 0.3)&(z3 < 0.5)],L4[(z4 > 0.3)&(z4 < 0.5)],L5[(z5 > 0.3)&(z5 < 0.5)]],
		# patch_artist=True,boxprops=dict(facecolor='b', color='r'))
		# ax1.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 >= zlim_1)&(z1 <= zlim_2)]),np.nanmedian(L2[(z2 >= zlim_1)&(z2 <= zlim_2)]),np.nanmedian(L3[(z3 >= zlim_1)&(z3 <= zlim_2)]),np.nanmedian(L4[(z4 >= zlim_1)&(z4 <= zlim_2)]),np.nanmedian(L5[(z5 >= zlim_1)&(z5 <= zlim_2)])],color='k')
		ax1.plot([1,2,3,4,5],[np.nanmean(L1[[np.logical_or(field1==2,field1==3)]]),np.nanmean(L2[[np.logical_or(field2==2,field2==3)]]),np.nanmean(L3[[np.logical_or(field3==2,field3==3)]]),np.nanmean(L4[[np.logical_or(field4==2,field4==3)]]),np.nanmean(L5[np.logical_or(field5==2,field5==3)])],color='k')	
		ax1.boxplot(L1[np.logical_or(field1==2,field1==3)],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L2[np.logical_or(field2==2,field2==3)],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L3[np.logical_or(field3==2,field3==3)],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L4[np.logical_or(field4==2,field4==3)],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax1.boxplot(L5[np.logical_or(field5==2,field5==3)],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax1.boxplot(goals_L-np.log10(3.8E33),positions=[6],patch_artist=True,boxprops=dict(facecolor='grey', color='grey'),medianprops=dict(color='k',lw=3))
		# ax1.boxplot(goals_L,positions=[6],patch_artist=True,boxprops=dict(facecolor='grey', color='grey'),medianprops=dict(color='k',lw=3))

		# ax1.set_ylim(43,47.5)
		# ax1.set_ylim(42,46.1)
		# ax1.set_ylim(10,14)
		ax1.set_ylim(8.5,12.6)
		plt.gca().invert_xaxis()		
		# ax1.set_ylim(0,40)
		# ax1.set_ylabel(r'log L$_{\mathrm{X}}$ [erg/s]')
		ax1.set_ylabel(r'log L$_{\mathrm{%s}}$ [L$_\odot$]' % label)
		# ax1.set_xticklabels(xticks)
		# ax1.set_xlabel('Panels')

		print(np.nanmean(L1[(z1 > zlim_2) & (z1 <= zlim_3)]))
		print(L1[(z1 > zlim_2) & (z1 <= zlim_3)])
		print(np.nanmean(L4[(z4 > zlim_2)&(z4 <= zlim_3)]))
		print(L4[(z4 > zlim_2)&(z4 <= zlim_3)])

		ax2 = plt.subplot(gs[1])
		# ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))
		ax2.set_title('COSMOS')
		# ax2.boxplot([L1[(z1 > 0.6)&(z1 < 0.8)],L2[(z2 > 0.6)&(z2 < 0.8)],L3[(z3 > 0.6)&(z3 < 0.8)],L4[(z4 > 0.6)&(z4 < 0.8)],L5[(z5 > 0.6)&(z5 < 0.8)]])
		# ax2.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_2)&(z1 <= zlim_3)]),np.nanmedian(L2[(z2 > zlim_2)&(z2 <= zlim_3)]),np.nanmedian(L3[(z3 > zlim_2)&(z3 <= zlim_3)]),np.nanmedian(L4[(z4 > zlim_2)&(z4 <= zlim_3)]),np.nanmedian(L5[(z5 > zlim_2)&(z5 <= zlim_3)])],color='k')
		ax2.plot([1,2,3,4,5],[np.nanmean(L1[field1==0]),np.nanmean(L2[field2==0]),np.nanmean(L3[field3==0]),np.nanmean(L4[field4==0]),np.nanmean(L5[field5==0])],color='k')
		ax2.boxplot(L1[field1==0],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L2[field2==0],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L3[field3==0],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L4[field4==0],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax2.boxplot(L5[field5==0],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax2.set_ylim(43,47.5)
		# ax2.set_ylim(42,46.1)
		# ax2.set_ylim(10,14)
		ax2.set_ylim(8.5,12.6)
		# ax2.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax2.set_yticklabels([])
		ax2.set_xlabel('Panel Number')


		ax3 = plt.subplot(gs[2])
		# ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))
		ax3.set_title('Stripe82X')
		# ax3.boxplot([L1[(z1 > 0.9)&(z1 < 1.1)],L2[(z2 > 0.9)&(z2 < 1.1)],L3[(z3 > 0.9)&(z3 < 1.1)],L4[(z4 > 0.9)&(z4 < 1.1)],L5[(z5 > 0.9)&(z5 < 1.1)]])
		# ax3.plot([1,2,3,4,5],[np.nanmedian(L1[(z1 > zlim_3)&(z1 <= zlim_4)]),np.nanmedian(L2[(z2 > zlim_3)&(z2 <= zlim_4)]),np.nanmedian(L3[(z3 > zlim_3)&(z3 <= zlim_4)]),np.nanmedian(L4[(z4 > zlim_3)&(z4 <= zlim_4)]),np.nanmedian(L5[(z5 > zlim_3)&(z5 <= zlim_4)])],color='k')
		ax3.plot([1,2,3,4,5],[np.nanmean(L1[field1==1]),np.nanmean(L2[field2==1]),np.nanmean(L3[field3==1]),np.nanmean(L4[field4==1]),np.nanmean(L5[field5==1])],color='k')
		ax3.boxplot(L1[field1==1],positions=[1],patch_artist=True,boxprops=dict(facecolor=c1,color=c1),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L2[field2==1],positions=[2],patch_artist=True,boxprops=dict(facecolor=c2,color=c2),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L3[field3==1],positions=[3],patch_artist=True,boxprops=dict(facecolor=c3,color=c3),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L4[field4==1],positions=[4],patch_artist=True,boxprops=dict(facecolor=c4,color=c4),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		ax3.boxplot(L5[field5==1],positions=[5],patch_artist=True,boxprops=dict(facecolor=c5,color=c5),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))
		# ax3.set_ylim(43,47.5)
		# ax3.set_ylim(42,46.1)
		# ax3.set_ylim(10,14)
		ax3.set_ylim(8.5,12.6)
		# ax3.set_ylim(0,40)
		plt.gca().invert_xaxis()
		ax3.set_yticklabels([])
		secax3 = ax3.secondary_yaxis('right', functions=(ergs, solar))
		# secax3.set_ylabel(r'log L$_\mathrm{X}$ [L$_{\odot}$]')
		secax3.set_ylabel(r'log L$_\mathrm{%s}$ [erg/s]' % label)

		# ax3.set_xlabel('Panels')

		ax1.grid()
		ax2.grid()
		ax3.grid()

		plt.savefig('/Users/connor_auge/Desktop/New_runSED/'+savestring+'.pdf')
		plt.show()



	def L_box(self,savestring,L,F1,spec_z,uv_slope,mir_slope1,mir_slope2):

		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		L = np.asarray(L)

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3]
		L4 = L[B4]
		L5 = L[B5]

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		fig = plt.figure(figsize=(10,8))
		gs = fig.add_gridspec(nrows=1, ncols=1)
		# gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		# gs.update(left=0.05,right=0.97,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		ax1.plot([1,2,3,4,5],[np.nanmedian(L1),np.nanmedian(L2),np.nanmedian(L3),np.nanmedian(L4),np.nanmedian(L5)],color='k')
		ax1.boxplot(L1,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color=c1),medianprops=dict(color='k',lw=3))
		ax1.boxplot(L2,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color=c2),medianprops=dict(color='k',lw=3))
		ax1.boxplot(L3,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color=c3),medianprops=dict(color='k',lw=3))
		ax1.boxplot(L4,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color=c4),medianprops=dict(color='k',lw=3))
		ax1.boxplot(L5,positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color=c5),medianprops=dict(color='k',lw=3))
		# ax1.set_ylim(43,47.5)
		ax1.set_ylim(42,46.5)
		# ax1.set_ylim(19.5,25)
		plt.gca().invert_xaxis()		
		# ax1.set_ylim(0,40)
		ax1.set_ylabel(r'log L$_{\mathrm{X}}$ [erg/s]')
		# ax1.set_ylabel(r'log N$_{\mathrm{H}}$ [cm$^{-2}$]')
		ax1.set_xlabel('Panel Numbers')

		ax1.grid()

		plt.savefig('/Users/connor_auge/Desktop/Paper/Lx_425/final/'+savestring+'.pdf')
		plt.show()



	def L_Lx_scatter_zbins(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis5=None,emis6=None,spec_z=None,emis7=None,emis8=None,emis9=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None,fir_frac=None):

		plt.rcParams['font.size']=16
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3


		l = np.asarray([10**i for i in L])
		# l -= (10**44.66)

		# print(L - np.log10(l))
		L = np.log10(l)
		L -= np.log10(3.8E33)

		emis5=[]
		emis6=[]
		emis9=[]
		F13=[]

		e5 = np.asarray([10**i for i in emis5])
		# e5 -= (10**44.66)
		# emis5 = np.log10(e5)
		emis5 = e5
		emis5 /= 3.8E33

		# print(emis6,emis5)
		# print(emis7,emis8)

		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		
		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		up_check1,up_check2,up_check3,up_check4,up_check5 = up_check[B1],up_check[B2],up_check[B3],up_check[B4],up_check[B5]


		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one
		scale_GOALS = np.asarray(F13)/temp_L_one

		temp_L_bol = 1E10

		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5
		temp_L_bol_GOALS = temp_L_bol*scale_GOALS

		f_3 = np.asarray([10**i for i in f3])
		f_3 *= F1
		f3 = np.log10(f_3)

		f_4 = np.asarray([10**i for i in f4])
		f_4 *= F1
		f4 = np.log10(f_4)

		f_1 = np.asarray([10**i for i in f1])
		f_1 *= F1
		f1 = np.log10(f_1)


		# e9 = np.asarray([10**i for i in emis9])
		# emis9 *= F13
		emis9 = np.log10(emis9)

		F1 = np.log10(F1)


		# l -= (tem_L_bol*)

		temp_L_bol3 = 0
		temp_L_bol4 = 0
		temp_L_bol5 = 0
		temp_L_bol_GOALS = 0


		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)
		emis6 -= np.log10(3.8E33)
		# B1 = np.where(spec_z < 0.5)[0]
		# B2 = np.where(np.logical_and(spec_z > 0.5, spec_z < 1.0))[0]
		# B3 = np.where(np.logical_and(spec_z > 1.0, spec_z < 1.5))[0]
		# B4 = np.where(np.logical_and(spec_z > 1.5, spec_z < 2.0))[0]
		# B5 = np.where(np.logical_and(spec_z > 2.0, spec_z < 2.5))[0]
		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		# fir_frac1 = fir_frac[B1]
		# fir_frac2 = fir_frac[B2]
		# fir_frac3 = fir_frac[B3]
		# fir_frac4 = fir_frac[B4]
		# fir_frac5 = fir_frac[B5]

		L1 = L[B1]
		L2 = L[B2]
		L3 = L[B3] - temp_L_bol3
		L4 = L[B4] - temp_L_bol4
		L5 = L[B5] - temp_L_bol5
		L_sub = np.append(L1,L2)
		L_sub = np.append(L_sub,L3)
		L_sub = np.append(L_sub,L4)
		L_sub = np.append(L_sub,L5)
		# L_sub = np.append(L_sub,emis5 - temp_L_bol_GOALS)

		lx = np.asarray([10**i for i in Lx])
		Lx = lx
		e6 = np.asarray([10**i for i in emis6])
		emis6 = e6
		
		Lx1 = Lx[B1]
		Lx2 = Lx[B2]
		Lx3 = Lx[B3]
		Lx4 = Lx[B4]
		Lx5 = Lx[B5]
		Lx_sub = np.append(Lx1,Lx2)
		Lx_sub = np.append(Lx_sub,Lx3)
		Lx_sub = np.append(Lx_sub,Lx4)
		Lx_sub = np.append(Lx_sub,Lx5)
		# Lx_sub = np.append(Lx_sub,emis6)

		fc1 = F1[B1]
		fc2 = F1[B2]
		fc3 = F1[B3]
		fc4 = F1[B4]
		fc5 = F1[B5]
		fc_sub = np.append(fc1,fc2)
		fc_sub = np.append(fc_sub,fc3)
		fc_sub = np.append(fc_sub,fc4)
		fc_sub = np.append(fc_sub,fc5)

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]
		z_sub = np.append(z1,z2)
		z_sub = np.append(z_sub,z3)
		z_sub = np.append(z_sub,z4)
		z_sub = np.append(z_sub,z5)


		sort = np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]).argsort()

		z = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],1)
		# z = np.polyfit(Lx_sub[np.isfinite(np.log10(L_sub))],L_sub[np.isfinite(np.log10(L_sub))],1)
		# z = np.polyfit(Lx_sub[np.isfinite(L_sub)],L_sub[np.isfinite(L_sub)],1)		
		p = np.poly1d(z)


		z2 = np.polyfit(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],np.log10(L_sub[np.isfinite(np.log10(L_sub))])[sort],2)
		# z2 = np.polyfit(Lx_sub[np.isfinite(np.log10(L_sub))],L_sub[np.isfinite(np.log10(L_sub))],2)
		p2 = np.poly1d(z2)

		


		x = np.linspace(10**8,10**15,20)
		# x = np.arange(8,14)
		y = p(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort])
		# y2 = p2(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort])
		# print(y)
		# print(z,np.log10(z))

		# for i in range(len(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))][sort]))):
			# print(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))][sort][i]),np.log10(L_sub[np.isfinite(np.log10(L_sub))][sort][i]))

		U = np.zeros(np.shape(Lx1[up_check1==1]))
		V = np.ones(np.shape(Lx1[up_check1==1]))*-1


		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		# print([(spec_z[B1] > 0.3)&(spec_z[B1] < 0.5)])
		# print(Lx1)
		# print(np.log10(Lx1[(spec_z[B1] > 0.3)&(spec_z[B1] < 0.5)]))
		# print(np.log10(L1[(spec_z[B1] > 0.3)&(spec_z[B1] < 0.5)]))
		fig = plt.figure(figsize=(8,8))
		ax = plt.subplot(111)
		# plt.scatter(np.log10(Lx1[(spec_z[B1] > 0.3)&(spec_z[B1] < 0.5)]),np.log10(L1[(spec_z[B1] > 0.3)&(spec_z[B1] < 0.5)]),marker='s',color='b',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx2[(spec_z[B2] > 0.3)&(spec_z[B2] < 0.5)]),np.log10(L2[(spec_z[B2] > 0.3)&(spec_z[B2] < 0.5)]),marker='s',color='purple',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx3[(spec_z[B3] > 0.3)&(spec_z[B3] < 0.5)]),np.log10(L3[(spec_z[B3] > 0.3)&(spec_z[B3] < 0.5)]),marker='s',color='green',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx4[(spec_z[B4] > 0.3)&(spec_z[B4] < 0.5)]),np.log10(L4[(spec_z[B4] > 0.3)&(spec_z[B4] < 0.5)]),marker='s',color='orange',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx5[(spec_z[B5] > 0.3)&(spec_z[B5] < 0.5)]),np.log10(L5[(spec_z[B5] > 0.3)&(spec_z[B5] < 0.5)]),marker='s',color='red',s=80,alpha=0.9,rasterized=True)

		# plt.scatter(np.log10(Lx1[(spec_z[B1] > 0.6)&(spec_z[B1] < 0.8)]),np.log10(L1[(spec_z[B1] > 0.6)&(spec_z[B1] < 0.8)]),marker='*',color='b',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx2[(spec_z[B2] > 0.6)&(spec_z[B2] < 0.8)]),np.log10(L2[(spec_z[B2] > 0.6)&(spec_z[B2] < 0.8)]),marker='*',color='purple',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx3[(spec_z[B3] > 0.6)&(spec_z[B3] < 0.8)]),np.log10(L3[(spec_z[B3] > 0.6)&(spec_z[B3] < 0.8)]),marker='*',color='green',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx4[(spec_z[B4] > 0.6)&(spec_z[B4] < 0.8)]),np.log10(L4[(spec_z[B4] > 0.6)&(spec_z[B4] < 0.8)]),marker='*',color='orange',s=80,alpha=0.9,rasterized=True)
		# plt.scatter(np.log10(Lx5[(spec_z[B5] > 0.6)&(spec_z[B5] < 0.8)]),np.log10(L5[(spec_z[B5] > 0.6)&(spec_z[B5] < 0.8)]),marker='*',color='red',s=80,alpha=0.9,rasterized=True)

		# plt.scatter(np.log10(Lx1[(spec_z[B1] > 0.9)&(spec_z[B1] < 1.1)]),np.log10(L1[(spec_z[B1] > 0.9)&(spec_z[B1] < 1.1)]),marker='o',color='b',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		# plt.scatter(np.log10(Lx2[(spec_z[B2] > 0.9)&(spec_z[B2] < 1.1)]),np.log10(L2[(spec_z[B2] > 0.9)&(spec_z[B2] < 1.1)]),marker='o',color='purple',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		# plt.scatter(np.log10(Lx3[(spec_z[B3] > 0.9)&(spec_z[B3] < 1.1)]),np.log10(L3[(spec_z[B3] > 0.9)&(spec_z[B3] < 1.1)]),marker='o',color='green',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		# plt.scatter(np.log10(Lx4[(spec_z[B4] > 0.9)&(spec_z[B4] < 1.1)]),np.log10(L4[(spec_z[B4] > 0.9)&(spec_z[B4] < 1.1)]),marker='o',color='orange',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		# plt.scatter(np.log10(Lx5[(spec_z[B5] > 0.9)&(spec_z[B5] < 1.1)]),np.log10(L5[(spec_z[B5] > 0.9)&(spec_z[B5] < 1.1)]),marker='o',color='red',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		# plt.scatter(-100,-100,marker='s',color='k',label='0.3 < z < 0.5')
		# plt.scatter(-100,-100,marker='*',color='k',label='0.6 < z < 0.8')
		# plt.scatter(-100,-100,marker='o',color='k',label='0.9 < z < 1.1')

		print(np.shape(Lx_sub),np.shape(z_sub))
		plt.scatter(np.log10(Lx_sub[(z_sub > 0.3)&(z_sub < 0.5)]),np.log10(L_sub[(z_sub > 0.3)&(z_sub < 0.5)]),color='orange',s=100,label='0.3 < z < 0.5',rasterized=True)
		plt.scatter(np.log10(Lx_sub[(z_sub > 0.6)&(z_sub < 0.8)]),np.log10(L_sub[(z_sub > 0.6)&(z_sub < 0.8)]),color='purple',s=100,label='0.6 < z < 0.8',rasterized=True)
		plt.scatter(np.log10(Lx_sub[(z_sub > 0.9)&(z_sub < 1.1)]),np.log10(L_sub[(z_sub > 0.9)&(z_sub < 1.1)]),color='cyan',s=100,label='0.9 < z < 1.1',rasterized=True)

		# plt.quiver(np.log10(Lx1[up_check1==1]),np.log10(L1[up_check1==1]),U,V,color='b',scale=np.log10(fir_frac1[up_check1==1]*L1[up_check1==1]))
		# plt.plot([np.log10(Lx1[up_check1==1]),np.log10(Lx1[up_check1==1])],[np.log10(L1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1])],color='k',lw=3)		
		# plt.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]<0.15]),color='blue',s=60,alpha=0.9)
		# plt.scatter(np.log10(Lx1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),np.log10(L1[up_check1==1][fir_frac1[up_check1==1]/L1[up_check1==1]>0.15]),marker='v',color='blue',s=60,alpha=0.9)

		# plt.scatter(np.log10(Lx1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx2[up_check2==1]),np.log10(Lx2[up_check2==1])],[np.log10(L2[up_check2==1]),np.log10(L2[up_check2==1]-fir_frac2[up_check2==1])],color='k',lw=3)		
		# plt.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]<0.15]),color='purple',s=60,alpha=0.9)
		# plt.scatter(np.log10(Lx2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),np.log10(L2[up_check2==1][fir_frac2[up_check2==1]/L2[up_check2==1]>0.15]),marker='v',color='purple',s=60,alpha=0.9)		
		# plt.scatter(np.log10(Lx2[up_check2==1]),np.log10(L2[up_check2==1]-fir_frac2[up_check2==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx3[up_check3==1]),np.log10(Lx3[up_check3==1])],[np.log10(L3[up_check3==1]),np.log10(L3[up_check3==1]-fir_frac3[up_check3==1])],color='k',lw=3)		
		# plt.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]<0.15]),color='green',s=60,alpha=0.9)		
		# plt.scatter(np.log10(Lx3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),np.log10(L3[up_check3==1][fir_frac3[up_check3==1]/L3[up_check3==1]>0.15]),marker='v',color='green',s=60,alpha=0.9)		

		# plt.scatter(np.log10(Lx3[up_check3==1]),np.log10(L3[up_check3==1]-fir_frac3[up_check3==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx4[up_check4==1]),np.log10(Lx4[up_check4==1])],[np.log10(L4[up_check4==1]),np.log10(L4[up_check4==1]-fir_frac4[up_check4==1])],color='k',lw=3)		
		# plt.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]<0.15]),color='orange',s=60,alpha=0.9)		
		# plt.scatter(np.log10(Lx4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),np.log10(L4[up_check4==1][fir_frac4[up_check4==1]/L4[up_check4==1]>0.15]),marker='v',color='orange',s=60,alpha=0.9)		# plt.scatter(np.log10(Lx4[up_check4==1]),np.log10(L4[up_check4==1]-fir_frac4[up_check4==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.plot([np.log10(Lx5[up_check5==1]),np.log10(Lx5[up_check5==1])],[np.log10(L5[up_check5==1]),np.log10(L5[up_check5==1]-fir_frac5[up_check5==1])],color='k',lw=3)		
		# plt.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]<0.15]),color='red',s=60,alpha=0.9)		
		# plt.scatter(np.log10(Lx5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),np.log10(L5[up_check5==1][fir_frac5[up_check5==1]/L5[up_check5==1]>0.15]),marker='v',color='red',s=60,alpha=0.9)
		# plt.scatter(np.log10(Lx5[up_check5==1]),np.log10(L5[up_check5==1]-fir_frac5[up_check5==1]),marker='v',color='k',s=60,alpha=0.9)

		# plt.scatter(np.log10(emis6),np.log10(emis5 - temp_L_bol_GOALS),color='gray',edgecolors='k',s=80,alpha=0.9,label='ULIRGs')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))]),np.log10(L_sub[np.isfinite(np.log10(L_sub))]),'.',color='k')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],y,color='k')
		# plt.plot(np.log10(Lx_sub[np.isfinite(np.log10(L_sub))])[sort],y2,color='k',ls='--')
		# print([np.log10(Lx1[up_check1==1]),np.log10(Lx1[up_check1==1])],[np.log10(L1[up_check1==1]),np.log10(L1[up_check1==1]-fir_frac1[up_check1==1])])
		# print(L1[up_check1==1])
		# print('FIR_L:',fir_frac1[up_check1==1])
		# print('FIR_frac1:',fir_frac1[up_check1==1]/L1[up_check1==1])
		# print('FIR_frac2:',fir_frac2[up_check2==1]/L2[up_check2==1])
		# print('FIR_frac3:',fir_frac3[up_check3==1]/L3[up_check3==1])
		# print('FIR_frac4:',fir_frac4[up_check4==1]/L4[up_check4==1])
		# print('FIR_frac5:',fir_frac5[up_check5==1]/L5[up_check5==1])
		# print('FIR_frac1:',max(fir_frac1[up_check1==1]/L1[up_check1==1]))
		# print('FIR_frac2:',max(fir_frac2[up_check2==1]/L2[up_check2==1]))
		# print('FIR_frac3:',max(fir_frac3[up_check3==1]/L3[up_check3==1]))
		# print('FIR_frac4:',max(fir_frac4[up_check4==1]/L4[up_check4==1]))
		# print('FIR_frac5:',max(fir_frac5[up_check5==1]/L5[up_check5==1]))
		# print(L1[up_check1==1]-fir_frac1[up_check1==1])
		# plt.scatter(Lx[B1],L[B1],color='b',s=80,alpha=0.9,label='Panel 1')
		# plt.scatter(Lx[B2],L[B2],color='purple',s=80,alpha=0.9,label='Panel 2')
		# plt.scatter(Lx[B3],L[B3] - temp_L_bol3,color='green',s=80,alpha=0.9,label='Panel 3')
		# plt.scatter(Lx[B4],L[B4] - temp_L_bol4,color='orange',s=80,alpha=0.9,label='Panel 4')
		# plt.scatter(Lx[B5],L[B5] - temp_L_bol5,color='red',s=80,alpha=0.9,label='Panel 5')
		# plt.scatter(emis6,emis5 - temp_L_bol_GOALS,color='gray',edgecolors='k',s=80,alpha=0.9,label='ULIRGs')
		# plt.plot(10**x,y,color='k')


		# plt.scatter(Lx[B1],np.log10(L[B1]),c=f3[B1],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='s',label='Panel 1')
		# plt.scatter(Lx[B2],np.log10(L[B2]),c=f3[B2],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='o',label='Panel 2')
		# plt.scatter(Lx[B3],np.log10(L[B3] - temp_L_bol3),c=f3[B3],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='P',label='Panel 3')
		# plt.scatter(Lx[B4],np.log10(L[B4] - temp_L_bol4),c=f3[B4],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='X',label='Panel 4')
		# plt.scatter(Lx[B5],np.log10(L[B5] - temp_L_bol5),c=f3[B5],cmap='viridis',edgecolors='k',s=80,alpha=1,marker='D',label='Panel 5')
		# plt.scatter(emis6,np.log10(emis5 - temp_L_bol_GOALS),c=emis9,cmap='viridis',edgecolors='k',s=100,alpha=1,marker='*',label='ULIRGs')
		
		# plt.scatter(Lx,np.log10(L),c=f3,cmap='viridis')
		# plt.plot(Lx,np.log10(L),'.',color='k')
		# print(F1)
		# print(f3)

		# plt.scatter(Lx_sub,np.log10(L_sub),c=fc_sub,cmap='viridis',edgecolors='k',s=80)




		# plt.scatter(np.nanmean(Lx[B1]),np.nanmean(np.log10(L[B1])),color='b',marker='s',s=120,label='Panel 1')
		# plt.errorbar(np.nanmean(Lx[B1]),np.nanmean(np.log10(L[B1])),np.nanstd(Lx[B1]),np.nanstd(np.log10(L[B1])),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B2]),np.nanmean(np.log10(L[B2])),color='purple',marker='s',s=120,label='Panel 2')
		# plt.errorbar(np.nanmean(Lx[B2]),np.nanmean(np.log10(L[B2])),np.nanstd(Lx[B2]),np.nanstd(np.log10(L[B2])),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B3]),np.nanmean(np.log10(L[B3] - temp_L_bol3)),color='green',marker='s',s=120,label='Panel 3')
		# plt.errorbar(np.nanmean(Lx[B3]),np.nanmean(np.log10(L[B3] - temp_L_bol3)),np.nanstd(Lx[B3]),np.nanstd(np.log10(L[B3] - temp_L_bol3)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B4]),np.nanmean(np.log10(L[B4] - temp_L_bol4)),color='orange',marker='s',s=120,label='Panel 4')
		# plt.errorbar(np.nanmean(Lx[B4]),np.nanmean(np.log10(L[B4] - temp_L_bol4)),np.nanstd(Lx[B4]),np.nanstd(np.log10(L[B4] - temp_L_bol4)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(Lx[B5]),np.nanmean(np.log10(L[B5] - temp_L_bol5)),color='red',marker='s',s=120,label='Panel 5')
		# plt.errorbar(np.nanmean(Lx[B5]),np.nanmean(np.log10(L[B5] - temp_L_bol5)),np.nanstd(Lx[B5]),np.nanstd(np.log10(L[B5] - temp_L_bol5)),fmt='none',ecolor='gray',zorder=0.)
		# plt.scatter(np.nanmean(emis6),np.nanmean(np.log10(emis5 - temp_L_bol_GOALS)),color='gray',marker='s',edgecolors='k',s=120,label='ULIRGs')
		# plt.errorbar(np.nanmean(emis6),np.nanmean(np.log10(emis5 - temp_L_bol_GOALS)),np.nanstd(emis6),np.nanstd(np.log10(emis5 - temp_L_bol_GOALS)),fmt='none',ecolor='gray',zorder=0.)

		# plt.scatter(emis7,np.log10(emis8)-np.log10(3.8E33),color='k',edgecolors='k',s=80,label='Swift/BAT')

		# plt.axvline(42.5,color='k',lw=3)
		# plt.colorbar(label=r'log L$_{1\mu\mathrm{m}}$ erg/s')
		secax = ax.secondary_xaxis('top',functions=(ergs, solar))
		secax = ax.secondary_yaxis('right',functions=(ergs, solar))
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [erg/s]')
		secax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}$ [L$_{\odot}$]')
		ax.set_xlabel(r'log $\mathrm{L}_{\mathrm{X}}({0.5-10\mathrm{kev}})$ [erg/s]')
		ax.set_ylabel(r'log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# ax.set_xlim(42,47)
		ax.set_xlim(8,13)
		ax.set_ylim(10,15)
		# plt.ylim(43,48)
		plt.legend(fontsize=12)
		# plt.xscale('log')
		# plt.yscale('log')
		plt.grid()
		plt.savefig('/Users/connor_auge/Desktop/final_paper_43/Lbol_Lx_zbins'+savestring+'.pdf')
		plt.show()

		# plt.hist(L_sub[L_sub > 0]/Lx_sub[L_sub > 0],bins=np.arange(0,200,5))
		# plt.axvline(np.nanmedian(L_sub[L_sub > 0]/Lx_sub[L_sub > 0]),color='k',ls='--')
		# plt.xlabel('Lbol/Lx')
		# plt.show()


		z3 = np.polyfit(np.log10(L_sub[L_sub > 0]),np.log10(L_sub[L_sub > 0]/Lx_sub[L_sub > 0]),1)
		p3 = np.poly1d(z3)

		# x3 = np.linspace(10**8,10**20,100)
		x3 = np.arange(5,20)
		y3 = 10**p3(x3)

		

	def Lbol_Lbol(self,savestring,Lx,L,F1,uv_slope,mir_slope1,mir_slope2,spec_z,up_check=None,fir_frac=None):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3


		L[L > 100] = np.nan
		# Lx[Lx > 45.5] = np.nan


		l = np.asarray([10**i for i in L])

		L = np.log10(l)
		L -= np.log10(3.8E33)



		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one

		scale = (np.nanmedian(F1[B4])-np.nanmedian(F1[B4])*0.5)/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5

		temp_L_bol_all = temp_L_bol*scale
		# temp_L_bol_all = np.nanmedian(temp_L_bol3)
		
		F1 = np.log10(F1)

		# temp_L_bol3 = 0
		# temp_L_bol4 = 0
		# temp_L_bol5 = 0
		# temp_L_bol_GOALS = 0
		# temp_L_bol_all = 0

		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)

		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		fir_frac1 = fir_frac[B1]
		fir_frac2 = fir_frac[B2]
		fir_frac3 = fir_frac[B3]
		fir_frac4 = fir_frac[B4]
		fir_frac5 = fir_frac[B5]

		# L11 = L[B1] - temp_L_bol_all
		# L21 = L[B2] - temp_L_bol_all
		# L31 = L[B3] - temp_L_bol_all
		# L41 = L[B4] - temp_L_bol_all
		# L51 = L[B5] - temp_L_bol_all
		# print(np.log10(temp_L_bol_all))

		temp_L_bol_all = 1.44124431E11


		print(np.log10(temp_L_bol_all))


		lx = np.asarray([10**i for i in Lx])
		Lx = lx

		L1_1 = L[B1[(z1 < 0.5)&(z1 > 0.3)]]
		L2_1 = L[B2[(z2 < 0.5)&(z2 > 0.3)]]
		L3_1 = L[B3[(z3 < 0.5)&(z3 > 0.3)]]
		L4_1 = L[B4[(z4 < 0.5)&(z4 > 0.3)]]
		L5_1 = L[B5[(z5 < 0.5)&(z5 > 0.3)]]

		Lx1_1 = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]]
		Lx2_1 = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]]
		Lx3_1 = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]]
		Lx4_1 = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]]
		Lx5_1 = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]]

		fir_frac1_1 = fir_frac[B1[(z1 < 0.5)&(z1 > 0.3)]]
		fir_frac2_1 = fir_frac[B2[(z2 < 0.5)&(z2 > 0.3)]]
		fir_frac3_1 = fir_frac[B3[(z3 < 0.5)&(z3 > 0.3)]]
		fir_frac4_1 = fir_frac[B4[(z4 < 0.5)&(z4 > 0.3)]]
		fir_frac5_1 = fir_frac[B5[(z5 < 0.5)&(z5 > 0.3)]]

		up_check1_1,up_check2_1,up_check3_1,up_check4_1,up_check5_1 = up_check[B1[(z1 < 0.5)&(z1 > 0.3)]],up_check[B2[(z2 < 0.5)&(z2 > 0.3)]],up_check[B3[(z3 < 0.5)&(z3 > 0.3)]],up_check[B4[(z4 < 0.5)&(z4 > 0.3)]],up_check[B5[(z5 < 0.5)&(z5 > 0.3)]]


		L1_2 = L[B1[(z1 < 0.8)&(z1 > 0.6)]]
		L2_2 = L[B2[(z2 < 0.8)&(z2 > 0.6)]]
		L3_2 = L[B3[(z3 < 0.8)&(z3 > 0.6)]]
		L4_2 = L[B4[(z4 < 0.8)&(z4 > 0.6)]]
		L5_2 = L[B5[(z5 < 0.8)&(z5 > 0.6)]]

		Lx1_2 = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]]
		Lx2_2 = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]]
		Lx3_2 = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]]
		Lx4_2 = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]]
		Lx5_2 = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]]

		fir_frac1_2 = fir_frac[B1[(z1 < 0.8)&(z1 > 0.6)]]
		fir_frac2_2 = fir_frac[B2[(z2 < 0.8)&(z2 > 0.6)]]
		fir_frac3_2 = fir_frac[B3[(z3 < 0.8)&(z3 > 0.6)]]
		fir_frac4_2 = fir_frac[B4[(z4 < 0.8)&(z4 > 0.6)]]
		fir_frac5_2 = fir_frac[B5[(z5 < 0.8)&(z5 > 0.6)]]

		up_check1_2,up_check2_2,up_check3_2,up_check4_2,up_check5_2 = up_check[B1[(z1 < 0.8)&(z1 > 0.6)]],up_check[B2[(z2 < 0.8)&(z2 > 0.6)]],up_check[B3[(z3 < 0.8)&(z3 > 0.6)]],up_check[B4[(z4 < 0.8)&(z4 > 0.6)]],up_check[B5[(z5 < 0.8)&(z5 > 0.6)]]


		L1_3 = L[B1[(z1 < 1.1)&(z1 > 0.9)]]
		L2_3 = L[B2[(z2 < 1.1)&(z2 > 0.9)]]
		L3_3 = L[B3[(z3 < 1.1)&(z3 > 0.9)]]
		L4_3 = L[B4[(z4 < 1.1)&(z4 > 0.9)]]
		L5_3 = L[B5[(z5 < 1.1)&(z5 > 0.9)]]

		Lx1_3 = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]]
		Lx2_3 = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]]
		Lx3_3 = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]]
		Lx4_3 = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]]
		Lx5_3 = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]]

		fir_frac1_3 = fir_frac[B1[(z1 < 1.1)&(z1 > 0.9)]]
		fir_frac2_3 = fir_frac[B2[(z2 < 1.1)&(z2 > 0.9)]]
		fir_frac3_3 = fir_frac[B3[(z3 < 1.1)&(z3 > 0.9)]]
		fir_frac4_3 = fir_frac[B4[(z4 < 1.1)&(z4 > 0.9)]]
		fir_frac5_3 = fir_frac[B5[(z5 < 1.1)&(z5 > 0.9)]]


		up_check1_3,up_check2_3,up_check3_3,up_check4_3,up_check5_3 = up_check[B1[(z1 < 1.1)&(z1 > 0.9)]],up_check[B2[(z2 < 1.1)&(z2 > 0.9)]],up_check[B3[(z3 < 1.1)&(z3 > 0.9)]],up_check[B4[(z4 < 1.1)&(z4 > 0.9)]],up_check[B5[(z5 < 1.1)&(z5 > 0.9)]]


		L1_1_sub = L[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		L2_1_sub = L[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		L3_1_sub = L[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		L4_1_sub = L[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		L5_1_sub = L[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		# Lx1_1_sub = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		# Lx2_1_sub = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		# Lx3_1_sub = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		# Lx4_1_sub = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		# Lx5_1_sub = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		L1_2_sub = L[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		L2_2_sub = L[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		L3_2_sub = L[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		L4_2_sub = L[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		L5_2_sub = L[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		# Lx1_2_sub = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		# Lx2_2_sub = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		# Lx3_2_sub = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		# Lx4_2_sub = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		# Lx5_2_sub = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		L1_3_sub = L[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		L2_3_sub = L[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		L3_3_sub = L[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		L4_3_sub = L[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		L5_3_sub = L[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all

		# Lx1_3_sub = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		# Lx2_3_sub = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		# Lx3_3_sub = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		# Lx4_3_sub = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		# Lx5_3_sub = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all



		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'
		print(np.log10(Lx1_1))
		print('Lbol_Lx',self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==0])))
		print('Lbol',np.log10(L1_1[up_check1_1==0]))


		fig = plt.figure(figsize=(20,12))
		gs = fig.add_gridspec(nrows=2, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.05,right=0.95,top=0.9,bottom=0.15)

		ax1 = plt.subplot(gs[0,0])
		ax1.set_title('0.3 < z < 0.5')
		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==0])),np.log10(L1_1[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==0])),np.log10(L2_1[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==0])),np.log10(L3_1[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==0])),np.log10(L4_1[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==0])),np.log10(L5_1[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15])),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15])),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15])),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15])),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15])),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15])),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15])),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15])),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15])),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15])),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax1.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax1.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax1 = ax1.secondary_xaxis('top',functions=(ergs, solar))
		secax1.set_xlabel(r' ')
		ax1.set_ylabel(r'Total log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		ax1.set_xticklabels([])
		ax1.set_xlim(9, 14)
		ax1.set_ylim(9,14)

		ax2 = plt.subplot(gs[1,0])
		ax2.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==0])),np.log10(L1_1_sub[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==0])),np.log10(L2_1_sub[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==0])),np.log10(L3_1_sub[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==0])),np.log10(L4_1_sub[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==0])),np.log10(L5_1_sub[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax2.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15])),np.log10(L1_1_sub[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15])),np.log10(L1_1_sub[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax2.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15])),np.log10(L2_1_sub[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax2.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15])),np.log10(L2_1_sub[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15])),np.log10(L3_1_sub[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15])),np.log10(L3_1_sub[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15])),np.log10(L4_1_sub[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15])),np.log10(L4_1_sub[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax2.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15])),np.log10(L5_1_sub[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax2.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15])),np.log10(L5_1_sub[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		
		ax2.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax2.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax2 = ax2.secondary_xaxis('top',functions=(ergs, solar))
		ax2.set_ylabel(r'Galaxy Subtracted log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		secax2.set_xticklabels([])
		ax2.set_xlim(9,14)
		ax2.set_ylim(9,14)


		ax3 = plt.subplot(gs[0,1])
		ax3.set_title('0.6 < z < 0.8')
		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==0])),np.log10(L1_2[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==0])),np.log10(L2_2[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==0])),np.log10(L3_2[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==0])),np.log10(L4_2[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==0])),np.log10(L5_2[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15])),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15])),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15])),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15])),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15])),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15])),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15])),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15])),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15])),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15])),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		
		ax3.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax3.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax3 = ax3.secondary_xaxis('top',functions=(ergs, solar))
		secax3.set_xlabel(r'Duras log $\mathrm{L}_{\mathrm{bol}}$ [erg/s]')
		ax3.set_xticklabels([])
		ax3.set_yticklabels([])
		ax3.set_xlim(9,14)
		ax3.set_ylim(9,14)


		ax4 = plt.subplot(gs[1,1])
		ax4.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==0])),np.log10(L1_2_sub[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==0])),np.log10(L2_2_sub[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==0])),np.log10(L3_2_sub[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==0])),np.log10(L4_2_sub[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==0])),np.log10(L5_2_sub[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax4.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15])),np.log10(L1_2_sub[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15])),np.log10(L1_2_sub[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax4.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15])),np.log10(L2_2_sub[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax4.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15])),np.log10(L2_2_sub[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15])),np.log10(L3_2_sub[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15])),np.log10(L3_2_sub[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15])),np.log10(L4_2_sub[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15])),np.log10(L4_2_sub[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax4.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15])),np.log10(L5_2_sub[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax4.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15])),np.log10(L5_2_sub[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax4.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax4.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax4 = ax4.secondary_xaxis('top',functions=(ergs, solar))
		ax4.set_xlabel(r'Duras log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		ax4.set_yticklabels([])
		secax4.set_xticklabels([])
		ax4.set_xlim(9,14)
		ax4.set_ylim(9,14)


		ax5 = plt.subplot(gs[0,2])
		ax5.set_title('0.9 < z < 1.1')
		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==0])),np.log10(L1_3[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==0])),np.log10(L2_3[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==0])),np.log10(L3_3[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==0])),np.log10(L4_3[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==0])),np.log10(L5_3[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15])),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15])),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15])),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15])),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15])),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15])),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15])),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15])),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15])),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15])),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		
		ax5.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax5.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax5 = ax5.secondary_xaxis('top',functions=(ergs, solar))
		secax5.set_xlabel(r' ')
		ax5.set_xticklabels([])
		ax5.set_yticklabels([])
		ax5.set_xlim(9,14)
		ax5.set_ylim(9,14)

		ax6 = plt.subplot(gs[1,2])
		ax6.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==0])),np.log10(L1_3_sub[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==0])),np.log10(L2_3_sub[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==0])),np.log10(L3_3_sub[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==0])),np.log10(L4_3_sub[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==0])),np.log10(L5_3_sub[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax6.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15])),np.log10(L1_3_sub[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15])),np.log10(L1_3_sub[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax6.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15])),np.log10(L2_3_sub[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax6.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15])),np.log10(L2_3_sub[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15])),np.log10(L3_3_sub[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15])),np.log10(L3_3_sub[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15])),np.log10(L4_3_sub[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15])),np.log10(L4_3_sub[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax6.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15])),np.log10(L5_3_sub[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax6.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15])),np.log10(L5_3_sub[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax6.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax6.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')

		secax6 = ax6.secondary_xaxis('top',functions=(ergs, solar))
		ax6.set_yticklabels([])
		secax6.set_xticklabels([])
		ax6.set_xlim(9,14)
		ax6.set_ylim(9,14)

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()
		ax6.grid()
		plt.savefig('/Users/connor_auge/Desktop/Paper/Lbol_Lbol_6zbins'+savestring+'.pdf')
		plt.show()


	def Lbol_Lbol_3panel(self,savestring,Lx,L,F1,uv_slope,mir_slope1,mir_slope2,spec_z,up_check=None,fir_frac=None):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3


		L[L > 100] = np.nan
		# Lx[Lx > 45.5] = np.nan


		l = np.asarray([10**i for i in L])

		L = np.log10(l)
		L -= np.log10(3.8E33)



		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		z1 = spec_z[B1]
		z2 = spec_z[B2]
		z3 = spec_z[B3]
		z4 = spec_z[B4]
		z5 = spec_z[B5]

		# Templete scaling
		temp_L_one = 1.1963983003219803E43

		scale_3 = F1[B3]/temp_L_one
		scale_4 = F1[B4]/temp_L_one
		scale_5 = F1[B5]/temp_L_one

		scale = (np.nanmedian(F1[B4])-np.nanmedian(F1[B4])*0.5)/temp_L_one

		temp_L_bol = 1E10
		
		temp_L_bol3 = temp_L_bol*scale_3
		temp_L_bol4 = temp_L_bol*scale_4
		temp_L_bol5 = temp_L_bol*scale_5

		temp_L_bol_all = temp_L_bol*scale
		# temp_L_bol_all = np.nanmedian(temp_L_bol3)
		
		F1 = np.log10(F1)

		# temp_L_bol3 = 0
		# temp_L_bol4 = 0
		# temp_L_bol5 = 0
		# temp_L_bol_GOALS = 0
		# temp_L_bol_all = 0

		L = l
		L /= 3.8E33
		Lx -= np.log10(3.8E33)

		fir_frac = np.asarray(fir_frac)
		fir_frac /= 3.8E33
		fir_frac1 = fir_frac[B1]
		fir_frac2 = fir_frac[B2]
		fir_frac3 = fir_frac[B3]
		fir_frac4 = fir_frac[B4]
		fir_frac5 = fir_frac[B5]

		# L11 = L[B1] - temp_L_bol_all
		# L21 = L[B2] - temp_L_bol_all
		# L31 = L[B3] - temp_L_bol_all
		# L41 = L[B4] - temp_L_bol_all
		# L51 = L[B5] - temp_L_bol_all
		# print(np.log10(temp_L_bol_all))

		temp_L_bol_all = 1.44124431E11


		print(np.log10(temp_L_bol_all))


		lx = np.asarray([10**i for i in Lx])
		Lx = lx

		L1_1 = L[B1[(z1 < 0.5)&(z1 > 0.3)]]
		L2_1 = L[B2[(z2 < 0.5)&(z2 > 0.3)]]
		L3_1 = L[B3[(z3 < 0.5)&(z3 > 0.3)]]
		L4_1 = L[B4[(z4 < 0.5)&(z4 > 0.3)]]
		L5_1 = L[B5[(z5 < 0.5)&(z5 > 0.3)]]

		Lx1_1 = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]]
		Lx2_1 = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]]
		Lx3_1 = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]]
		Lx4_1 = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]]
		Lx5_1 = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]]

		fir_frac1_1 = fir_frac[B1[(z1 < 0.5)&(z1 > 0.3)]]
		fir_frac2_1 = fir_frac[B2[(z2 < 0.5)&(z2 > 0.3)]]
		fir_frac3_1 = fir_frac[B3[(z3 < 0.5)&(z3 > 0.3)]]
		fir_frac4_1 = fir_frac[B4[(z4 < 0.5)&(z4 > 0.3)]]
		fir_frac5_1 = fir_frac[B5[(z5 < 0.5)&(z5 > 0.3)]]

		up_check1_1,up_check2_1,up_check3_1,up_check4_1,up_check5_1 = up_check[B1[(z1 < 0.5)&(z1 > 0.3)]],up_check[B2[(z2 < 0.5)&(z2 > 0.3)]],up_check[B3[(z3 < 0.5)&(z3 > 0.3)]],up_check[B4[(z4 < 0.5)&(z4 > 0.3)]],up_check[B5[(z5 < 0.5)&(z5 > 0.3)]]


		L1_2 = L[B1[(z1 < 0.8)&(z1 > 0.6)]]
		L2_2 = L[B2[(z2 < 0.8)&(z2 > 0.6)]]
		L3_2 = L[B3[(z3 < 0.8)&(z3 > 0.6)]]
		L4_2 = L[B4[(z4 < 0.8)&(z4 > 0.6)]]
		L5_2 = L[B5[(z5 < 0.8)&(z5 > 0.6)]]

		Lx1_2 = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]]
		Lx2_2 = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]]
		Lx3_2 = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]]
		Lx4_2 = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]]
		Lx5_2 = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]]

		fir_frac1_2 = fir_frac[B1[(z1 < 0.8)&(z1 > 0.6)]]
		fir_frac2_2 = fir_frac[B2[(z2 < 0.8)&(z2 > 0.6)]]
		fir_frac3_2 = fir_frac[B3[(z3 < 0.8)&(z3 > 0.6)]]
		fir_frac4_2 = fir_frac[B4[(z4 < 0.8)&(z4 > 0.6)]]
		fir_frac5_2 = fir_frac[B5[(z5 < 0.8)&(z5 > 0.6)]]

		up_check1_2,up_check2_2,up_check3_2,up_check4_2,up_check5_2 = up_check[B1[(z1 < 0.8)&(z1 > 0.6)]],up_check[B2[(z2 < 0.8)&(z2 > 0.6)]],up_check[B3[(z3 < 0.8)&(z3 > 0.6)]],up_check[B4[(z4 < 0.8)&(z4 > 0.6)]],up_check[B5[(z5 < 0.8)&(z5 > 0.6)]]


		L1_3 = L[B1[(z1 < 1.1)&(z1 > 0.9)]]
		L2_3 = L[B2[(z2 < 1.1)&(z2 > 0.9)]]
		L3_3 = L[B3[(z3 < 1.1)&(z3 > 0.9)]]
		L4_3 = L[B4[(z4 < 1.1)&(z4 > 0.9)]]
		L5_3 = L[B5[(z5 < 1.1)&(z5 > 0.9)]]

		Lx1_3 = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]]
		Lx2_3 = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]]
		Lx3_3 = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]]
		Lx4_3 = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]]
		Lx5_3 = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]]

		fir_frac1_3 = fir_frac[B1[(z1 < 1.1)&(z1 > 0.9)]]
		fir_frac2_3 = fir_frac[B2[(z2 < 1.1)&(z2 > 0.9)]]
		fir_frac3_3 = fir_frac[B3[(z3 < 1.1)&(z3 > 0.9)]]
		fir_frac4_3 = fir_frac[B4[(z4 < 1.1)&(z4 > 0.9)]]
		fir_frac5_3 = fir_frac[B5[(z5 < 1.1)&(z5 > 0.9)]]


		up_check1_3,up_check2_3,up_check3_3,up_check4_3,up_check5_3 = up_check[B1[(z1 < 1.1)&(z1 > 0.9)]],up_check[B2[(z2 < 1.1)&(z2 > 0.9)]],up_check[B3[(z3 < 1.1)&(z3 > 0.9)]],up_check[B4[(z4 < 1.1)&(z4 > 0.9)]],up_check[B5[(z5 < 1.1)&(z5 > 0.9)]]


		L1_1_sub = L[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		L2_1_sub = L[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		L3_1_sub = L[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		L4_1_sub = L[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		L5_1_sub = L[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		# Lx1_1_sub = Lx[B1[(z1 < 0.5)&(z1 > 0.3)]] - temp_L_bol_all
		# Lx2_1_sub = Lx[B2[(z2 < 0.5)&(z2 > 0.3)]] - temp_L_bol_all
		# Lx3_1_sub = Lx[B3[(z3 < 0.5)&(z3 > 0.3)]] - temp_L_bol_all
		# Lx4_1_sub = Lx[B4[(z4 < 0.5)&(z4 > 0.3)]] - temp_L_bol_all
		# Lx5_1_sub = Lx[B5[(z5 < 0.5)&(z5 > 0.3)]] - temp_L_bol_all

		L1_2_sub = L[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		L2_2_sub = L[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		L3_2_sub = L[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		L4_2_sub = L[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		L5_2_sub = L[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		# Lx1_2_sub = Lx[B1[(z1 < 0.8)&(z1 > 0.6)]] - temp_L_bol_all
		# Lx2_2_sub = Lx[B2[(z2 < 0.8)&(z2 > 0.6)]] - temp_L_bol_all
		# Lx3_2_sub = Lx[B3[(z3 < 0.8)&(z3 > 0.6)]] - temp_L_bol_all
		# Lx4_2_sub = Lx[B4[(z4 < 0.8)&(z4 > 0.6)]] - temp_L_bol_all
		# Lx5_2_sub = Lx[B5[(z5 < 0.8)&(z5 > 0.6)]] - temp_L_bol_all

		L1_3_sub = L[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		L2_3_sub = L[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		L3_3_sub = L[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		L4_3_sub = L[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		L5_3_sub = L[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all

		# Lx1_3_sub = Lx[B1[(z1 < 1.1)&(z1 > 0.9)]] - temp_L_bol_all
		# Lx2_3_sub = Lx[B2[(z2 < 1.1)&(z2 > 0.9)]] - temp_L_bol_all
		# Lx3_3_sub = Lx[B3[(z3 < 1.1)&(z3 > 0.9)]] - temp_L_bol_all
		# Lx4_3_sub = Lx[B4[(z4 < 1.1)&(z4 > 0.9)]] - temp_L_bol_all
		# Lx5_3_sub = Lx[B5[(z5 < 1.1)&(z5 > 0.9)]] - temp_L_bol_all



		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		
		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		fig = plt.figure(figsize=(21,7))
		gs = fig.add_gridspec(nrows=1, ncols=3)
		gs.update(wspace=0.05,hspace=0.05) # set the spacing between axes
		gs.update(left=0.07,right=0.93,top=0.83,bottom=0.12)

		ax1 = plt.subplot(gs[0])
		ax1.set_title('0.3 < z < 0.5')
		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==0])),np.log10(L1_1[up_check1_1==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==0])),np.log10(L2_1[up_check2_1==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==0])),np.log10(L3_1[up_check3_1==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==0])),np.log10(L4_1[up_check4_1==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==0])),np.log10(L5_1[up_check5_1==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15])),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15])),np.log10(L1_1[up_check1_1==1][fir_frac1_1[up_check1_1==1]/L1_1[up_check1_1==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15])),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax1.scatter(self.Durras_Lbol(np.log10(Lx2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15])),np.log10(L2_1[up_check2_1==1][fir_frac2_1[up_check2_1==1]/L2_1[up_check2_1==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15])),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15])),np.log10(L3_1[up_check3_1==1][fir_frac3_1[up_check3_1==1]/L3_1[up_check3_1==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15])),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15])),np.log10(L4_1[up_check4_1==1][fir_frac4_1[up_check4_1==1]/L4_1[up_check4_1==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15])),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax1.scatter(self.Durras_Lbol(np.log10(Lx5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15])),np.log10(L5_1[up_check5_1==1][fir_frac5_1[up_check5_1==1]/L5_1[up_check5_1==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)

		ax1.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax1.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax1 = ax1.secondary_xaxis('top',functions=(ergs, solar))
		secax1.set_xlabel(r' ')
		ax1.set_ylabel(r'Total log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# ax1.set_xticklabels([])
		ax1.set_xlim(9, 14)
		ax1.set_ylim(9,14)



		ax3 = plt.subplot(gs[1])
		ax3.set_title('0.6 < z < 0.8')
		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==0])),np.log10(L1_2[up_check1_2==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==0])),np.log10(L2_2[up_check2_2==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==0])),np.log10(L3_2[up_check3_2==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==0])),np.log10(L4_2[up_check4_2==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==0])),np.log10(L5_2[up_check5_2==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15])),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15])),np.log10(L1_2[up_check1_2==1][fir_frac1_2[up_check1_2==1]/L1_2[up_check1_2==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15])),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax3.scatter(self.Durras_Lbol(np.log10(Lx2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15])),np.log10(L2_2[up_check2_2==1][fir_frac2_2[up_check2_2==1]/L2_2[up_check2_2==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15])),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15])),np.log10(L3_2[up_check3_2==1][fir_frac3_2[up_check3_2==1]/L3_2[up_check3_2==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15])),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15])),np.log10(L4_2[up_check4_2==1][fir_frac4_2[up_check4_2==1]/L4_2[up_check4_2==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15])),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax3.scatter(self.Durras_Lbol(np.log10(Lx5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15])),np.log10(L5_2[up_check5_2==1][fir_frac5_2[up_check5_2==1]/L5_2[up_check5_2==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		
		ax3.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax3.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax3 = ax3.secondary_xaxis('top',functions=(ergs, solar))
		secax3.set_xlabel(r'AGN log $\mathrm{L}_{\mathrm{bol}}$ [erg/s]')
		ax3.set_xlabel(r'AGN log $\mathrm{L}_{\mathrm{bol}}$ [L$_{\odot}$]')
		# ax3.set_xticklabels([])
		ax3.set_yticklabels([])
		ax3.set_xlim(9,14)
		ax3.set_ylim(9,14)



		ax5 = plt.subplot(gs[2])
		ax5.set_title('0.9 < z < 1.1')
		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==0])),np.log10(L1_3[up_check1_3==0]),color=c1,edgecolors='k',s=80,alpha=0.9,label='Panel 1',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==0])),np.log10(L2_3[up_check2_3==0]),color=c2,edgecolors='k',s=80,alpha=0.9,label='Panel 2',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==0])),np.log10(L3_3[up_check3_3==0]),color=c3,edgecolors='k',s=80,alpha=0.9,label='Panel 3',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==0])),np.log10(L4_3[up_check4_3==0]),color=c4,edgecolors='k',s=80,alpha=0.9,label='Panel 4',rasterized=True)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==0])),np.log10(L5_3[up_check5_3==0]),color=c5,edgecolors='k',s=80,alpha=0.9,label='Panel 5',rasterized=True)

		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15])),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]<0.15]),color=c1,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15])),np.log10(L1_3[up_check1_3==1][fir_frac1_3[up_check1_3==1]/L1_3[up_check1_3==1]>0.15]),marker='v',color=c1,edgecolors='k',s=60,alpha=0.9)

		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15])),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]<0.15]),color=c2,edgecolors='k',s=60,alpha=0.9)
		ax5.scatter(self.Durras_Lbol(np.log10(Lx2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15])),np.log10(L2_3[up_check2_3==1][fir_frac2_3[up_check2_3==1]/L2_3[up_check2_3==1]>0.15]),marker='v',color=c2,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15])),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]<0.15]),color=c3,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15])),np.log10(L3_3[up_check3_3==1][fir_frac3_3[up_check3_3==1]/L3_3[up_check3_3==1]>0.15]),marker='v',color=c3,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15])),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]<0.15]),color=c4,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15])),np.log10(L4_3[up_check4_3==1][fir_frac4_3[up_check4_3==1]/L4_3[up_check4_3==1]>0.15]),marker='v',color=c4,edgecolors='k',s=60,alpha=0.9)		

		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15])),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]<0.15]),color=c5,edgecolors='k',s=60,alpha=0.9)		
		ax5.scatter(self.Durras_Lbol(np.log10(Lx5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15])),np.log10(L5_3[up_check5_3==1][fir_frac5_3[up_check5_3==1]/L5_3[up_check5_3==1]>0.15]),marker='v',color=c5,edgecolors='k',s=60,alpha=0.9)
		
		ax5.plot(np.linspace(8,16,10),np.linspace(8,16,10),c='k')
		# ax5.plot(np.linspace(8,13,10),self.Durras_Lbol(np.linspace(8,13,10)),color='k',label='Durras et al. Correction')


		secax5 = ax5.secondary_xaxis('top',functions=(ergs, solar))
		secax5.set_xlabel(r' ')
		# ax5.set_xticklabels([])
		ax5.set_yticklabels([])
		ax5.set_xlim(9,14)
		ax5.set_ylim(9,14)

		ax1.grid()
		ax3.grid()
		ax5.grid()
		plt.savefig('/Users/connor_auge/Desktop/Paper/Lbol_Lbol_3zbins'+savestring+'.pdf')
		plt.show()




	def Emission_Scatter_Comp(self,savestring,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,Nh,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis32=None,emis42=None,emis5=None,emis6=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None):
		plt.rcParams['font.size']=20
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		Nh[Nh <= 0] = np.nan

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		B1 = (uv_slope < -0.3)&(mir_slope1 >= -0.1)
		B2 = (uv_slope >= -0.3)&(uv_slope <=0.2)&(mir_slope1 >= -0.2)
		B3 = (uv_slope > 0.2)&(mir_slope1 >= -0.2)
		B4 = (uv_slope > -0.05)&(mir_slope1 < -0.2)&(mir_slope2 > 0.0)
		B5 = (uv_slope > -0.05)&(mir_slope1 < -0.2)&(mir_slope2 <= 0.0)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		zbins1 = (spec_z >= zlim_1)&(spec_z <= zlim_2)
		zbins2 = (spec_z >= zlim_2)&(spec_z <= zlim_3)
		zbins3 = (spec_z >= zlim_3)&(spec_z <= zlim_4)

		up_check1,up_check2,up_check3,up_check4,up_check5 = up_check[B1],up_check[B2],up_check[B3],up_check[B4],up_check[B5]

		f_3 = np.asarray([10**i for i in f3])
		f_3 *= F1
		f3 = np.log10(f_3)

		f_4 = np.asarray([10**i for i in f4])
		f_4 *= F1
		f4 = np.log10(f_4)

		f_1 = np.asarray([10**i for i in f1])
		f_1 *= F1
		f1 = np.log10(f_1)

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		Nh = np.log10(Nh)

		# plt.hist(Nh,bins=np.arange(20,26,0.25))
		# plt.show()

		Nh[(Nh > 24)] = 275
		Nh[(Nh > 23)&(Nh < 24)] = 180
		Nh[(Nh > 22)&(Nh < 23)] = 100
		Nh[(Nh < 22)] = 40

		up_check1 = up_check[zbins1]
		up_check2 = up_check[zbins2]
		up_check3 = up_check[zbins3]

		f4_11 = f4[zbins1][B1[zbins1]] 
		f3_11 = f3[zbins1][B1[zbins1]] 	
		f1_11 = f1[zbins1][B1[zbins1]]
		Nh_11 = Nh[zbins1][B1[zbins1]]
		up_check11 = up_check[zbins1][B1[zbins1]]
		f4_12 = f4[zbins1][B2[zbins1]]
		f3_12 = f3[zbins1][B2[zbins1]]
		f1_12 = f1[zbins1][B2[zbins1]]
		Nh_12 = Nh[zbins1][B2[zbins1]]
		up_check12 = up_check[zbins1][B2[zbins1]]
		f4_13 = f4[zbins1][B3[zbins1]]
		f3_13 = f3[zbins1][B3[zbins1]]  
		f1_13 = f1[zbins1][B3[zbins1]]
		Nh_13 = Nh[zbins1][B3[zbins1]]
		up_check13 = up_check[zbins1][B3[zbins1]]
		f4_14 = f4[zbins1][B4[zbins1]]
		f3_14 = f3[zbins1][B4[zbins1]]
		f1_14 = f1[zbins1][B4[zbins1]]
		Nh_14 = Nh[zbins1][B4[zbins1]]
		up_check14 = up_check[zbins1][B4[zbins1]]
		f4_15 = f4[zbins1][B5[zbins1]]
		f3_15 = f3[zbins1][B5[zbins1]]
		f1_15 = f1[zbins1][B5[zbins1]]
		Nh_15 = Nh[zbins1][B5[zbins1]]
		up_check15 = up_check[zbins1][B5[zbins1]]

		f4_21 = f4[zbins2][B1[zbins2]]
		f3_21 = f3[zbins2][B1[zbins2]]
		f1_21 = f1[zbins2][B1[zbins2]]
		Nh_21 = Nh[zbins2][B1[zbins2]]
		up_check21 = up_check[zbins2][B1[zbins2]]
		f4_22 = f4[zbins2][B2[zbins2]]
		f3_22 = f3[zbins2][B2[zbins2]]
		f1_22 = f1[zbins2][B2[zbins2]]
		Nh_22 = Nh[zbins2][B2[zbins2]]
		up_check22 = up_check[zbins2][B2[zbins2]]
		f4_23 = f4[zbins2][B3[zbins2]]
		f3_23 = f3[zbins2][B3[zbins2]]
		f1_23 = f1[zbins2][B3[zbins2]]
		Nh_23 = Nh[zbins2][B3[zbins2]]
		up_check23 = up_check[zbins2][B3[zbins2]]
		f4_24 = f4[zbins2][B4[zbins2]]
		f3_24 = f3[zbins2][B4[zbins2]]
		f1_24 = f1[zbins2][B4[zbins2]]
		Nh_24 = Nh[zbins2][B4[zbins2]]
		up_check24 = up_check[zbins2][B4[zbins2]]
		f4_25 = f4[zbins2][B5[zbins2]]
		f3_25 = f3[zbins2][B5[zbins2]]
		f1_25 = f1[zbins2][B5[zbins2]]
		Nh_25 = Nh[zbins2][B5[zbins2]]
		up_check25 = up_check[zbins2][B5[zbins2]]

		f4_31 = f4[zbins3][B1[zbins3]]
		f3_31 = f3[zbins3][B1[zbins3]]
		f1_31 = f1[zbins3][B1[zbins3]]
		Nh_31 = Nh[zbins3][B1[zbins3]]
		up_check31 = up_check[zbins3][B1[zbins3]]
		f4_32 = f4[zbins3][B2[zbins3]]
		f3_32 = f3[zbins3][B2[zbins3]]
		f1_32 = f1[zbins3][B2[zbins3]]
		Nh_32 = Nh[zbins3][B2[zbins3]]
		up_check32 = up_check[zbins3][B2[zbins3]]
		f4_33 = f4[zbins3][B3[zbins3]]
		f3_33 = f3[zbins3][B3[zbins3]]
		f1_33 = f1[zbins3][B3[zbins3]]
		Nh_33 = Nh[zbins3][B3[zbins3]]
		up_check33 = up_check[zbins3][B3[zbins3]]
		f4_34 = f4[zbins3][B4[zbins3]]
		f3_34 = f3[zbins3][B4[zbins3]]
		f1_34 = f1[zbins3][B4[zbins3]]
		Nh_34 = Nh[zbins3][B4[zbins3]]
		up_check34 = up_check[zbins3][B4[zbins3]]
		f4_35 = f4[zbins3][B5[zbins3]]
		f3_35 = f3[zbins3][B5[zbins3]]
		f1_35 = f1[zbins3][B5[zbins3]]
		Nh_35 = Nh[zbins3][B5[zbins3]]
		up_check35 = up_check[zbins3][B5[zbins3]]

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		plt.figure(figsize=(18,6.5))
		ax1 = plt.subplot(131)
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))		

		print(np.shape(f3[zbins1][up_check1 == 1]))

		# ax1.quiver(f3_11[up_check11 == 1],f1_11[up_check11 == 1],np.ones(np.shape(f3_11[up_check11 == 1]))*-1, np.zeros(np.shape(f3_11[up_check11 == 1])),facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(f3_12[up_check12 == 1],f1_12[up_check12 == 1],np.ones(np.shape(f3_12[up_check12 == 1]))*-1, np.zeros(np.shape(f3_12[up_check12 == 1])),facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(f3_13[up_check13 == 1],f1_13[up_check13 == 1],np.ones(np.shape(f3_13[up_check13 == 1]))*-1, np.zeros(np.shape(f3_13[up_check13 == 1])),facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(f3_14[up_check14 == 1],f1_14[up_check14 == 1],np.ones(np.shape(f3_14[up_check14 == 1]))*-1, np.zeros(np.shape(f3_14[up_check14 == 1])),facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(f3_15[up_check15 == 1],f1_15[up_check15 == 1],np.ones(np.shape(f3_15[up_check15 == 1]))*-1, np.zeros(np.shape(f3_15[up_check15 == 1])),facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)


		ax1.scatter(f4_11[~np.isnan(Nh_11)],f1_11[~np.isnan(Nh_11)],color=c1,edgecolors='k',s=Nh_11[~np.isnan(Nh_11)],rasterized=True)
		ax1.scatter(f4_12[~np.isnan(Nh_12)],f1_12[~np.isnan(Nh_12)],color=c2,edgecolors='k',s=Nh_12[~np.isnan(Nh_12)],rasterized=True)
		ax1.scatter(f4_13[~np.isnan(Nh_13)],f1_13[~np.isnan(Nh_13)],color=c3,edgecolors='k',s=Nh_13[~np.isnan(Nh_13)],rasterized=True)
		ax1.scatter(f4_14[~np.isnan(Nh_14)],f1_14[~np.isnan(Nh_14)],color=c4,edgecolors='k',s=Nh_14[~np.isnan(Nh_14)],rasterized=True)
		ax1.scatter(f4_15[~np.isnan(Nh_15)],f1_15[~np.isnan(Nh_15)],color=c5,edgecolors='k',s=Nh_15[~np.isnan(Nh_15)],rasterized=True)

		ax1.scatter(f4_11[np.isnan(Nh_11)],f1_11[np.isnan(Nh_11)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True)
		ax1.scatter(f4_12[np.isnan(Nh_12)],f1_12[np.isnan(Nh_12)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True)
		ax1.scatter(f4_13[np.isnan(Nh_13)],f1_13[np.isnan(Nh_13)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True)
		ax1.scatter(f4_14[np.isnan(Nh_14)],f1_14[np.isnan(Nh_14)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True)
		ax1.scatter(f4_15[np.isnan(Nh_15)],f1_15[np.isnan(Nh_15)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True)

		
		# ax1.set_xticklabels([])
		# ax1.set_ylim(-2,2)
		# ax1.set_xlim(-2,2)
		ax1.set_ylim(42,47)
		ax1.set_xlim(42,47)

		# ax1.set_xlabel(r'$\lambda$L$_{\lambda}$ (10$\mu$m) [erg/s]')
		# ax1.set_xlabel(r'$\lambda$L$_{\lambda}$ (100$\mu$m) [erg/s]')
		# ax1.set_xlabel(r'$\lambda$L$_{\lambda}$ (10$\mu$m) [erg/s]')
		ax1.set_ylabel(r'$\lambda$L$_{\lambda}$ (0.25$\mu$m) [erg/s]')
		# ax1.set_ylabel(r'$\lambda$L$_{\lambda}$ (10$\mu$m) [erg/s]')

		# ax1.set_ylabel(r'100$\mu$mL$_{100\mu \mathrm{m}}$/1.0$\mu$mL$_{1.0\mu \mathrm{m}}$')
		# ax1.set_ylabel(r'10$\mu$mL$_{10\mu \mathrm{m}}$/1.0$\mu$mL$_{1\mu \mathrm{m}}$')
		# ax1.set_ylabel(r'0.25$\mu$mL$_{0.25\mu \mathrm{m}}$/1.0$\mu$mL$_{1\mu \mathrm{m}}$')
		# ax1.legend(fontsize=14)

		secax1 = ax1.secondary_xaxis('top', functions=(solar, ergs))
		secax1.set_xlabel(r' ')
		ax1.grid()

		ax2 = plt.subplot(132)
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))

		# ax2.quiver(f3_21[up_check21 == 1],f1_21[up_check21 == 1],np.ones(np.shape(f3_21[up_check21 == 1]))*-1, np.zeros(np.shape(f3_21[up_check21 == 1])),facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(f3_22[up_check22 == 1],f1_22[up_check22 == 1],np.ones(np.shape(f3_22[up_check22 == 1]))*-1, np.zeros(np.shape(f3_22[up_check22 == 1])),facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(f3_23[up_check23 == 1],f1_23[up_check23 == 1],np.ones(np.shape(f3_23[up_check23 == 1]))*-1, np.zeros(np.shape(f3_23[up_check23 == 1])),facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(f3_24[up_check24 == 1],f1_24[up_check24 == 1],np.ones(np.shape(f3_24[up_check24 == 1]))*-1, np.zeros(np.shape(f3_24[up_check24 == 1])),facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(f3_25[up_check25 == 1],f1_25[up_check25 == 1],np.ones(np.shape(f3_25[up_check25 == 1]))*-1, np.zeros(np.shape(f3_25[up_check25 == 1])),facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)


		ax2.scatter(f4_21[~np.isnan(Nh_21)],f1_21[~np.isnan(Nh_21)],color=c1,edgecolors='k',s=Nh_21[~np.isnan(Nh_21)],rasterized=True)
		ax2.scatter(f4_22[~np.isnan(Nh_22)],f1_22[~np.isnan(Nh_22)],color=c2,edgecolors='k',s=Nh_22[~np.isnan(Nh_22)],rasterized=True)
		ax2.scatter(f4_23[~np.isnan(Nh_23)],f1_23[~np.isnan(Nh_23)],color=c3,edgecolors='k',s=Nh_23[~np.isnan(Nh_23)],rasterized=True)
		ax2.scatter(f4_24[~np.isnan(Nh_24)],f1_24[~np.isnan(Nh_24)],color=c4,edgecolors='k',s=Nh_24[~np.isnan(Nh_24)],rasterized=True)
		ax2.scatter(f4_25[~np.isnan(Nh_25)],f1_25[~np.isnan(Nh_25)],color=c5,edgecolors='k',s=Nh_25[~np.isnan(Nh_25)],rasterized=True)

		ax2.scatter(f4_21[np.isnan(Nh_21)],f1_21[np.isnan(Nh_21)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True)
		ax2.scatter(f4_22[np.isnan(Nh_22)],f1_22[np.isnan(Nh_22)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True)
		ax2.scatter(f4_23[np.isnan(Nh_23)],f1_23[np.isnan(Nh_23)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True)
		ax2.scatter(f4_24[np.isnan(Nh_24)],f1_24[np.isnan(Nh_24)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True)
		ax2.scatter(f4_25[np.isnan(Nh_25)],f1_25[np.isnan(Nh_25)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True)

		ax2.scatter(-100,-100,color=c1,edgecolors='k',s=75,label='Panel 1',rasterized=True)
		ax2.scatter(-100,-100,color=c2,edgecolors='k',s=75,label='Panel 2',rasterized=True)
		ax2.scatter(-100,-100,color=c3,edgecolors='k',s=75,label='Panel 3',rasterized=True)
		ax2.scatter(-100,-100,color=c4,edgecolors='k',s=75,label='Panel 4',rasterized=True)
		ax2.scatter(-100,-100,color=c5,edgecolors='k',s=75,label='Panel 5',rasterized=True)

		

		

	
		# ax2.scatter(f3[B1][up_check1 == 1],f1[B1][up_check1 == 1],marker='<',color='b',s=100)
		# ax2.scatter(f3[B2][up_check2 == 1],f1[B2][up_check2 == 1],marker='<',color='purple',s=100)
		# ax2.scatter(f3[B3][up_check3 == 1],f1[B3][up_check3 == 1],marker='<',color='green',s=100)
		# ax2.scatter(f3[B4][up_check4 == 1],f1[B4][up_check4 == 1],marker='<',color='orange',s=100)
		# ax2.scatter(f3[B5][up_check5 == 1],f1[B5][up_check5 == 1],marker='<',color='red',s=100)
		# ax2.scatter(emis32,emis3,color='gray',edgecolors='k',s=100,label='ULIRGs')
		ax2.set_yticklabels([])
		# ax2.set_xticklabels([])
		# ax2.set_ylim(-2,2)
		# ax2.set_xlim(-2,2)
		ax2.set_ylim(42,47)
		ax2.set_xlim(42,47)
		# ax2.set_xlabel(r'$\lambda$L$_{\lambda}$ (100$\mu$m) [erg/s]')
		ax2.set_xlabel(r'$\lambda$L$_{\lambda}$ (10$\mu$m) [erg/s]')
		# ax2.set_ylabel(r'$\lambda$L$_{\lambda}$ (0.25$\mu$m) [erg/s]')
		# ax2.set_xlabel(r'100$\mu$mL$_{100\mu \mathrm{m}}$/1.0$\mu$mL$_{1.0\mu \mathrm{m}}$')
		# ax2.set_xlabel(r'10$\mu$mL$_{10\mu \mathrm{m}}$/1.0$\mu$mL$_{1\mu \mathrm{m}}$')
		# ax2.set_xlabel(r'0.25$\mu$mL$_{0.25\mu \mathrm{m}}$/1.0$\mu$mL$_{1\mu \mathrm{m}}$')
		ax2.legend(fontsize=14)
		secax2 = ax2.secondary_xaxis('top', functions=(solar, ergs))	
		secax2.set_xlabel(r'$\lambda$ L$_\lambda$ (10$\mu$m) [L$_{\odot}$]')

		ax2.grid()

		ax3 = plt.subplot(133)
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))

		# ax3.quiver(f3_31[up_check31 == 1],f1_31[up_check31 == 1],np.ones(np.shape(f3_31[up_check31 == 1]))*-1, np.zeros(np.shape(f3_31[up_check31 == 1])),facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(f3_32[up_check32 == 1],f1_32[up_check32 == 1],np.ones(np.shape(f3_32[up_check32 == 1]))*-1, np.zeros(np.shape(f3_32[up_check32 == 1])),facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(f3_33[up_check33 == 1],f1_33[up_check33 == 1],np.ones(np.shape(f3_33[up_check33 == 1]))*-1, np.zeros(np.shape(f3_33[up_check33 == 1])),facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(f3_34[up_check34 == 1],f1_34[up_check34 == 1],np.ones(np.shape(f3_34[up_check34 == 1]))*-1, np.zeros(np.shape(f3_34[up_check34 == 1])),facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(f3_35[up_check35 == 1],f1_35[up_check35 == 1],np.ones(np.shape(f3_35[up_check35 == 1]))*-1, np.zeros(np.shape(f3_35[up_check35 == 1])),facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)

		ax3.scatter(f4_31[~np.isnan(Nh_31)],f1_31[~np.isnan(Nh_31)],color=c1,edgecolors='k',s=Nh_31[~np.isnan(Nh_31)],rasterized=True)
		ax3.scatter(f4_32[~np.isnan(Nh_32)],f1_32[~np.isnan(Nh_32)],color=c2,edgecolors='k',s=Nh_32[~np.isnan(Nh_32)],rasterized=True)
		ax3.scatter(f4_33[~np.isnan(Nh_33)],f1_33[~np.isnan(Nh_33)],color=c3,edgecolors='k',s=Nh_33[~np.isnan(Nh_33)],rasterized=True)
		ax3.scatter(f4_34[~np.isnan(Nh_34)],f1_34[~np.isnan(Nh_34)],color=c4,edgecolors='k',s=Nh_34[~np.isnan(Nh_34)],rasterized=True)
		ax3.scatter(f4_35[~np.isnan(Nh_35)],f1_35[~np.isnan(Nh_35)],color=c5,edgecolors='k',s=Nh_35[~np.isnan(Nh_35)],rasterized=True)

		ax3.scatter(f4_31[np.isnan(Nh_31)],f1_31[np.isnan(Nh_31)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True)
		ax3.scatter(f4_32[np.isnan(Nh_32)],f1_32[np.isnan(Nh_32)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True)
		ax3.scatter(f4_33[np.isnan(Nh_33)],f1_33[np.isnan(Nh_33)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True)
		ax3.scatter(f4_34[np.isnan(Nh_34)],f1_34[np.isnan(Nh_34)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True)
		ax3.scatter(f4_35[np.isnan(Nh_35)],f1_35[np.isnan(Nh_35)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True)

		ax3.scatter(-100,-100,marker='s',color='k',s=75,label=r'No N$_{\mathrm{H}}$ limits',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=40,label=r'log N$_{\mathrm{H}}$ < 22',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=100,label=r'22 < log N$_{\mathrm{H}}$ < 23',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=180,label=r'23 < log N$_{\mathrm{H}}$ < 24',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=275,label=r'24 < log N$_{\mathrm{H}}$',rasterized=True)

		

		ax1.legend(loc='upper left',fontsize=14)

		ax3.legend(fontsize=14)


		# ax3.scatter(f3[B1][up_check1 == 1],f4[B1][up_check1 == 1],marker='<',color='b',s=100)
		# ax3.scatter(f3[B2][up_check2 == 1],f4[B2][up_check2 == 1],marker='<',color='purple',s=100)
		# ax3.scatter(f3[B3][up_check3 == 1],f4[B3][up_check3 == 1],marker='<',color='green',s=100)
		# ax3.scatter(f3[B4][up_check4 == 1],f4[B4][up_check4 == 1],marker='<',color='orange',s=100)
		# ax3.scatter(f3[B5][up_check5 == 1],f4[B5][up_check5 == 1],marker='<',color='red',s=100)

		# ax3.scatter(emis42,emis3,color='gray',edgecolors='k',s=100,label='ULIRGs')
		ax3.set_yticklabels([])
		# ax3.set_xticklabels([])
		# ax3.set_ylim(-2,2)
		# ax3.set_xlim(-2,2)
		ax3.set_ylim(42,47)
		ax3.set_xlim(42,47)
		# ax3.legend(fontsize=14)

		secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
		secax3.set_xlabel(r' ')
		secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
		secax3.set_ylabel(r'$\lambda$ L$_\lambda$ (0.25$\mu$m) [L$_{\odot}$]')
		ax3.grid()

		
		plt.tight_layout()

		plt.savefig('/Users/connor_auge/Desktop/New_runSED/'+savestring+'.pdf')
		plt.show()
	
	
	def Lx_Scatter_Comp(self,savestring,Lx,L,Fx1,Fx2,Fx3,emis1,emis2,f1,f2,f3,f4,Nh,F1=None,F12=None,F13=None,F2=None,emis3=None,emis4=None,emis32=None,emis42=None,emis5=None,emis6=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None,up_check=None):
		plt.rcParams['font.size']=20
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		Nh[Nh <= 0] = np.nan


		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		B1 = (uv_slope < -0.3)&(mir_slope1 >= -0.1)
		B2 = (uv_slope >= -0.3)&(uv_slope <=0.2)&(mir_slope1 >= -0.2)
		B3 = (uv_slope > 0.2)&(mir_slope1 >= -0.2)
		B4 = (uv_slope > -0.05)&(mir_slope1 < -0.2)&(mir_slope2 > 0.0)
		B5 = (uv_slope > -0.05)&(mir_slope1 < -0.2)&(mir_slope2 <= 0.0)

		zlim_1 = 0.0
		zlim_2 = 0.6
		zlim_3 = 0.9
		zlim_4 = 1.2

		zbins1 = (spec_z >= zlim_1)&(spec_z <= zlim_2)
		zbins2 = (spec_z >= zlim_2)&(spec_z <= zlim_3)
		zbins3 = (spec_z >= zlim_3)&(spec_z <=zlim_4)

		up_check1,up_check2,up_check3,up_check4,up_check5 = up_check[B1],up_check[B2],up_check[B3],up_check[B4],up_check[B5]


		f_1 = np.asarray([10**i for i in f1])
		f_2 = np.asarray([10**i for i in f2])
		f_3 = np.asarray([10**i for i in f3])
		f_4 = np.asarray([10**i for i in f4])

		f1 = np.log10(f_1*F1)
		f2 = np.log10(f_2*F1)
		f3 = np.log10(f_3*F1)
		f4 = np.log10(f_4*F1)



		lx = np.asarray([10**i for i in Lx])
		l = np.asarray([10**i for i in L])
		# l -= (10**44.66)

		# Lx = np.log10(lx/F1)
		# L = np.log10(l/F1)
		# Lx = np.log10(lx) + np.log10(3.8E33)
		L = np.log10(l)

		# L -= np.log10(3.6E33)


		emis3 = np.asarray(emis3)
		emis4 = np.asarray(emis4)
		emis32 = np.asarray(emis32)
		emis42 = np.asarray(emis42)
		emis5 = np.asarray(emis5)
		emis6 = np.asarray(emis6)

		fig = plt.figure(figsize=(18, 6.5))
		# plt.suptitle('0.3 < z < 1.1')
		ax1 = plt.subplot(131)
		# z = np.polyfit(Lx[zbins1],f1[zbins1],1)
		# p1 = np.poly1d(z)

		# z = np.polyfit(Lx[zbins1][B1[zbins1]],f1[zbins1][B1[zbins1]],1)
		# p12 = np.poly1d(z)
		# x = np.linspace(42,48,5)

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		Nh = np.log10(Nh)

		Nh[(Nh > 24)] = 275
		Nh[(Nh > 23) & (Nh < 24)] = 180
		Nh[(Nh > 22) & (Nh < 23)] = 100
		Nh[(Nh < 22)] = 40

		Lx_11 = Lx[zbins1][B1[zbins1]]
		f1_11 = f1[zbins1][B1[zbins1]]
		f4_11 = f4[zbins1][B1[zbins1]]
		f3_11 = f3[zbins1][B1[zbins1]] 	
		Nh_11 = Nh[zbins1][B1[zbins1]]
		up_check11 = up_check[zbins1][B1[zbins1]]
		Lx_12 = Lx[zbins1][B2[zbins1]]
		f1_12 = f1[zbins1][B2[zbins1]]
		f4_12 = f4[zbins1][B2[zbins1]]
		f3_12 = f3[zbins1][B2[zbins1]]
		Nh_12 = Nh[zbins1][B2[zbins1]]
		up_check12 = up_check[zbins1][B2[zbins1]]
		Lx_13 = Lx[zbins1][B3[zbins1]]
		f1_13 = f1[zbins1][B3[zbins1]]
		f4_13 = f4[zbins1][B3[zbins1]]
		f3_13 = f3[zbins1][B3[zbins1]] 
		Nh_13 = Nh[zbins1][B3[zbins1]]
		up_check13 = up_check[zbins1][B3[zbins1]]
		Lx_14 = Lx[zbins1][B4[zbins1]]
		f1_14 = f1[zbins1][B4[zbins1]]
		f4_14 = f4[zbins1][B4[zbins1]]
		f3_14 = f3[zbins1][B4[zbins1]]
		Nh_14 = Nh[zbins1][B4[zbins1]]
		up_check14 = up_check[zbins1][B4[zbins1]]
		Lx_15 = Lx[zbins1][B5[zbins1]]
		f1_15 = f1[zbins1][B5[zbins1]]
		f4_15 = f4[zbins1][B5[zbins1]]
		f3_15 = f3[zbins1][B5[zbins1]]
		Nh_15 = Nh[zbins1][B5[zbins1]]
		up_check15 = up_check[zbins1][B5[zbins1]]

		Lx_21 = Lx[zbins2][B1[zbins2]]
		f1_21 = f1[zbins2][B1[zbins2]]
		f4_21 = f4[zbins2][B1[zbins2]]
		f3_21 = f3[zbins2][B1[zbins2]]
		Nh_21 = Nh[zbins2][B1[zbins2]]
		up_check21 = up_check[zbins2][B1[zbins2]]
		Lx_22 = Lx[zbins2][B2[zbins2]]
		f1_22 = f1[zbins2][B2[zbins2]]
		f4_22 = f4[zbins2][B2[zbins2]]
		f3_22 = f3[zbins2][B2[zbins2]]
		Nh_22 = Nh[zbins2][B2[zbins2]]
		up_check22 = up_check[zbins2][B2[zbins2]]
		Lx_23 = Lx[zbins2][B3[zbins2]]
		f1_23 = f1[zbins2][B3[zbins2]]
		f4_23 = f4[zbins2][B3[zbins2]]
		f3_23 = f3[zbins2][B3[zbins2]]
		Nh_23 = Nh[zbins2][B3[zbins2]]
		up_check23 = up_check[zbins2][B3[zbins2]]
		Lx_24 = Lx[zbins2][B4[zbins2]]
		f1_24 = f1[zbins2][B4[zbins2]]
		f4_24 = f4[zbins2][B4[zbins2]]
		f3_24 = f3[zbins2][B4[zbins2]]
		Nh_24 = Nh[zbins2][B4[zbins2]]
		up_check24 = up_check[zbins2][B4[zbins2]]
		Lx_25 = Lx[zbins2][B5[zbins2]]
		f1_25 = f1[zbins2][B5[zbins2]]
		f4_25 = f4[zbins2][B5[zbins2]]
		f3_25 = f3[zbins2][B5[zbins2]]
		Nh_25 = Nh[zbins2][B5[zbins2]]
		up_check25 = up_check[zbins2][B5[zbins2]]

		Lx_31 = Lx[zbins3][B1[zbins3]]
		f1_31 = f1[zbins3][B1[zbins3]]
		f4_31 = f4[zbins3][B1[zbins3]]
		f3_31 = f3[zbins3][B1[zbins3]]
		Nh_31 = Nh[zbins3][B1[zbins3]]
		up_check31 = up_check[zbins3][B1[zbins3]]
		Lx_32 = Lx[zbins3][B2[zbins3]]
		f1_32 = f1[zbins3][B2[zbins3]]
		f4_32 = f4[zbins3][B2[zbins3]]
		f3_32 = f3[zbins3][B2[zbins3]]
		Nh_32 = Nh[zbins3][B2[zbins3]]
		up_check32 = up_check[zbins3][B2[zbins3]]
		Lx_33 = Lx[zbins3][B3[zbins3]]
		f1_33 = f1[zbins3][B3[zbins3]]
		f4_33 = f4[zbins3][B3[zbins3]]
		f3_33 = f3[zbins3][B3[zbins3]]
		Nh_33 = Nh[zbins3][B3[zbins3]]
		up_check33 = up_check[zbins3][B3[zbins3]]
		Lx_34 = Lx[zbins3][B4[zbins3]]
		f1_34 = f1[zbins3][B4[zbins3]]
		f4_34 = f4[zbins3][B4[zbins3]]
		f3_34 = f3[zbins3][B4[zbins3]]
		Nh_34 = Nh[zbins3][B4[zbins3]]
		up_check34 = up_check[zbins3][B4[zbins3]]
		Lx_35 = Lx[zbins3][B5[zbins3]]
		f1_35 = f1[zbins3][B5[zbins3]]
		f4_35 = f4[zbins3][B5[zbins3]]
		f3_35 = f3[zbins3][B5[zbins3]]
		Nh_35 = Nh[zbins3][B5[zbins3]]
		up_check35 = up_check[zbins3][B5[zbins3]]

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		# print(Lx_11[~np.isnan(Nh_11)], f1_11[~np.isnan(Nh_11)])

		xp = np.linspace(43,46)	
		ax1.set_title(str(zlim_1)+' < z < '+str(zlim_2))

		fit_z1 = np.polyfit(Lx[zbins1],f4[zbins1],1)
		fit_p1 = np.poly1d(fit_z1)

		fit_z12 = np.polyfit([np.nanmedian(Lx_11),np.nanmedian(Lx_12),np.nanmedian(Lx_13),np.nanmedian(Lx_14),np.nanmedian(Lx_15),],
		[np.nanmedian(f4_11),np.nanmedian(f4_12),np.nanmedian(f4_13),np.nanmedian(f4_14),np.nanmedian(f4_15),],1)
		fit_p12 = np.poly1d(fit_z12)

		# ax1.quiver(Lx_11[up_check11 == 1],f3_11[up_check11 == 1],np.zeros(np.shape(f3_11[up_check11 == 1]))*-1, np.ones(np.shape(f3_11[up_check11 == 1]))*-1,facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(Lx_12[up_check12 == 1],f3_12[up_check12 == 1],np.zeros(np.shape(f3_12[up_check12 == 1]))*-1, np.ones(np.shape(f3_12[up_check12 == 1]))*-1,facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(Lx_13[up_check13 == 1],f3_13[up_check13 == 1],np.zeros(np.shape(f3_13[up_check13 == 1]))*-1, np.ones(np.shape(f3_13[up_check13 == 1]))*-1,facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(Lx_14[up_check14 == 1],f3_14[up_check14 == 1],np.zeros(np.shape(f3_14[up_check14 == 1]))*-1, np.ones(np.shape(f3_14[up_check14 == 1]))*-1,facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax1.quiver(Lx_15[up_check15 == 1],f3_15[up_check15 == 1],np.zeros(np.shape(f3_15[up_check15 == 1]))*-1, np.ones(np.shape(f3_15[up_check15 == 1]))*-1,facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)

		ax1.scatter(Lx_11[~np.isnan(Nh_11)],f4_11[~np.isnan(Nh_11)],color=c1,edgecolors='k',s=Nh_11[~np.isnan(Nh_11)],rasterized=True,alpha=0.65)
		ax1.scatter(Lx_12[~np.isnan(Nh_12)],f4_12[~np.isnan(Nh_12)],color=c2,edgecolors='k',s=Nh_12[~np.isnan(Nh_12)],rasterized=True,alpha=0.65)
		ax1.scatter(Lx_13[~np.isnan(Nh_13)],f4_13[~np.isnan(Nh_13)],color=c3,edgecolors='k',s=Nh_13[~np.isnan(Nh_13)],rasterized=True,alpha=0.65)
		ax1.scatter(Lx_14[~np.isnan(Nh_14)],f4_14[~np.isnan(Nh_14)],color=c4,edgecolors='k',s=Nh_14[~np.isnan(Nh_14)],rasterized=True,alpha=0.65)
		ax1.scatter(Lx_15[~np.isnan(Nh_15)],f4_15[~np.isnan(Nh_15)],color=c5,edgecolors='k',s=Nh_15[~np.isnan(Nh_15)],rasterized=True,alpha=0.65)

		ax1.scatter(Lx_11[np.isnan(Nh_11)],f4_11[np.isnan(Nh_11)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax1.scatter(Lx_12[np.isnan(Nh_12)],f4_12[np.isnan(Nh_12)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax1.scatter(Lx_13[np.isnan(Nh_13)],f4_13[np.isnan(Nh_13)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax1.scatter(Lx_14[np.isnan(Nh_14)],f4_14[np.isnan(Nh_14)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax1.scatter(Lx_15[np.isnan(Nh_15)],f4_15[np.isnan(Nh_15)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)

		ax1.scatter(np.nanmedian(Lx_11),np.nanmedian(f4_11),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(Lx_12),np.nanmedian(f4_12),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(Lx_13),np.nanmedian(f4_13),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(Lx_14),np.nanmedian(f4_14),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax1.scatter(np.nanmedian(Lx_15),np.nanmedian(f4_15),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)

		# ax1.plot(Lx_13[(Lx_13 > 43) & (f4_13 < 43)],f4_13[(Lx_13 > 43) & (f4_13 < 43)],'.',color='k')
		# ax1.plot(Lx_14[(Lx_14 > 43) & (f4_14 < 43)],f4_14[(Lx_14 > 43) & (f4_14 < 43)],'.',color='k')

		# ax1.plot(xp,fit_p1(xp),color='k')
		# ax1.plot(xp,fit_p12(xp),'--',color='k')


		# ax1.scatter(emis6,emis3,color='gray',edgecolors='k',s=100,label='ULIRGs')
		# ax1.set_xticklabels([])
		# ax1.set_ylim(-3,3)
		# ax1.set_xlim(-3,3)
		ax1.set_ylim(42,47)
		ax1.set_xlim(42,47)
		ax1.set_ylabel(r'log L (10$\mu$m) [erg/s]')
		# ax1.set_xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
		secax1 = ax1.secondary_xaxis('top', functions=(solar, ergs))
		secax1.set_xlabel(r' ')
		ax1.grid()

		ax2 = plt.subplot(132)
		ax2.set_title(str(zlim_2)+' < z < '+str(zlim_3))

		fit_z2 = np.polyfit(Lx[zbins2],f4[zbins2],1)
		fit_p2 = np.poly1d(fit_z2)

		fit_z22 = np.polyfit([np.nanmedian(Lx_21),np.nanmedian(Lx_22),np.nanmedian(Lx_23),np.nanmedian(Lx_24),np.nanmedian(Lx_25),],
		[np.nanmedian(f4_21),np.nanmedian(f4_22),np.nanmedian(f4_23),np.nanmedian(f4_24),np.nanmedian(f4_25),],1)
		fit_p22 = np.poly1d(fit_z22)

		# ax2.quiver(Lx_21[up_check21 == 1],f3_21[up_check21 == 1],np.zeros(np.shape(f3_21[up_check21 == 1]))*-1, np.ones(np.shape(f3_21[up_check21 == 1]))*-1,facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(Lx_22[up_check22 == 1],f3_22[up_check22 == 1],np.zeros(np.shape(f3_22[up_check22 == 1]))*-1, np.ones(np.shape(f3_22[up_check22 == 1]))*-1,facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(Lx_23[up_check23 == 1],f3_23[up_check23 == 1],np.zeros(np.shape(f3_23[up_check23 == 1]))*-1, np.ones(np.shape(f3_23[up_check23 == 1]))*-1,facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(Lx_24[up_check24 == 1],f3_24[up_check24 == 1],np.zeros(np.shape(f3_24[up_check24 == 1]))*-1, np.ones(np.shape(f3_24[up_check24 == 1]))*-1,facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax2.quiver(Lx_25[up_check25 == 1],f3_25[up_check25 == 1],np.zeros(np.shape(f3_25[up_check25 == 1]))*-1, np.ones(np.shape(f3_25[up_check25 == 1]))*-1,facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)

		ax2.scatter(Lx_21[~np.isnan(Nh_21)],f4_21[~np.isnan(Nh_21)],color=c1,edgecolors='k',s=Nh_21[~np.isnan(Nh_21)],rasterized=True,alpha=0.65)
		ax2.scatter(Lx_22[~np.isnan(Nh_22)],f4_22[~np.isnan(Nh_22)],color=c2,edgecolors='k',s=Nh_22[~np.isnan(Nh_22)],rasterized=True,alpha=0.65)
		ax2.scatter(Lx_23[~np.isnan(Nh_23)],f4_23[~np.isnan(Nh_23)],color=c3,edgecolors='k',s=Nh_23[~np.isnan(Nh_23)],rasterized=True,alpha=0.65)
		ax2.scatter(Lx_24[~np.isnan(Nh_24)],f4_24[~np.isnan(Nh_24)],color=c4,edgecolors='k',s=Nh_24[~np.isnan(Nh_24)],rasterized=True,alpha=0.65)
		ax2.scatter(Lx_25[~np.isnan(Nh_25)],f4_25[~np.isnan(Nh_25)],color=c5,edgecolors='k',s=Nh_25[~np.isnan(Nh_25)],rasterized=True,alpha=0.65)

		ax2.scatter(Lx_21[np.isnan(Nh_21)],f4_21[np.isnan(Nh_21)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax2.scatter(Lx_22[np.isnan(Nh_22)],f4_22[np.isnan(Nh_22)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax2.scatter(Lx_23[np.isnan(Nh_23)],f4_23[np.isnan(Nh_23)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax2.scatter(Lx_24[np.isnan(Nh_24)],f4_24[np.isnan(Nh_24)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)
		ax2.scatter(Lx_25[np.isnan(Nh_25)],f4_25[np.isnan(Nh_25)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.65)

		ax2.scatter(np.nanmedian(Lx_21),np.nanmedian(f4_21),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax2.scatter(np.nanmedian(Lx_22),np.nanmedian(f4_22),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax2.scatter(np.nanmedian(Lx_23),np.nanmedian(f4_23),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax2.scatter(np.nanmedian(Lx_24),np.nanmedian(f4_24),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax2.scatter(np.nanmedian(Lx_25),np.nanmedian(f4_25),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)

		# ax2.plot(Lx_23[(Lx_23 > 44.5) & (f4_23 < 45)],f4_23[(Lx_23 > 44.5) & (f4_23 < 45)],'.',color='k')
		# ax2.plot(Lx_24[(Lx_24 > 44.5) & (f4_24 < 45)],f4_24[(Lx_24 > 44.5) & (f4_24 < 45)],'.',color='k')
		# print(len(self.ID[zbins2]),len(Lx_23),len(Lx_24))
		# print(self.ID[zbins2][B3[zbins2]][(Lx_23 > 44.5) & (f4_23 < 45)])
		# print(self.ID[zbins2][B4[zbins2]][(Lx_24 > 44.5) & (f4_24 < 45)])

		ax2.scatter(-100, -100, color=c1, edgecolors='k',s=75, label='Panel 1', rasterized=True)
		ax2.scatter(-100, -100, color=c2, edgecolors='k',s=75, label='Panel 2', rasterized=True)
		ax2.scatter(-100, -100, color=c3, edgecolors='k',s=75, label='Panel 3', rasterized=True)
		ax2.scatter(-100, -100, color=c4, edgecolors='k',s=75, label='Panel 4', rasterized=True)
		ax2.scatter(-100, -100, color=c5, edgecolors='k',s=75, label='Panel 5', rasterized=True)

		# ax2.plot(xp,fit_p2(xp),color='k')
		# ax2.plot(xp,fit_p22(xp),'--',color='k')

		ax2.set_yticklabels([])
		# ax2.set_xticklabels([])
		# ax2.set_ylim(-3,3)
		# ax2.set_xlim(-3,3)
		ax2.set_ylim(42,47)
		ax2.set_xlim(42,47)
		# ax2.set_ylabel(r'log L (0.25$\mu$m) [erg/s]')
		ax2.set_xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')

		secax2 = ax2.secondary_xaxis('top', functions=(solar, ergs))
		secax2.set_xlabel(r'log L$_{\mathrm{X}}$ [L$_{\odot}$]')
		ax2.grid()
		ax2.legend(fontsize=14)

		ax3 = plt.subplot(133)
		ax3.set_title(str(zlim_3)+' < z < '+str(zlim_4))

		fit_z3 = np.polyfit(Lx[zbins3],f4[zbins3],1)
		fit_p3 = np.poly1d(fit_z3)

		fit_z32 = np.polyfit([np.nanmedian(Lx_31),np.nanmedian(Lx_32),np.nanmedian(Lx_33),np.nanmedian(Lx_34),np.nanmedian(Lx_35),],
		[np.nanmedian(f4_31),np.nanmedian(f4_32),np.nanmedian(f4_33),np.nanmedian(f4_34),np.nanmedian(f4_35),],1)
		fit_p32 = np.poly1d(fit_z32)

		# ax3.quiver(Lx_31[up_check31 == 1],f3_31[up_check31 == 1],np.zeros(np.shape(f3_31[up_check31 == 1]))*-1, np.ones(np.shape(f3_31[up_check31 == 1]))*-1,facecolor=c1,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(Lx_32[up_check32 == 1],f3_32[up_check32 == 1],np.zeros(np.shape(f3_32[up_check32 == 1]))*-1, np.ones(np.shape(f3_32[up_check32 == 1]))*-1,facecolor=c2,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(Lx_33[up_check33 == 1],f3_33[up_check33 == 1],np.zeros(np.shape(f3_33[up_check33 == 1]))*-1, np.ones(np.shape(f3_33[up_check33 == 1]))*-1,facecolor=c3,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(Lx_34[up_check34 == 1],f3_34[up_check34 == 1],np.zeros(np.shape(f3_34[up_check34 == 1]))*-1, np.ones(np.shape(f3_34[up_check34 == 1]))*-1,facecolor=c4,edgecolor='k',linewidth=0.5,rasterized=True)
		# ax3.quiver(Lx_35[up_check35 == 1],f3_35[up_check35 == 1],np.zeros(np.shape(f3_35[up_check35 == 1]))*-1, np.ones(np.shape(f3_35[up_check35 == 1]))*-1,facecolor=c5,edgecolor='k',linewidth=0.5,rasterized=True)

		ax3.scatter(Lx_31[~np.isnan(Nh_31)],f4_31[~np.isnan(Nh_31)],color=c1,edgecolors='k',s=Nh_31[~np.isnan(Nh_31)],rasterized=True,alpha=0.65)
		ax3.scatter(Lx_32[~np.isnan(Nh_32)],f4_32[~np.isnan(Nh_32)],color=c2,edgecolors='k',s=Nh_32[~np.isnan(Nh_32)],rasterized=True,alpha=0.65)
		ax3.scatter(Lx_33[~np.isnan(Nh_33)],f4_33[~np.isnan(Nh_33)],color=c3,edgecolors='k',s=Nh_33[~np.isnan(Nh_33)],rasterized=True,alpha=0.65)
		ax3.scatter(Lx_34[~np.isnan(Nh_34)],f4_34[~np.isnan(Nh_34)],color=c4,edgecolors='k',s=Nh_34[~np.isnan(Nh_34)],rasterized=True,alpha=0.65)
		ax3.scatter(Lx_35[~np.isnan(Nh_35)],f4_35[~np.isnan(Nh_35)],color=c5,edgecolors='k',s=Nh_35[~np.isnan(Nh_35)],rasterized=True,alpha=0.65)

		ax3.scatter(Lx_31[np.isnan(Nh_31)],f4_31[np.isnan(Nh_31)],color=c1,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.75)
		ax3.scatter(Lx_32[np.isnan(Nh_32)],f4_32[np.isnan(Nh_32)],color=c2,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.75)
		ax3.scatter(Lx_33[np.isnan(Nh_33)],f4_33[np.isnan(Nh_33)],color=c3,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.75)
		ax3.scatter(Lx_34[np.isnan(Nh_34)],f4_34[np.isnan(Nh_34)],color=c4,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.75)
		ax3.scatter(Lx_35[np.isnan(Nh_35)],f4_35[np.isnan(Nh_35)],color=c5,edgecolors='k',marker='s',s=75,rasterized=True,alpha=0.75)

		ax3.scatter(np.nanmedian(Lx_31),np.nanmedian(f4_31),color=c1,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(Lx_32),np.nanmedian(f4_32),color=c2,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(Lx_33),np.nanmedian(f4_33),color=c3,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(Lx_34),np.nanmedian(f4_34),color=c4,edgecolors='k',marker='P',linewidth=2,s=250)
		ax3.scatter(np.nanmedian(Lx_35),np.nanmedian(f4_35),color=c5,edgecolors='k',marker='P',linewidth=2,s=250)

		# ax3.plot(xp,fit_p3(xp),color='k')
		# ax3.plot(xp,fit_p32(xp),'--',color='k')


		ax3.scatter(-100,100,marker='P',color='k',s=250,label='Median',rasterized=True)
		ax3.scatter(-100,-100,marker='s',color='k',s=75,label=r'No N$_{\mathrm{H}}$ limits',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=40,label=r'log N$_{\mathrm{H}}$ < 22',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=100,label=r'22 < log N$_{\mathrm{H}}$ < 23',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=180,label=r'23 < log N$_{\mathrm{H}}$ < 24',rasterized=True)
		ax1.scatter(-100,-100,color='k',s=275,label=r'24 < log N$_{\mathrm{H}}$',rasterized=True)

		ax1.legend(fontsize=14)

		ax3.legend(fontsize=14)


		# ax3.scatter(emis6,emis32,color='gray',edgecolors='k',s=100,label='ULIRGs')
		ax3.set_yticklabels([])
		# ax3.set_xticklabels([])
		# ax3.set_ylim(-3,3)
		# ax3.set_xlim(-3,3)
		ax3.set_ylim(42,47)
		ax3.set_xlim(42,47)
		# ax3.set_ylabel(r'log L (0.25$\mu$m) [erg/s]')
		# ax3.set_xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
		
		secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
		secax3.set_xlabel(r' ')
		secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
		secax3.set_ylabel(r'$\lambda$ L$_\lambda$ (10$\mu$m) [L$_{\odot}$]')
		
		ax3.grid()

		plt.tight_layout()

		plt.savefig('/Users/connor_auge/Desktop/New_runSED/'+savestring+'.pdf')
		plt.show()


	def SED_MIR_bins(self,savestring,Lx,MIR,norm,x,y,L,median_wavelength,median_flux):
		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.5
		clim2 = 46

		bin_size1 = np.arange(-1,3,0.25)
		
		Lx_flt = np.asarray([10**i for i in Lx])
		ratio = np.log10(MIR/(Lx_flt/norm))

		B1 = ratio > 1.5
		B2 = np.logical_and(1 < ratio, ratio < 1.5)
		B3 = np.logical_and(0.5 < ratio, ratio < 1)
		B4 = ratio < 0.5

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]


		fig = plt.figure(figsize=(15,15),constrained_layout=False)
		gs1 = fig.add_gridspec(nrows=4, ncols=2, left=0.1,right=0.55,wspace=-0.25,hspace=0.1,width_ratios=[3,0.25])
		gs2 = fig.add_gridspec(nrows=4, ncols=1, left=0.7,right=0.95,wspace=0.1,hspace=0.1)


		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax2 = fig.add_subplot(gs1[1,0])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]

		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')

		ax3 = fig.add_subplot(gs1[2,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]

		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')


		ax4 = fig.add_subplot(gs1[3,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]

		lc4 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5)
		ax4.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		axcb4 = fig.colorbar(lc4)
		axcb4.mappable.set_clim(clim1,clim2)
		axcb4.remove()

		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlim(8E-5,7E2)
		ax4.set_ylim(5E-3,50)
		ax4.set_xticklabels([])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax4.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		
		cbar_ax = fig.add_subplot(gs1[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')



		yticks = [0,10,20,30]

		ax6 = fig.add_subplot(gs2[0,0])
		ax6.hist(ratio[B1],bins=bin_size1,color='red',alpha=0.65,lw=3)
		ax6.axvline(np.nanmedian(ratio[B1]),color='red',ls='--',lw=3)
		ax6.set_ylim(0,50)
		ax6.set_xlim(-3,3)
		# ax6.set_yticks(yticks)
		ax6.set_xticklabels([])
		ax6.grid()

		ax7 = fig.add_subplot(gs2[1,0])
		ax7.hist(ratio[B2],bins=bin_size1,color='red',alpha=0.65,lw=3)
		ax7.axvline(np.nanmedian(ratio[B2]),color='red',ls='--',lw=3)
		ax7.set_ylim(0,50)
		ax7.set_xlim(-3,3)
		# ax7.set_yticks(yticks)
		ax7.set_xticklabels([])
		ax7.grid()

		ax8 = fig.add_subplot(gs2[2,0])
		ax8.hist(ratio[B3],bins=bin_size1,color='red',alpha=0.65,lw=3)
		ax8.axvline(np.nanmedian(ratio[B3]),color='red',ls='--',lw=3)
		ax8.set_ylim(0,50)
		ax8.set_xlim(-3,3)
		# ax8.set_yticks(yticks)
		ax8.set_xticklabels([])
		ax8.grid()

		ax9 = fig.add_subplot(gs2[3,0])
		ax9.hist(ratio[B4],bins=bin_size1,color='red',alpha=0.65,lw=3)
		ax9.axvline(np.nanmedian(ratio[B4]),color='red',ls='--',lw=3)
		ax9.set_ylim(0,50)
		ax9.set_xlim(-3,3)
		# ax9.set_yticks(yticks)
		# ax9.set_xticklabels([])
		ax9.set_xlabel(r'log $\lambda$L$_\mathrm{10\mu m}$/$\lambda$L$_{0.5-10\mathrm{kev}}$')
		ax9.grid()

		plt.savefig('/Users/connor_auge/Desktop/'+savestring+'.png')
		plt.show()


	def Five_col_morph_hist(self,file,param,BT,BTn,DT,DTn,NTn,Sr,Srn,SrnT,f1,f2,f3,f4,ID):

		ID = np.asarray(ID)
		ylimit = [0,30]

		if file == 'bd':
			if param == 'BT':
				X = BT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_BT.png'
				xlabel = 'B/T'

			elif param == 'DT':
				X = DT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_DT.png'
				xlabel = 'D/T'

			else:
				print('param not recognized with file')

		elif file == 'bdn':
			if param == 'BTn':
				X = BTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_BT.png'
				xlabel = 'B/T'

			elif param == 'DTn':
				X = DTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_DT.png'
				xlabel = 'D/T'

			elif param == 'NTn':
				X = NTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_NT.png'
				xlabel = 'N/T'
				ylimit = [0,75]

			else:
				print('param not recognized with file')

		elif file == 'sersic':
			if param == 'sersic':
				X = Sr
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sersic.png'
				xlabel = 'Sersic Index'

			else:
				print('param not recognized with file')

		elif file == 'sernuc':
			if param == 'sersic_n':
				X = Srn
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_sersic.png'
				xlabel = 'Sersic Index'

			elif param == 'sersic_NT':
				X = SrnT
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_NT.png'
				xlabel = 'N/T'

			else:
				print('param not recognized with file')

		X = np.asarray(X)
		# print(X)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3
		
		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]


		fig = plt.figure(figsize=(6,18),constrained_layout=False)
		gs = fig.add_gridspec(nrows=5, ncols=1, left=0.1,right=0.9,hspace=0.15)

		ax1 = fig.add_subplot(gs[0])
		ax1.hist(X[B1],bins=xbins,color='gray')
		ax1.axvline(np.nanmedian(X[B1]),color='k',lw=3)
		ax1.set_xlim(xlimit)
		ax1.set_ylim(ylimit)
		ax1.set_xticklabels([])
		# ax1.text(0.05,0.7,f'n = {len(X[B1])}',transform=ax1.transAxes)
		ax1.grid()

		ax2 = fig.add_subplot(gs[1])
		ax2.hist(X[B2],bins=xbins,color='gray')
		ax2.axvline(np.nanmedian(X[B2]),color='k',lw=3)
		ax2.set_xlim(xlimit)
		ax2.set_ylim(ylimit)
		ax2.set_xticklabels([])
		# ax2.text(0.05,0.7,f'n = {len(X[B2])}',transform=ax2.transAxes)
		ax2.grid()

		ax3 = fig.add_subplot(gs[2])
		ax3.hist(X[B3],bins=xbins,color='gray')
		ax3.axvline(np.nanmedian(X[B3]),color='k',lw=3)
		ax3.set_xlim(xlimit)
		ax3.set_ylim(ylimit)
		ax3.set_xticklabels([])
		# ax3.text(0.05,0.7,f'n = {len(X[B3])}',transform=ax3.transAxes)
		ax3.grid()

		ax4 = fig.add_subplot(gs[3])
		ax4.hist(X[B4],bins=xbins,color='gray')
		ax4.axvline(np.nanmedian(X[B4]),color='k',lw=3)
		ax4.set_xlim(xlimit)
		ax4.set_ylim(ylimit)
		ax4.set_xticklabels([])
		# ax4.text(0.05,0.7,f'n = {len(X[B4])}',transform=ax4.transAxes)
		ax4.grid()

		ax5 = fig.add_subplot(gs[4])
		ax5.hist(X[B5],bins=xbins,color='gray')
		ax5.axvline(np.nanmedian(X[B5]),color='k',lw=3)
		ax5.set_xlim(xlimit)
		ax5.set_ylim(ylimit)
		ax5.set_xlabel(xlabel)
		# ax5.text(0.05,0.7,f'n = {len(X[B5])}',transform=ax5.transAxes)
		ax5.grid()

		plt.savefig('/Users/connor_auge/Desktop/06z08_'+savename)
		plt.show()


	def One_morph_hist(self,file,param,BT,BTn,DT,DTn,NTn,Sr,Srn,SrnT,ID):

		ID = np.asarray(ID)
		ylimit = [0,30]

		if file == 'bd':
			if param == 'BT':
				X = BT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_BT.png'
				xlabel = 'B/T'

			elif param == 'DT':
				X = DT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_DT.png'
				xlabel = 'D/T'

			else:
				print('param not recognized with file')

		elif file == 'bdn':
			if param == 'BTn':
				X = BTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_BT.png'
				xlabel = 'B/T'

			elif param == 'DTn':
				X = DTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_DT.png'
				xlabel = 'D/T'

			elif param == 'NTn':
				X = NTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_NT.png'
				xlabel = 'N/T'
				ylimit = [0,75]

			else:
				print('param not recognized with file')

		elif file == 'sersic':
			if param == 'sersic':
				X = Sr
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sersic.png'
				xlabel = 'Sersic Index'

			else:
				print('param not recognized with file')

		elif file == 'sernuc':
			if param == 'sersic_n':
				X = Srn
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_sersic.png'
				xlabel = 'Sersic Index'

			elif param == 'sersic_NT':
				X = SrnT
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_NT.png'
				xlabel = 'N/T'

			else:
				print('param not recognized with file')

		X = np.asarray(X)
		# print(X)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		fig = plt.figure(figsize=(8,6),constrained_layout=False)

		ax1 = fig.add_subplot(111)
		ax1.hist(X,bins=xbins,color='gray')
		ax1.axvline(np.nanmedian(X),color='k',lw=3)
		ax1.set_xlim(xlimit)
		ax1.set_ylim(ylimit)
		ax1.set_xlabel(xlabel)
		ax1.grid()

		plt.savefig('/Users/connor_auge/Desktop/morphology_plots/one_panel_'+savename)
		plt.show()


	def Five_col_morph_hist_check24(self,file,param,BT,BTn,DT,DTn,NTn,Sr,Srn,SrnT,f1,f2,f3,f4,ID,F1=None):

		ID = np.asarray(ID)
		ylimit = [0,30]

		if file == 'bd':
			if param == 'BT':
				X = BT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_BT_24separation.png'
				xlabel = 'B/T'

			elif param == 'DT':
				X = DT
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'db_DT_24separation.png'
				xlabel = 'D/T'

			else:
				print('param not recognized with file')

		elif file == 'bdn':
			if param == 'BTn':
				X = BTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_BT_24separation.png'
				xlabel = 'B/T'

			elif param == 'DTn':
				X = DTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_DT_24separation.png'
				xlabel = 'D/T'

			elif param == 'NTn':
				X = NTn
				xbins = np.arange(0.0,1.0,0.1)
				xlimit = [-0.15,1.15]
				savename = 'dbn_NT_24separation.png'
				xlabel = 'N/T'
				ylimit = [0,75]

			else:
				print('param not recognized with file')

		elif file == 'sersic':
			if param == 'sersic':
				X = Sr
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sersic_24separation.png'
				xlabel = 'Sersic Index'

			else:
				print('param not recognized with file')

		elif file == 'sernuc':
			if param == 'sersic_n':
				X = Srn
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_sersic_24separation.png'
				xlabel = 'Sersic Index'

			elif param == 'sersic_NT':
				X = SrnT
				xbins = np.arange(0.0,9.0,0.5)
				xlimit = [-0.15,9.15]
				savename = 'sernuc_NT_24separation.png'
				xlabel = 'N/T'

			else:
				print('param not recognized with file')

		X = np.asarray(X)
		# print(X)
		check_24 = np.asarray(F1)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3
		
		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		# B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'no detection')))[0]


		B1 = np.where(np.logical_and(f1 > 0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		B5 = np.where(check_24 == 'no detection')[0]

		fig = plt.figure(figsize=(6,18),constrained_layout=False)
		gs = fig.add_gridspec(nrows=5, ncols=1, left=0.1,right=0.9,hspace=0.15)

		ax1 = fig.add_subplot(gs[0])
		ax1.hist(X[B1],bins=xbins,color='gray')
		ax1.axvline(np.nanmedian(X[B1]),color='k',lw=3)
		ax1.set_xlim(xlimit)
		ax1.set_ylim(ylimit)
		ax1.set_xticklabels([])
		# ax1.text(0.05,0.7,f'n = {len(X[B1])}',transform=ax1.transAxes)
		ax1.grid()

		ax2 = fig.add_subplot(gs[1])
		ax2.hist(X[B2],bins=xbins,color='gray')
		ax2.axvline(np.nanmedian(X[B2]),color='k',lw=3)
		ax2.set_xlim(xlimit)
		ax2.set_ylim(ylimit)
		ax2.set_xticklabels([])
		# ax2.text(0.05,0.7,f'n = {len(X[B2])}',transform=ax2.transAxes)
		ax2.grid()

		ax3 = fig.add_subplot(gs[2])
		ax3.hist(X[B3],bins=xbins,color='gray')
		ax3.axvline(np.nanmedian(X[B3]),color='k',lw=3)
		ax3.set_xlim(xlimit)
		ax3.set_ylim(ylimit)
		ax3.set_xticklabels([])
		# ax3.text(0.05,0.7,f'n = {len(X[B3])}',transform=ax3.transAxes)
		ax3.grid()

		ax4 = fig.add_subplot(gs[3])
		ax4.hist(X[B4],bins=xbins,color='gray')
		ax4.axvline(np.nanmedian(X[B4]),color='k',lw=3)
		ax4.set_xlim(xlimit)
		ax4.set_ylim(ylimit)
		ax4.set_xticklabels([])
		# ax4.text(0.05,0.7,f'n = {len(X[B4])}',transform=ax4.transAxes)
		ax4.grid()

		ax5 = fig.add_subplot(gs[4])
		ax5.hist(X[B5],bins=xbins,color='gray')
		ax5.axvline(np.nanmedian(X[B5]),color='k',lw=3)
		ax5.set_xlim(xlimit)
		ax5.set_ylim(ylimit)
		ax5.set_xlabel(xlabel)
		# ax5.text(0.05,0.7,f'n = {len(X[B5])}',transform=ax5.transAxes)
		ax5.grid()

		plt.savefig('/Users/connor_auge/Desktop/09z11_All_'+savename)
		plt.show()


	def stern_Lx_2_10(self,mir):
		L6 = np.asarray([10**i for i in mir])
		L6 = np.log10(L6/1E41)

		Lx = 40.981 + 1.024*L6 - 0.047*L6**2

		return Lx


	def amus_Lx_2_10_MIR12(self,mir):
		L12 = np.asarray([10**i for i in mir])
		L12 = np.log10(L12/1E43)

		Lx = (10**(-0.32 + 0.95*L12))*1E43 


		# Lx = np.asarray([(10**i)*1E43 for i in Lx_210])
		
		return np.log10(Lx)


	def ichikawa_MIR_12(self,Lx):
		Lx_14_195 = np.asarray([10**i for i in Lx])
		Lx_210 = Lx_14_195/2.1
		Lx_210 = np.log10(Lx_210/1E43)

		# Lx_2_10 = np.asarray([10**i for i in Lx])
		# Lx_14_195 = 2.1*Lx_2_10
		# Lx_14_195 = np.log10(Lx_14_195/1E43)

		L12 = (10**(-0.21 +1.056*Lx_210))*1E43

		return np.log10(L12)


	def Durras_Lbol(self,Lx):
		a = 15.33
		b = 11.48
		c = 16.20

		L_x = np.asarray([10**i for i in Lx])

		L_bol = a*(1+(np.log10(L_x)/b)**c)*L_x

		return np.log10(L_bol)
