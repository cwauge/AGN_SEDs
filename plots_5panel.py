import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from astropy.cosmology import FlatLambdaCDM
from filters import Filters
from SED_v7 import Flux_to_Lum



class Plotter_5panel():
	'''A class to plot the properties of each source from the AGN class'''

	def __init__(self,ID,z,wavelength,flux,Lx=None,Lbol=None,spec_type=None):
		self.ID = np.asarray(ID)
		self.z = np.asarray(z)
		self.wavelength_array = np.asarray(wavelength)
		self.flux_array = np.asarray(flux)
		self.Lx = np.asarray(Lx)
		self.Lbol = np.asarray(Lbol)
		self.spec_type = np.asarray(spec_type)

	def tick_function(self,X):
		X = 10**X
		V = X/3.8E33
		solar = np.log10(V)
		return '%.2f' % solar

	def multilines(self,xs,ys,cs,ax=None,**kwargs):
		ax = plt.gca() if ax is None else ax # find axes
		segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
		lc = LineCollection(segments, **kwargs)
		lc.set_array(np.asarray(cs)) # set coloring of line segments
		ax.add_collection(lc) # add lines to axes and rescale
		ax.autoscale()
		return lc

	
	def PlotSingleSED(self,flux_point=None):	
		plt.rcParams['font.size']=12
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		fig, ax = plt.subplots(figsize=(8,6))
		ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
		ax.set_ylabel(r'$\nu$ F$_\nu$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(5E-5,7E2)
		ax.set_ylim(1E-2,3E2)
		ax.set_title(str(self.ID))

		plt.grid()

		self.SED_line = ax.plot(self.wavelength_array,self.flux_array)
		self.SED_points = ax.plot(self.wavelength_array,self.flux_array,color='k',marker='x')
		ax.plot([0.25,5,100],flux_point,'x',color='red')

		plt.show()


	def plot_multiSED_bins(self,savestring,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,suptitle=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None):

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

		clim1 = 42.5
		clim2 = 46

		cosmos_s82x_list = []
		cosmos_s82x_wave = []
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

			cosmos_nuLnu_upper = Flux_to_Lum(cosmos_nuFnu_upper,1.0)
			s82X_nuLnu_upper = Flux_to_Lum(s82X_nuFnu_upper,1.0)
			goodsN_nuLnu_upper = Flux_to_Lum(goodsN_nuFnu_upper,1.0)
			goodsS_nuLnu_upper = Flux_to_Lum(goodsS_nuFnu_upper,1.0)

			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]


			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					# cosmos_s82x_list.append(cosmos_nuLnu_upper/norm[i])
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					# cosmos_s82x_list.append(s82X_nuLnu_upper/norm[i])
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
				else:
					a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					# cosmos_s82x_list.append(goodsN_nuLnu_upper/norm[i])
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-5:])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					# cosmos_s82x_list.append(goodsS_nuLnu_upper/norm[i])
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
			cosmos_s82x_wave.append(rest_upper_w_microns)

		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
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
		# ytick_labels = [1E-1,1E0,1E1]

		
	
		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(9,15),nrows=5,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		# fig.suptitle(suptitle,fontsize=22)
		# fig.suptitle('0.6 < z < 0.8 & 0.9 < z < 1.1',fontsize=20)

		# ax1 = plt.subplot(511)
		ax1 = plt.subplot(gs[0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		# spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		
		# points1 = plt.plot(x1,y1,'.',color='k')
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
		ax1.text(0.75,0.08,str((len(x1)/len(x))*100)[0:4]+'%',transform=ax1.transAxes,weight='bold')
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax1.text(0.0,1.03,r'A',transform=ax1.transAxes,fontsize=27,weight='bold')
		ax1.set_title('0.3 < z < 0.5')
		# ax2 = plt.subplot(512)
		ax2 = plt.subplot(gs[1])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		# spec_type2 = spec_type[B2]

		# ax2.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm2),cosmos_nuFnu_upper/min(cosmos_norm2)],color='b',lw=3)
		# ax2.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm2),s82X_nuFnu_upper/min(s82X_norm2)],color='g',lw=3)
		# points2 = plt.plot(x2,y2,'.',color='k')
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
		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax3 = plt.subplot(513)
		ax3 = plt.subplot(gs[2])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		# spec_type3 = spec_type[B3]

		# ax3.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm3),cosmos_nuFnu_upper/min(cosmos_norm3)],color='b',lw=3)
		# ax3.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm3),s82X_nuFnu_upper/min(s82X_norm3)],color='g',lw=3)
		# points3 = plt.plot(x3,y3,'.',color='k')
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
		ax3.text(0.75,0.08,str((len(x3)/len(x))*100)[0:4]+'%',transform=ax3.transAxes,weight='bold')
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax4 = plt.subplot(514)
		ax4 = plt.subplot(gs[3])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		# spec_type4 = spec_type[B4]

		# ax4.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm4),cosmos_nuFnu_upper/min(cosmos_norm4)],color='b',lw=3)
		# ax4.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm4),s82X_nuFnu_upper/min(s82X_norm4)],color='g',lw=3)
		# points4 = plt.plot(x4,y4,'.',color='k')
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
		ax4.text(0.75,0.08,str((len(x4)/len(x))*100)[0:4]+'%',transform=ax4.transAxes,weight='bold')
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax5 = plt.subplot(515)
		ax5 = plt.subplot(gs[4])
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		# spec_type5 = spec_type[B5]
		# mask5 = np.ma.masked_invalid(y5).mask
		# print('SHAPE:',np.shape(y5))

		# y6 = np.zeros((47,40))
		# x6 = np.zeros((47,40))
		# for i in range(len(y5)):
			# y6[i] = np.delete(y5[i],[37,38])
			# x6[i] = np.delete(x5[i],[37,38])
		x6 = x5
		y6 = y5

		# ax5.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm5),cosmos_nuFnu_upper/min(cosmos_norm5)],color='b',lw=3)
		# ax5.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm5),s82X_nuFnu_upper/min(s82X_norm5)],color='g',lw=3)
		# points5 = plt.plot(x6,y6,'.',color='k')
		lc5 = self.multilines(x6,y6,L5,cmap='rainbow',lw=1.5,rasterized=True)
		ax5.plot(x6,y6,marker='x',color='gray',alpha=0.75)
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
		ax5.text(0.75,0.08,str((len(x5)/len(x))*100)[0:4]+'%',transform=ax5.transAxes,weight='bold')
		ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		print('bin 1: ',np.nanmedian(L1)-np.log10(3.8E33))
		print('bin 2: ',np.nanmedian(L2)-np.log10(3.8E33))
		print('bin 3: ',np.nanmedian(L3)-np.log10(3.8E33))
		print('bin 4: ',np.nanmedian(L4)-np.log10(3.8E33))
		print('bin 5: ',np.nanmedian(L5)-np.log10(3.8E33))


		# fig.subplots_adjust(right=0.7)
		
		
		# fig.tight_layout(rect=[0, 0.03, 0.5, 1.0])
		# cbar_ax = fig.add_axes([0.78, 0.1, 0.025, 0.8])
		cb = fig.colorbar(test,ax=axes)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')


		plt.savefig('/Users/connor_auge/Desktop/Paper/5panel_'+savestring+'NEW.png')
		plt.show()


	def plot_multiSED_bins_24check(self,savestring,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None,F3=None,suptitle=None):

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

		clim1 = 42.5
		clim2 = 46

		w250 = 2536859.83
		rest_w250 = w250/(1+1.0)
		rest_w250_cgs = rest_w250*1E-8
		rest_w250_microns = rest_w250*1E-4
		rest_w250_freq = 3E10/rest_w250_cgs

		cosmos_upper_lim_jy = 8.1E3*1E-6
		s82X_upper_lim_jy = 13.0E3*1E-6

		cosmos_upper_lim_cgs = cosmos_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		s82X_upper_lim_cgs = s82X_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		
		cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_w250_freq
		s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_w250_freq

		mean_upper = np.median([cosmos_nuFnu_upper,s82X_nuFnu_upper])

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)
		check_24 = np.asarray(F3)

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'no detection')))[0]


		# B1 = np.where(np.logical_and(f1 > 0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		# B5 = np.where(check_24 == 'no detection')[0]

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
		# ytick_labels = [1E-1,1E0,1E1]

	
		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(9,15),nrows=5,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle(suptitle,fontsize=22)

		# ax1 = plt.subplot(511)
		ax1 = plt.subplot(gs[0])
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
		# ax1.set_yticklabels(ytick_labels)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		# ax1.text(0.05,0.85,f'type 1: {len(spec_type1[spec_type1 == 1])}',transform=ax1.transAxes,fontsize=16)
		# ax1.text(0.05,0.7,f'type 2: {len(spec_type1[spec_type1 == 2])}',transform=ax1.transAxes,fontsize=16)
		# ax1.set_title(f'{bins[0]} < log {label} < {bins[1]}')
		ax1.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax2 = plt.subplot(512)
		ax2 = plt.subplot(gs[1])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]
		spec_type2 = spec_type[B2]

		# ax2.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm2),cosmos_nuFnu_upper/min(cosmos_norm2)],color='b',lw=3)
		# ax2.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm2),s82X_nuFnu_upper/min(s82X_norm2)],color='g',lw=3)
		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		ax2.plot(np.nanmedian(10**median_wavelength[B2],axis=0),np.nanmedian(10**median_flux[B2],axis=0),color='k',lw=3.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1,clim2)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		# ax2.set_yticklabels([])
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		# ax2.set_yticklabels(ytick_labels)
		ax2.text(0.05,0.7,f'n = {len(x2)}',transform=ax2.transAxes)
		# ax2.text(0.05,0.85,f'type 1: {len(spec_type2[spec_type2 == 1])}',transform=ax2.transAxes,fontsize=16)
		# ax2.text(0.05,0.7,f'type 2: {len(spec_type2[spec_type2 == 2])}',transform=ax2.transAxes,fontsize=16)
		# ax2.set_title(f'{bins[1]} < log  {label} < {bins[2]}')
		ax2.set_ylabel(r'$\lambda$ L$_\lambda$')
		# ax2.set_xlabel(r'Rest Wavelength [$\mu$ m]')

		# ax3 = plt.subplot(513)
		ax3 = plt.subplot(gs[2])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		spec_type3 = spec_type[B3]

		# ax3.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm3),cosmos_nuFnu_upper/min(cosmos_norm3)],color='b',lw=3)
		# ax3.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm3),s82X_nuFnu_upper/min(s82X_norm3)],color='g',lw=3)
		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		ax3.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		# ax3.set_yticklabels([])
		ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_yticklabels(ytick_labels)
		ax3.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax3.text(0.05,0.85,f'type 1: {len(spec_type3[spec_type3 == 1])}',transform=ax3.transAxes,fontsize=16)
		# ax3.text(0.05,0.7,f'type 2: {len(spec_type3[spec_type3 == 2])}',transform=ax3.transAxes,fontsize=16)
		# ax3.set_title(f'{bins[2]} < log  {label} < {bins[3]}')
		ax3.set_ylabel(r'$\lambda$ L$_\lambda$')


		# ax4 = plt.subplot(514)
		ax4 = plt.subplot(gs[3])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		spec_type4 = spec_type[B4]

		# ax4.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm4),cosmos_nuFnu_upper/min(cosmos_norm4)],color='b',lw=3)
		# ax4.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm4),s82X_nuFnu_upper/min(s82X_norm4)],color='g',lw=3)
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
		# ax4.set_yticklabels(ytick_labels)
		# ax4.set_yticklabels([])
		ax4.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		# ax4.text(0.05,0.85,f'type 1: {len(spec_type4[spec_type4 == 1])}',transform=ax4.transAxes,fontsize=16)
		# ax4.text(0.05,0.7,f'type 2: {len(spec_type4[spec_type4 == 2])}',transform=ax4.transAxes,fontsize=16)
		# ax4.set_title(f'{bins[0]} < log {label} < {bins[1]}')
		ax4.set_ylabel(r'$\lambda$ L$_\lambda$')

		# ax5 = plt.subplot(515)
		ax5 = plt.subplot(gs[4])
		x5 = x[B5]
		y5 = y[B5]
		L5 = L[B5]
		spec_type5 = spec_type[B5]
		# mask5 = np.ma.masked_invalid(y5).mask
		# print('SHAPE:',np.shape(y5))

		# y6 = np.zeros((47,40))
		# x6 = np.zeros((47,40))
		# for i in range(len(y5)):
			# y6[i] = np.delete(y5[i],[37,38])
			# x6[i] = np.delete(x5[i],[37,38])
		x6 = x5
		y6 = y5

		# ax5.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm5),cosmos_nuFnu_upper/min(cosmos_norm5)],color='b',lw=3)
		# ax5.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm5),s82X_nuFnu_upper/min(s82X_norm5)],color='g',lw=3)
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
		# ax5.set_yticklabels(ytick_labels)
		# ax5.set_xticklabels([])
		ax5.text(0.05,0.7,f'n = {len(x5)}',transform=ax5.transAxes)
		# ax5.text(0.05,0.85,f'type 1: {len(spec_type5[spec_type5 == 1])}',transform=ax5.transAxes,fontsize=16)
		# ax5.text(0.05,0.7,f'type 2: {len(spec_type5[spec_type5 == 2])}',transform=ax5.transAxes,fontsize=16)
		# ax5.set_title(f'{bins[1]} < log  {label} < {bins[2]}')
		ax5.set_ylabel(r'$\lambda$ L$_\lambda$')
		ax5.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		# fig.subplots_adjust(right=0.7)
		
		
		# fig.tight_layout(rect=[0, 0.03, 0.5, 1.0])
		# cbar_ax = fig.add_axes([0.78, 0.1, 0.025, 0.8])
		cb = fig.colorbar(test,ax=axes)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')


		plt.savefig('/Users/connor_auge/Desktop/'+savestring+'.png')
		plt.show()

	def plot_BTP_bins(self,ID,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):
		sf = [2563,2961,3028,3388,4321,4623,4665,4839,4945,4951,5036,2573,2803,3783,551328,631161,761900,893099,907852]
		comp = [2425,2436,2567,2702,3016,3029,3106,3131,3354,3493,3520,3645,4053,4161,4456,4873,5146,439,2758,2831,3206,135286,3514,3814,4455,4688,4758,5089,5135,429011,484120, 558743, 580295, 700285, 859602, 945519]
		agn = [2360,   2536,   2609,   3015,   3031,   3055,   3292,   3703,   3841,   3846,3881,   4203,   4278,   4334,   4414,   4422,   4437,   4459,   4491,   4560,4592,   4696,   4843,   5028,   5095,   2940,   3135,   3334,   3799,   3976,4546,   4941,   5143, 215929, 317720, 420912, 429760, 492525, 508074, 564776,649852, 716430, 725161, 744557, 768111, 867735, 902831, 904278, 910082, 930210,977411]

		sf_ind = np.where(ID == sf)
		comp_ind = np.where(ID == comp)
		agn_ind = np.where(ID == agn)

		print(sf_ind)
		print(comp_ind)
		print(agn_ind)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# xticks = [1E-3,1E-2,1E-1,1,10,100]
		xticks = [1E-4,1E-2,1,100]
		yticks = [1E-2,0.1,1,10]

		# Vertical
		fig, axes = plt.subplots(figsize=(9,15),nrows=3,ncols=1)
		gs = fig.add_gridspec(nrows=3, ncols=2, left=0.2,right=0.85,wspace=-0.25,hspace=0.2,width_ratios=[3,0.15])

		# gs = gridspec.GridSpec(3, 2)
		# gs.update(hspace=0.05) # set the spacing between axes
		# gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		widths = [3,3,3,0.25]
		# heights = [4,4,4,3]
		# print('Lx:',len(L[L > 45]),L[L > 45])

		# Horizontal
		# fig, axes = plt.subplots(figsize=(figheight*plot_aspect_ratio,figheight))
		# fig, axes = plt.subplots(figsize=(15,5),nrows=1,ncols=3)

		# fig = plt.figure(figsize=(15,5),constrained_layout=False)
		# # gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)#, left=0.1,right=0.9,top=0.93,bottom=0.17, wspace=0.0, hspace=0.0)
		# gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig, left=0.1,right=0.89,top=0.93,bottom=0.12, wspace=0.05, hspace=0.0, width_ratios=widths)
		# # gs.update(left=0.1,right=0.9,top=0.93,bottom=0.17)
		# # gs.update(wspace=0.05)


		# fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = fig.add_subplot(gs[0,0])
		x1 = x[sf_ind]
		y1 = y[sf_ind]
		L1 = L[sf_ind]
		spec_type1 = spec_type[sf_ind]
		# print('Lx 1:',len(L1[L1 > 45]),L1[L1 > 45])

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(42.5,46,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[sf_ind],axis=0),np.nanmedian(10**median_flux[sf_ind],axis=0),color='k',lw=3.5)
		# ax1.plot([125,125],[mean_upper/max(norm1),mean_upper/min(norm1)],color='gray',alpha=0.75,lw=3)
		# ax1.plot([125,125],[(mean_upper/(np.nanpercentile(norm1,95))),(mean_upper/(np.nanpercentile(norm1,5)))],color='k',alpha=0.75,lw=3)
		# ax1.plot(125,mean_upper/np.median(norm1),'v',color='r',ms=7)
		axcb1 = fig.colorbar(lc1,orientation='horizontal',pad=-0.1)
		axcb1.set_clim(42,46)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		# ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		# ax1.set_yticklabels(ytick_labels)
		# ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=20)
		# ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax2 = fig.add_subplot(gs[1,0])
		x3 = x[comp_ind]
		y3 = y[comp_ind]
		L3 = L[comp_ind]
		spec_type3 = spec_type[comp_ind]
		# print('Lx 3:',len(L3[L3 > 45]),L3[L3 > 45])

		lc2 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength[comp_ind],axis=0),np.nanmedian(10**median_flux[comp_ind],axis=0),color='k',lw=3.5)
		# ax2.plot([125,125],[mean_upper/max(norm3),mean_upper/min(norm3)],color='gray',alpha=0.75,lw=3)
		# ax2.plot([125,125],[(mean_upper/(np.nanpercentile(norm3,95))),(mean_upper/(np.nanpercentile(norm3,5)))],color='k',alpha=0.75,lw=3)
		# ax2.plot(125,mean_upper/np.median(norm3),'v',color='r',ms=7)
		# ax2.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm3),s82X_nuFnu_upper/np.mean(s82X_norm3)]),'v',color='k',ms=7)
		axcb2 = fig.colorbar(lc2,orientation='horizontal',pad=-0.1)
		axcb2.set_clim(42,46)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_yticklabels([])
		# ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		# ax2.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax2.set_ylabel(r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=25)
		# ax2.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)

		ax3 = fig.add_subplot(gs[2,0])
		x4 = x[agn_ind]
		y4 = y[agn_ind]
		L4 = L[agn_ind]
		spec_type4 = spec_type[agn_ind]
		# print('Lx 4:',len(L4[L4 > 45]),L4[L4 > 45])

		lc3 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength[agn_ind],axis=0),np.nanmedian(10**median_flux[agn_ind],axis=0),color='k',lw=3.5)
		# ax3.plot([125,125],[mean_upper/max(norm4),mean_upper/min(norm4)],color='gray',alpha=0.75,lw=3)
		# ax3.plot([125,125],[(mean_upper/(np.nanpercentile(norm4,95))),(mean_upper/(np.nanpercentile(norm4,5)))],color='k',alpha=0.75,lw=3)
		# ax3.plot(125,mean_upper/np.median(norm4),'v',color='r',ms=7)
		# ax3.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm4),s82X_nuFnu_upper/np.mean(s82X_norm4)]),'v',color='k',ms=7)
		axcb3 = fig.colorbar(lc3,orientation='horizontal',pad=-0.1)
		axcb3.set_clim(42,46)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		# ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_yticklabels(ytick_labels)
		ax3.set_yticklabels([])
		# ax3.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		# ax3.set_ylabel(r'$\lambda$ F$_\lambda$')
		ax3.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax1.grid()
		ax2.grid()
		ax3.grid()
		

		# Vertical
		# cb = fig.colorbar(test,ax=axes)
		# cb.set_label(r'log $L_{\mathrm{X}}$ [erg/s]')

		cbar_ax = fig.add_subplot(gs[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')


		# Horizontal
		# cbar_ax = fig.add_subplot(gs[3])
		# # fig.tight_layout()
		# # fig.subplots_adjust(bottom=0.17)
		# # fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		# cb = fig.colorbar(test,cax=cbar_ax)
		# cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')



		# plt.savefig('/Users/connor_auge/Desktop/SEDshape_3panel_BPT.pdf')
		plt.show()

	def plot_multiSED_3bins(self,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):

		w250 = 2536859.83
		rest_w250 = w250/(1+1.0)
		rest_w250_cgs = rest_w250*1E-8
		rest_w250_microns = rest_w250*1E-4
		rest_w250_freq = 3E10/rest_w250_cgs

		cosmos_upper_lim_jy = 8.1E3*1E-6
		s82X_upper_lim_jy = 13.0E3*1E-6

		cosmos_upper_lim_cgs = cosmos_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		s82X_upper_lim_cgs = s82X_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		
		cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_w250_freq
		s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_w250_freq

		mean_upper = np.median([cosmos_nuFnu_upper,s82X_nuFnu_upper])

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		x[y > 5E2] = np.nan
		y[y > 5E2] = np.nan
		x[y < 1E-4] = np.nan
		y[y < 1E-4] = np.nan

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

		# xticks = [1E-3,1E-2,1E-1,1,10,100]
		xticks = [1E-4,1E-2,1,100]
		yticks = [1E-2,0.1,1,10]
		# ytick_labels = [1E-1,1E0,1E1]



		# Vertical
		fig, axes = plt.subplots(figsize=(9,15),nrows=3,ncols=1)
		gs = fig.add_gridspec(nrows=3, ncols=2, left=0.2,right=0.85,wspace=-0.25,hspace=0.2,width_ratios=[3,0.15])

		# gs = gridspec.GridSpec(3, 2)
		# gs.update(hspace=0.05) # set the spacing between axes
		# gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		widths = [3,3,3,0.25]
		# heights = [4,4,4,3]
		# print('Lx:',len(L[L > 45]),L[L > 45])

		# Horizontal
		# fig, axes = plt.subplots(figsize=(figheight*plot_aspect_ratio,figheight))
		# fig, axes = plt.subplots(figsize=(15,5),nrows=1,ncols=3)

		# fig = plt.figure(figsize=(15,5),constrained_layout=False)
		# # gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)#, left=0.1,right=0.9,top=0.93,bottom=0.17, wspace=0.0, hspace=0.0)
		# gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig, left=0.1,right=0.89,top=0.93,bottom=0.12, wspace=0.05, hspace=0.0, width_ratios=widths)
		# # gs.update(left=0.1,right=0.9,top=0.93,bottom=0.17)
		# # gs.update(wspace=0.05)


		# fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = fig.add_subplot(gs[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		spec_type1 = spec_type[B1]
		# print('Lx 1:',len(L1[L1 > 45]),L1[L1 > 45])

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(42.5,46,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		ax1.plot([125,125],[mean_upper/max(norm1),mean_upper/min(norm1)],color='gray',alpha=0.75,lw=3)
		ax1.plot([125,125],[(mean_upper/(np.nanpercentile(norm1,95))),(mean_upper/(np.nanpercentile(norm1,5)))],color='k',alpha=0.75,lw=3)
		ax1.plot(125,mean_upper/np.median(norm1),'v',color='r',ms=7)
		axcb1 = fig.colorbar(lc1,orientation='horizontal',pad=-0.1)
		axcb1.set_clim(42.5,46)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		# ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		# ax1.set_yticklabels(ytick_labels)
		# ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=20)
		# ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax2 = fig.add_subplot(gs[1,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		spec_type3 = spec_type[B3]
		# print('Lx 3:',len(L3[L3 > 45]),L3[L3 > 45])

		lc2 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		ax2.plot([125,125],[mean_upper/max(norm3),mean_upper/min(norm3)],color='gray',alpha=0.75,lw=3)
		ax2.plot([125,125],[(mean_upper/(np.nanpercentile(norm3,95))),(mean_upper/(np.nanpercentile(norm3,5)))],color='k',alpha=0.75,lw=3)
		ax2.plot(125,mean_upper/np.median(norm3),'v',color='r',ms=7)
		# ax2.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm3),s82X_nuFnu_upper/np.mean(s82X_norm3)]),'v',color='k',ms=7)
		axcb2 = fig.colorbar(lc2,orientation='horizontal',pad=-0.1)
		axcb2.set_clim(42.5,46)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_yticklabels([])
		# ax2.set_xticklabels([])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		# ax2.text(0.05,0.7,f'n = {len(x3)}',transform=ax3.transAxes)
		# ax2.set_ylabel(r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=25)
		# ax2.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)

		ax3 = fig.add_subplot(gs[2,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		spec_type4 = spec_type[B4]
		# print('Lx 4:',len(L4[L4 > 45]),L4[L4 > 45])

		lc3 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		ax3.plot([125,125],[mean_upper/max(norm4),mean_upper/min(norm4)],color='gray',alpha=0.75,lw=3)
		ax3.plot([125,125],[(mean_upper/(np.nanpercentile(norm4,95))),(mean_upper/(np.nanpercentile(norm4,5)))],color='k',alpha=0.75,lw=3)
		ax3.plot(125,mean_upper/np.median(norm4),'v',color='r',ms=7)
		# ax3.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm4),s82X_nuFnu_upper/np.mean(s82X_norm4)]),'v',color='k',ms=7)
		axcb3 = fig.colorbar(lc3,orientation='horizontal',pad=-0.1)
		axcb3.set_clim(42.5,46)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		# ax3.set_xticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		# ax3.set_yticklabels(ytick_labels)
		ax3.set_yticklabels([])
		# ax3.text(0.05,0.7,f'n = {len(x4)}',transform=ax4.transAxes)
		# ax3.set_ylabel(r'$\lambda$ F$_\lambda$')
		ax3.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax1.grid()
		ax2.grid()
		ax3.grid()
		

		# Vertical
		# cb = fig.colorbar(test,ax=axes)
		# cb.set_label(r'log $L_{\mathrm{X}}$ [erg/s]')

		cbar_ax = fig.add_subplot(gs[:,-1:])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ (0.5-10kev) [erg/s]')


		# Horizontal
		# cbar_ax = fig.add_subplot(gs[3])
		# # fig.tight_layout()
		# # fig.subplots_adjust(bottom=0.17)
		# # fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		# cb = fig.colorbar(test,cax=cbar_ax)
		# cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')



		plt.savefig('/Users/connor_auge/Desktop/SEDshape_3panel_vert_05z10.pdf')
		plt.show()

	def plot_multiSED_1bin(self,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):

		w250 = 2536859.83
		rest_w250 = w250/(1+1.0)
		rest_w250_cgs = rest_w250*1E-8
		rest_w250_microns = rest_w250*1E-4
		rest_w250_freq = 3E10/rest_w250_cgs

		cosmos_upper_lim_jy = 8.1E3*1E-6
		s82X_upper_lim_jy = 13.0E3*1E-6

		cosmos_upper_lim_cgs = cosmos_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		s82X_upper_lim_cgs = s82X_upper_lim_jy*1E-23 # 3σ upper limits in cgs
		
		cosmos_nuFnu_upper = cosmos_upper_lim_cgs*rest_w250_freq
		s82X_nuFnu_upper = s82X_upper_lim_cgs*rest_w250_freq

		mean_upper = np.median([cosmos_nuFnu_upper,s82X_nuFnu_upper])

		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		spec_type = np.asarray(spec_type, dtype=float)
		norm = np.asarray(F1)
		mark = np.asarray(F2)

		# B1 = np.where(np.logical_and(np.logical_and(f1 > 0.15, f2 >= -0.15),L>44.2))[0]
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
		# xticks = [1E12,1E13,1E14,1E15,1E16,1E17]
		yticks = [1E-2,0.1,1,10]
		ytick_labels = [1E-1,1E0,1E1]

		print(mean_upper/np.mean(norm1))
		print(mean_upper/(np.mean(norm1)+3*np.std(norm1)))
		print(mean_upper/abs(np.mean(norm1)-3*np.std(norm1)))
		print(mean_upper/np.mean(norm4))
		print(mean_upper/np.mean(norm4)+(mean_upper/(3*np.std(norm4))))
		print(mean_upper/np.mean(norm4)-(mean_upper/(3*np.std(norm4))))

		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(10,8),nrows=1,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		# gs = gridspec.GridSpec(1, 1)
		# gs.update(hspace=0.05) # set the spacing between axes
		# gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = plt.subplot(111)
		# ax1 = plt.subplot(gs[0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		spec_type1 = spec_type[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(42.5,46,10),cmap='rainbow')

		X = np.arange(1E-2,1E1)
		Y = np.arange(1E-2,1E1)**-0.5

		print('slope:',(Y[5]-Y[3])/(X[5]-X[3]))
		print('log slope:',(np.log10(Y[5])-np.log10(Y[3]))/(np.log10(X[5])-np.log10(X[3])))
		
		# ax1.plot(X,Y,ls='--',color='k')

		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		# ax1.plot([150,150],[cosmos_nuFnu_upper/max(cosmos_norm1),cosmos_nuFnu_upper/min(cosmos_norm1)],color='b',lw=3)
		# ax1.plot(125,cosmos_nuFnu_upper/min(cosmos_norm1),'x',color='b')
		# ax1.plot(125,cosmos_nuFnu_upper/max(cosmos_norm1),'x',color='b')
		# ax1.plot([110,110],[s82X_nuFnu_upper/max(s82X_norm1),s82X_nuFnu_upper/min(s82X_norm1)],color='g',lw=3)
		# ax1.plot(125,s82X_nuFnu_upper/min(s82X_norm1),'x',color='g')
		# ax1.plot(125,s82X_nuFnu_upper/max(s82X_norm1),'x',color='g')
		# ax1.plot([125,125],[mean_upper/max(norm1),mean_upper/min(norm1)],color='gray',alpha=0.75,lw=3)
		# ax1.plot([125,125],[(mean_upper/(np.nanpercentile(norm1,95))),(mean_upper/(np.nanpercentile(norm1,5)))],color='k',alpha=0.75,lw=3)
		# ax1.plot(125,mean_upper/np.median(norm1),'v',color='r',ms=7)
		# ax1.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm1),s82X_nuFnu_upper/np.mean(s82X_norm1)]),'v',color='k',ms=7)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(42.5,46)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(1E-1,7E2)
		# ax1.set_xlim(1E12,1E17)
		ax1.set_ylim(5E-3,50)
		# ax1.set_xticklabels([])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		# ax1.set_yticklabels(ytick_labels)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		# ax1.text(0.05,0.85,f'type 1: {len(spec_type1[spec_type1 == 1])}',transform=ax1.transAxes,fontsize=16)
		# ax1.text(0.05,0.7,f'type 2: {len(spec_type1[spec_type1 == 2])}',transform=ax1.transAxes,fontsize=16)
		# ax1.set_title(f'{bins[0]} < log {label} < {bins[1]}')
		# ax1.set_ylabel(r'$\nu$ F$_\nu$')
		ax1.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
		ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')
		
		ax1.grid()

		# fig.subplots_adjust(right=0.7)
		
	
		# fig.tight_layout(rect=[0, 0.03, 0.5, 1.0])
		# cbar_ax = fig.add_axes([0.78, 0.1, 0.025, 0.8])
		cb = fig.colorbar(test,ax=axes)
		cb.set_label(r'log L$_{\mathrm{X}}$ [erg/s]')


		plt.savefig('/Users/connor_auge/Desktop/SEDshape_1panel_unwise.png')
		plt.show()

	def plot_sed_hist_6bins(self,param,param2,Fx1,Fx2,Fx3,emis1,emis2,x,y,L,spec_type,f1,f2,f3,f4,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,F1=None,F2=None):
		L = np.asarray(L)
		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
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
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{2-10\mathrm{kev}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-2\mathrm{kev}}$'
		elif param == 'full':
			Fx = Fx3
			# xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-10\mathrm{kev}}$'
			xlabel = r'Log $\frac{\lambda\mathrm{L}_{\lambda}(\mathrm{a})}{\lambda\mathrm{L}_{\lambda}({0.5-10\mathrm{kev}})}$'

		if param2 == '01,10':
			
			legend1 = r'UV: a = 0.1$\mu$m'
			legend2 = r'MIR: a = 10$\mu$m'

			c1 = 'blue'
			c2 = 'red'

		elif param2 == '025,5':
			emis1 = 10**f1
			emis2 = 10**f2

			legend1 = r'a = 0.25$\mu$m'
			legend2 = r'a = 5$\mu$m'

			c1 = 'green'
			c2 = 'orange'

		elif param2 == '100':
			emis1 = 10**f3
			emis2 = f3*np.nan

			legend1 = r'a = 100$\mu$m'
			legend2 = ''

			c1 = 'black'
			c2 = 'black'


		L1e = np.log10(emis1/Fx)
		L2e = np.log10(emis2/Fx)
		bin_size1 = np.arange(-3,3,0.25)
		bin_size2 = np.arange(-3,3,0.25)

		# B1 = np.where(np.logical_and(np.logical_and(f1 > 0.15, f2 >= -0.15),L>44.2))[0]
		# B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		# B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B3 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B4 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]


		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3
		plt.rcParams['ytick.minor.size'] = 1

		xticks = [1E-3,1E-2,1E-1,1,10,100]
		yticks = [1E-2,0.1,1,10]
		# ytick_labels = [1E-1,1E0,1E1]

	
		# fig, axes = plt.subplots(figsize=(12,15),nrows=3,ncols=2)
		# gs = gridspec.GridSpec(3, 2)
		# gs.update(hspace=0.05) # set the spacing between axes
		# # gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		widths = [3,0.25]
		xticks1 = [1E-3,1E-2,1E-1,1E0,1E1,1E2]
		xticks2 = [-2,-1,0,1,2]
		yticks1 = [1E-2,0.1,1,10]
		yticks2 = [0,10,20,30,40,50]

		xticklabel1 = [-3,-2,-1,0,1,2]
		xticklabel2 = [-2,-1,0,1,2]
		yticklabel1 = [-2,-1,0,1]
		yticklabel2 = [0,10,20,30,40,50]

		fig = plt.figure(figsize=(18,15),constrained_layout=False)
		# gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig, left=0.1,right=0.89,top=0.93,bottom=0.12, wspace=0.05, hspace=0.0, width_ratios=widths)
		gs1 = gridspec.GridSpec(nrows=3, ncols=2, figure=fig, width_ratios=widths, top=0.9, bottom=0.1, left=0.10, right=0.48, wspace=0.05, hspace=0.05)
		gs2 = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, top=0.9, bottom=0.11, left=0.6, right=0.95, hspace=0.11)

		# fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]
		spec_type1 = spec_type[B1]
		# print('Lx 1:',len(L1[L1 > 45]),L1[L1 > 45])

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(42.5,46,10),cmap='rainbow')
		
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5,rasterized=True)
		ax1.plot(np.nanmedian(10**median_wavelength[B1],axis=0),np.nanmedian(10**median_flux[B1],axis=0),color='k',lw=3.5)
		axcb1 = fig.colorbar(lc1,orientation='horizontal',pad=-0.1)
		axcb1.set_clim(42.5,46)
		axcb1.remove()

		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(8E-5,7E2)
		ax1.set_ylim(5E-3,50)
		ax1.set_xticklabels([])
		ax1.set_xticks(xticks1)
		ax1.set_yticks(yticks1)
		ax1.set_yticklabels(yticklabel1)
		ax1.text(0.05,0.7,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_ylabel(r'Log $\frac{\lambda\mathrm{L}_\lambda}{\lambda\mathrm{L}_\lambda(1\mu\mathrm{m})}$',fontsize=25)
		ax1.set_title(r'Restframe SEDs Normalized at 1$\mu$m',fontsize=22)
		# ax1.set_xlabel(r'Rest Wavelength [$\mu$m]')

		ax2 = fig.add_subplot(gs1[1,0])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]
		spec_type3 = spec_type[B3]
		# print('Lx 3:',len(L3[L3 > 45]),L3[L3 > 45])

		lc2 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5,rasterized=True)
		ax2.plot(np.nanmedian(10**median_wavelength[B3],axis=0),np.nanmedian(10**median_flux[B3],axis=0),color='k',lw=3.5)
		# ax2.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm3),s82X_nuFnu_upper/np.mean(s82X_norm3)]),'v',color='k',ms=7)
		axcb2 = fig.colorbar(lc2,orientation='horizontal',pad=-0.1)
		axcb2.set_clim(42.5,46)
		axcb2.remove()

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(8E-5,7E2)
		ax2.set_ylim(5E-3,50)
		ax2.set_yticklabels([])
		ax2.set_xticklabels([])
		ax2.set_xticks(xticks1)
		ax2.set_yticks(yticks1)
		ax2.text(0.05,0.7,f'n = {len(x3)}',transform=ax2.transAxes)
		# ax2.set_ylabel(r'$\lambda$L$_\lambda$ normalized at 1$\mu$m',fontsize=25)
		# ax2.set_xlabel(r'Rest Wavelength [$\mu$m]',fontsize=22)

		ax3 = fig.add_subplot(gs1[2,0])
		x4 = x[B4]
		y4 = y[B4]
		L4 = L[B4]
		spec_type4 = spec_type[B4]
		# print('Lx 4:',len(L4[L4 > 45]),L4[L4 > 45])

		lc3 = self.multilines(x4,y4,L4,cmap='rainbow',lw=1.5,rasterized=True)
		ax3.plot(np.nanmedian(10**median_wavelength[B4],axis=0),np.nanmedian(10**median_flux[B4],axis=0),color='k',lw=3.5)
		# ax3.plot(125,np.mean([cosmos_nuFnu_upper/np.mean(cosmos_norm4),s82X_nuFnu_upper/np.mean(s82X_norm4)]),'v',color='k',ms=7)
		axcb3 = fig.colorbar(lc3,orientation='horizontal',pad=-0.1)
		axcb3.set_clim(42.5,46)
		axcb3.remove()

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(8E-5,7E2)
		ax3.set_ylim(5E-3,50)
		ax3.set_xticklabels(xticklabel1)
		ax3.set_xticks(xticks1)
		ax3.set_yticks(yticks1)
		ax3.set_yticklabels([])
		ax3.text(0.05,0.7,f'n = {len(x4)}',transform=ax3.transAxes)
		# ax3.set_ylabel(r'$\lambda$ F$_\lambda$')
		ax3.set_xlabel(r'Log $\lambda$ [$\mu$m]',fontsize=26)

		ax4 = fig.add_subplot(gs2[0])
		ax4.hist(L1e[B1],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax4.axvline(np.nanmedian(L1e[B1]),color=c1,ls='--',lw=2)
		ax4.hist(L2e[B1],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax4.axvline(np.nanmedian(L2e[B1]),color=c2,ls='--',lw=2)
		ax4.legend()

		ax4.set_ylabel('N',fontsize=25)
		ax4.set_xticklabels([])
		ax4.set_ylim(0,35)
		ax4.set_xlim(-2.5,2.5)
		ax4.set_xticks(xticks2)
		ax4.set_title(r'Ratios of L$_\mathrm{UV}$ & L$_\mathrm{MIR}$ vs. L$_\mathrm{X}$',fontsize=22)
		# ax4.text(-2.1,30,'UV')
		# ax4.text(-2.1,27,'MIR')


		ax5 = fig.add_subplot(gs2[1])
		ax5.hist(L1e[B3],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax5.axvline(np.nanmedian(L1e[B3]),color=c1,ls='--',lw=2)
		ax5.hist(L2e[B3],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax5.axvline(np.nanmedian(L2e[B3]),color=c2,ls='--',lw=2)
		# ax5.legend()

		ax5.set_ylim(0,35)
		ax5.set_xticklabels([])
		ax5.set_xlim(-2.5,2.5)
		ax5.set_xticks(xticks2)
		ax5.set_yticklabels([])
		# ax5.set_xlabel(xlabel)

		ax6 = fig.add_subplot(gs2[2])
		ax6.hist(L1e[B4],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax6.axvline(np.nanmedian(L1e[B4]),color=c1,ls='--',lw=2)
		ax6.hist(L2e[B4],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax6.axvline(np.nanmedian(L2e[B4]),color=c2,ls='--',lw=2)
		# ax6.legend()

		ax6.set_ylim(0,35)
		ax6.set_xlim(-2.5,2.5)
		ax6.set_xticks(xticks2)
		ax6.set_xlabel(xlabel,fontsize=26)
		ax6.set_yticklabels([])

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()
		ax6.grid()


		cbar_ax = fig.add_subplot(gs1[:,1])
		# fig.tight_layout()
		# fig.subplots_adjust(bottom=0.17)
		# fig.tight_layout(rect=[0.1, 0.5, 0.9, 0.9])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'Log L$_{\mathrm{X}}$ [erg/s]')

		# cb = fig.colorbar(test,ax=axes)
		# cb.set_label(r'log $L_{\mathrm{X}}$ [erg/s]')

		plt.savefig('/Users/connor_auge/Desktop/6panel.png')
		plt.show()

	def plot_hist_bins(self,param,x,y,L,f1,f2,f3,f4,F1=None,F2=None):
		L = np.asarray(L)

		if param == 'Lx':
			xlabel = r'log L$_{\mathrm{X}}$ [erg s$^{-1}$]'
			xticks = [42,43,44,45,46]

			X = np.asarray([44.6,43.5,43.5,44.1,43.2])

			def tick_function(X):
				X = 10**X
				V = X/3.8E33
				solar = np.log10(V)
				return '%.2f' % solar

			new_tick_locations = np.array([42,43,44,45,46])
			xlabel2 = r'log L$_{\mathrm{X}}$ [L$_\odot$]'

		elif param == 'Lbol':
			xlabel = r'log L$_{\mathrm{bol}}$ [L$_\odot$]'	
			xticks = [11,12,13,14,15]
			L = np.log10(L) - np.log10(3.8E33)

			X = np.asarray([46.8,46.1,46.1,46.6,45.5]) - np.log10(3.8E33)

			# def tick_function(X):
			# 	X = 10**X
			# 	V = X*3.8E33
			# 	solar = np.log10(V)
			# 	return '%d' % solar

			# new_tick_locations = np.array([11,12,13,14,15])
			# xlabel2 = r'log L$_{\mathrm{bol}}$ [erg s$^{-1}$]'

		elif param == 'Lone':
			# xlabel = r'log L$_{1.0 \mu \mathrm{m}}$ [erg s$^{-1}$]'
			xlabel = r'log M$_\bigstar$ [M$_\odot$]'
			xticks = [43,44,45,46]
			xticks = [10,11,12,13]
			L = np.log10((L/3.8E33)*0.8)

			X = np.asarray([11.9, 11.2, 11.1, 11.8, 11.2]) + np.log10(0.8)
			
			# def tick_function(X):
			# 	V = X*0.8
			# 	mass = np.log10(V)
			# 	return '%.2f' % mass

			# new_tick_locations = np.array([43,44,45,46,47])
			# xlabel2 = r'log M$_\Big\star$ [M$_\odot$]'

		elif param == 'abs_corr':

			xlabel = 'X-ray Correction Factor'
			xticks = [0,1,2,3]
			L = 1/L


		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		mark = np.asarray(F1)

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]
		# print(len(mark),len(B1),len(B2),len(B3),len(B4),len(B5))
		print(L[B1])

		mark1, mark2, mark3, mark4, mark5 = mark[B1], mark[B2], mark[B3], mark[B4], mark[B5]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		bin_size = np.arange(min(L)-0.5,max(L)+0.25,0.25)

		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(7,15),nrows=5,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle('0.9 < z < 1.1',fontsize=22)

		# ax1 = plt.subplot(511)
		ax1 = plt.subplot(gs[0])
		# ax0 = ax1.twiny()
		ax1.hist(L[B1],bins=bin_size,color='gray',alpha=0.7)
		# ax1.hist(L[B1][mark1 == 0],bins=bin_size,histtype='step',color='blue',lw=2.5,label='COSMOS')
		# ax1.hist(L[B1][mark1 == 1],bins=bin_size,histtype='step',color='red',lw=2.5,label='S82X')
		ax1.axvline(np.nanmedian(L[B1]),color='k',ls='--',lw=3)
		# ax1.axvline(X[0],color='r',ls='--',lw=2)

		ax1.set_xticklabels([])
		ax1.set_ylim(0,50)
		ax1.set_xticks(xticks)
		
		# ax2 = plt.subplot(512)
		ax2 = plt.subplot(gs[1])
		# ax00 = ax2.twiny()
		ax2.hist(L[B2],bins=bin_size,color='gray',alpha=0.7)
		# ax2.hist(L[B2][mark2 == 0],bins=bin_size,histtype='step',color='blue',lw=2.5,label='COSMOS')
		# ax2.hist(L[B2][mark2 == 1],bins=bin_size,histtype='step',color='red',lw=2.5,label='S82X')
		ax2.axvline(np.nanmedian(L[B2]),color='k',ls='--',lw=3)
		# ax2.axvline(X[1],color='r',ls='--',lw=2)

		ax2.set_xticklabels([])
		# ax2.set_yticklabels([])
		ax2.set_ylim(0,50)
		ax2.set_xticks(xticks)


		# ax3 = plt.subplot(513)
		ax3 = plt.subplot(gs[2])
		ax3.hist(L[B3],bins=bin_size,color='gray',alpha=0.7)
		# ax3.hist(L[B3][mark3 == 0],bins=bin_size,histtype='step',color='blue',lw=2.5,label='COSMOS')
		# ax3.hist(L[B3][mark3 == 1],bins=bin_size,histtype='step',color='red',lw=2.5,label='S82X')
		ax3.axvline(np.nanmedian(L[B3]),color='k',ls='--',lw=3)
		# ax3.axvline(X[2],color='r',ls='--',lw=2)

		ax3.set_xticklabels([])
		ax3.set_ylim(0,50)
		ax3.set_xticks(xticks)

		# ax4 = plt.subplot(514)
		ax4 = plt.subplot(gs[3])
		ax4.hist(L[B4],bins=bin_size,color='gray',alpha=0.7)
		# ax4.hist(L[B4][mark4 == 0],bins=bin_size,histtype='step',color='blue',lw=2.5,label='COSMOS')
		# ax4.hist(L[B4][mark4 == 1],bins=bin_size,histtype='step',color='red',lw=2.5,label='S82X')
		ax4.axvline(np.nanmedian(L[B4]),color='k',ls='--',lw=3)
		# ax4.axvline(X[3],color='r',ls='--',lw=2)

		ax4.set_xticklabels([])
		# ax4.set_yticklabels([])
		ax4.set_ylim(0,50)
		ax4.set_xticks(xticks)

		# ax5 = plt.subplot(515)
		ax5 = plt.subplot(gs[4])
		ax5.hist(L[B5],bins=bin_size,color='gray',alpha=0.7)
		# ax5.hist(L[B5][mark5 == 0],bins=bin_size,histtype='step',color='blue',lw=2.5,label='COSMOS')
		# ax5.hist(L[B5][mark5 == 1],bins=bin_size,histtype='step',color='red',lw=2.5,label='S82X')
		ax5.axvline(np.nanmedian(L[B5]),color='k',ls='--',lw=3)
		# ax5.axvline(X[4],color='r',ls='--',lw=2)
		# ax5.legend()

		# ax5.set_xticklabels([])
		ax5.set_ylim(0,50)
		ax5.set_xticks(xticks)
		ax5.set_xlabel(xlabel)
		

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_{param}_5panel_hist.png')
		plt.show()

		print(len(L[B1][mark1 == 0]),len(L[B1][mark1 == 1]))
		print(len(L[B2][mark2 == 0]),len(L[B2][mark2 == 1]))
		print(len(L[B3][mark3 == 0]),len(L[B3][mark3 == 1]))
		print(len(L[B4][mark4 == 0]),len(L[B4][mark4 == 1]))
		print(len(L[B5][mark5 == 0]),len(L[B5][mark5 == 1]))


	def plot_hist_emission_bins(self,savestring,param,param2,Lx,x,y,f1,f2,f3,f4,Fx1,Fx2,Fx3,emis1,emis2,F1=None,suptitle=None):

		x = np.asarray(x)
		y = np.asarray(y)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)

		if param == 'hard':
			Fx = Fx1
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{2-10\mathrm{kev}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-2\mathrm{kev}}$'
		elif param == 'full':
			Fx = Fx3
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-10\mathrm{kev}}$'
			# xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{\mathrm{bol}}$'

		if param2 == '01,10':
			
			legend1 = r'a = 0.1$\mu$m'
			legend2 = r'a = 10$\mu$m'

			c1 = 'blue'
			c2 = 'red'

		elif param2 == '025,5':
			emis1 = 10**f1
			emis2 = 10**f2

			legend1 = r'a = 0.25$\mu$m'
			legend2 = r'a = 5$\mu$m'

			c1 = 'green'
			c2 = 'orange'

		elif param2 == '100':
			emis1 = 10**f3
			emis2 = f3*np.nan

			legend1 = r'a = 100$\mu$m'
			legend2 = ''

			c1 = 'black'
			c2 = 'black'


		# L1 = np.log10(emis1/Fx)
		# L2 = np.log10(emis2/Fx)

		Lx = np.asarray([10**i for i in Lx])
		Lx = Lx/F1

		L1 = np.log10(emis1/Lx)
		L2 = np.log10(emis2/Lx)



		xticks = [-3,-2,-1,0,1,2,3]

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# try:
		# 	xlim=[min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25]
		# 	bin_size1 = np.arange(min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25,0.25)
		# 	bin_size2 = np.arange(min(L2[np.isfinite(L2)])-0.5,max(L2[np.isfinite(L2)])+0.25,0.25)

		# except ValueError:
		# 	bin_size2 = np.arange(min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25,0.25)
		xlim=[-3,3]
		bin_size1 = np.arange(-4,4,0.25)
		bin_size2 = np.arange(-4,4,0.25)



		fig, axes = plt.subplots(figsize=(6,15),nrows=5,ncols=1)
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle(suptitle,fontsize=22)
		# fig.suptitle('0.6 < z < 0.8 & 0.9 < z < 1.1',fontsize=20)

		ax1 = plt.subplot(gs[0])
		ax1.hist(L1[B1],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax1.axvline(np.nanmedian(L1[B1]),color=c1,ls='--',lw=2)
		ax1.hist(L2[B1],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax1.axvline(np.nanmedian(L2[B1]),color=c2,ls='--',lw=2)

		ax1.set_xticklabels([])
		ax1.set_ylim(0,50)
		ax1.set_xlim(xlim[0],xlim[1])
		ax1.set_xticks(xticks)
		

		ax2 = plt.subplot(gs[1])
		ax2.hist(L1[B2],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax2.axvline(np.nanmedian(L1[B2]),color=c1,ls='--',lw=2)
		ax2.hist(L2[B2],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax2.axvline(np.nanmedian(L2[B2]),color=c2,ls='--',lw=2)

		ax2.set_xticklabels([])
		ax2.set_ylim(0,50)
		ax2.set_xlim(xlim[0],xlim[1])
		ax2.set_xticks(xticks)


		ax3 = plt.subplot(gs[2])
		ax3.hist(L1[B3],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax3.axvline(np.nanmedian(L1[B3]),color=c1,ls='--',lw=2)
		ax3.hist(L2[B3],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax3.axvline(np.nanmedian(L2[B3]),color=c2,ls='--',lw=2)

		ax3.set_xticklabels([])
		ax3.set_ylim(0,50)
		ax3.set_xlim(xlim[0],xlim[1])
		ax3.set_xticks(xticks)


		ax4 = plt.subplot(gs[3])
		ax4.hist(L1[B4],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax4.axvline(np.nanmedian(L1[B4]),color=c1,ls='--',lw=2)
		ax4.hist(L2[B4],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax4.axvline(np.nanmedian(L2[B4]),color=c2,ls='--',lw=2)

		ax4.set_xticklabels([])
		ax4.set_ylim(0,50)
		ax4.set_xlim(xlim[0],xlim[1])
		ax4.set_xticks(xticks)


		ax5 = plt.subplot(gs[4])
		ax5.hist(L1[B5],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax5.axvline(np.nanmedian(L1[B5]),color=c1,ls='--',lw=2)
		ax5.hist(L2[B5],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax5.axvline(np.nanmedian(L2[B5]),color=c2,ls='--',lw=2)
		ax5.legend()

		ax5.set_ylim(0,50)
		ax5.set_xlim(xlim[0],xlim[1])
		ax5.set_xticks(xticks)
		ax5.set_xlabel(xlabel)
		

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_5panel_'+savestring+'.png')
		plt.show()



	def plot_hist_emission_bins_24check(self,savestring,param,param2,Lx,x,y,f1,f2,f3,f4,Fx1,Fx2,Fx3,emis1,emis2,F1=None,F2=None,suptitle=None):

		x = np.asarray(x)
		y = np.asarray(y)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)
		check_24 = np.asarray(F2)

		if param == 'hard':
			Fx = Fx1
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{2-10\mathrm{kev}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-2\mathrm{kev}}$'
		elif param == 'full':
			Fx = Fx3
			xlabel = r'log $\lambda$L$_\mathrm{a}$/$\lambda$L$_{0.5-10\mathrm{kev}}$'

		if param2 == '01,10':
			
			legend1 = r'a = 0.1$\mu$m'
			legend2 = r'a = 10$\mu$m'

			c1 = 'blue'
			c2 = 'red'

		elif param2 == '025,5':
			emis1 = 10**f1
			emis2 = 10**f2

			legend1 = r'a = 0.25$\mu$m'
			legend2 = r'a = 5$\mu$m'

			c1 = 'green'
			c2 = 'orange'

		elif param2 == '100':
			emis1 = 10**f3
			emis2 = f3*np.nan

			legend1 = r'a = 100$\mu$m'
			legend2 = ''

			c1 = 'black'
			c2 = 'black'


		# L1 = np.log10(emis1/Fx)
		# L2 = np.log10(emis2/Fx)

		Lx = np.asarray([10**i for i in Lx])
		Lx = Lx/F1

		L1 = np.log10(emis1/Lx)
		L2 = np.log10(emis2/Lx)


		xticks = [-2,-1,0,1,2]

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'no detection')))[0]

		# B1 = np.where(np.logical_and(f1 > 0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		# B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]	
		# B3 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 >= -0.15,check_24 == 'detection')))[0]
		# B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,check_24 == 'detection')))[0]
		# B5 = np.where(check_24 == 'no detection')[0]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		# try:
		# 	xlim=[min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25]
		# 	bin_size1 = np.arange(min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25,0.25)
		# 	bin_size2 = np.arange(min(L2[np.isfinite(L2)])-0.5,max(L2[np.isfinite(L2)])+0.25,0.25)

		# except ValueError:
		# 	bin_size2 = np.arange(min(L1[np.isfinite(L1)])-0.5,max(L1[np.isfinite(L1)])+0.25,0.25)
		xlim=[-2.75,2.75]
		bin_size1 = np.arange(-3,3,0.25)
		bin_size2 = np.arange(-3,3,0.25)



		fig, axes = plt.subplots(figsize=(6,15),nrows=5,ncols=1)
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle(suptitle,fontsize=22)

		ax1 = plt.subplot(gs[0])
		ax1.hist(L1[B1],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax1.axvline(np.nanmedian(L1[B1]),color=c1,ls='--',lw=2)
		ax1.hist(L2[B1],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax1.axvline(np.nanmedian(L2[B1]),color=c2,ls='--',lw=2)

		ax1.set_xticklabels([])
		ax1.set_ylim(0,50)
		ax1.set_xlim(xlim[0],xlim[1])
		ax1.set_xticks(xticks)
		

		ax2 = plt.subplot(gs[1])
		ax2.hist(L1[B2],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax2.axvline(np.nanmedian(L1[B2]),color=c1,ls='--',lw=2)
		ax2.hist(L2[B2],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax2.axvline(np.nanmedian(L2[B2]),color=c2,ls='--',lw=2)

		ax2.set_xticklabels([])
		ax2.set_ylim(0,50)
		ax2.set_xlim(xlim[0],xlim[1])
		ax2.set_xticks(xticks)


		ax3 = plt.subplot(gs[2])
		ax3.hist(L1[B3],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax3.axvline(np.nanmedian(L1[B3]),color=c1,ls='--',lw=2)
		ax3.hist(L2[B3],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax3.axvline(np.nanmedian(L2[B3]),color=c2,ls='--',lw=2)

		ax3.set_xticklabels([])
		ax3.set_ylim(0,50)
		ax3.set_xlim(xlim[0],xlim[1])
		ax3.set_xticks(xticks)


		ax4 = plt.subplot(gs[3])
		ax4.hist(L1[B4],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax4.axvline(np.nanmedian(L1[B4]),color=c1,ls='--',lw=2)
		ax4.hist(L2[B4],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax4.axvline(np.nanmedian(L2[B4]),color=c2,ls='--',lw=2)

		ax4.set_xticklabels([])
		ax4.set_ylim(0,50)
		ax4.set_xlim(xlim[0],xlim[1])
		ax4.set_xticks(xticks)


		ax5 = plt.subplot(gs[4])
		ax5.hist(L1[B5],bins=bin_size1,histtype='step',color=c1,alpha=0.85,label=legend1,lw=2)
		ax5.axvline(np.nanmedian(L1[B5]),color=c1,ls='--',lw=2)
		ax5.hist(L2[B5],bins=bin_size2,histtype='step',color=c2,alpha=0.85,label=legend2,lw=2)
		ax5.axvline(np.nanmedian(L2[B5]),color=c2,ls='--',lw=2)
		ax5.legend()

		ax5.set_ylim(0,50)
		ax5.set_xlim(xlim[0],xlim[1])
		ax5.set_xticks(xticks)
		ax5.set_xlabel(xlabel)
		

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_5panel_'+savestring+'.png')
		plt.show()


	def emission_scatter(self,param,x,y,L,f1,f2,f3,f4,Fx1,Fx2,Fx3,emis1,emis2):

		x = np.asarray(x)
		y = np.asarray(y)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)

		if param == 'hard':
			Fx = Fx1
			xlabel = r'$\nu$F$_{2-10\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'$\nu$F$_{0.5-2\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'
		elif param == 'full':
			Fx = Fx3
			xlabel = r'$\nu$F$_{0.5-10\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'

		xlabel = r'log $\nu$F$_{10\mu \mathrm{m}}$'
		ylabel = r'log $\nu$F$_{0.1\mu \mathrm{m}}$'

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xlim = [-2.3,2.3]
		ylim = [-2.3,2.3]
		xticks = [-2,-1,0,1,2]
		yticks = [-2,-1,0,1,2]
		xticklabels = ['-2','-1','0','1','2']
		yticklabels = ['-2','-1','0','1','2']

		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(5,15),nrows=5,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		gs = gridspec.GridSpec(5, 1)
		gs.update(hspace=0.05) # set the spacing between axes
		gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = plt.subplot(gs[0])
		ax1.scatter(np.log10(emis2[B1]),np.log10(emis1[B1]),c=L[B1],cmap='rainbow',alpha=0.8)
		# ax1.scatter(np.log10(Fx[B1]),np.log10(emis1[B1]),c='b',alpha=0.8,label='a = 0.1')
		# ax1.scatter(np.log10(Fx[B1]),np.log10(emis2[B1]),c='r',alpha=0.8,label='a = 10')
		ax1.set_xticklabels([])
		ax1.set_ylim(ylim[0],ylim[1])
		ax1.set_xlim(xlim[0],xlim[1])
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_xticklabels([])
		ax1.set_yticklabels(yticklabels)
		
		ax2 = plt.subplot(gs[1])
		ax2.scatter(np.log10(emis2[B2]),np.log10(emis1[B2]),c=L[B2],cmap='rainbow',alpha=0.8)
		# ax2.scatter(np.log10(Fx[B2]),np.log10(emis1[B2]),c='b',alpha=0.8,label='a = 0.1')
		# ax2.scatter(np.log10(Fx[B2]),np.log10(emis2[B2]),c='r',alpha=0.8,label='a = 10')
		ax2.set_xticklabels([])
		ax2.set_ylim(ylim[0],ylim[1])
		ax2.set_xlim(xlim[0],xlim[1])
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_xticklabels([])
		ax2.set_yticklabels(yticklabels)

		ax3 = plt.subplot(gs[2])
		ax3.scatter(np.log10(emis2[B3]),np.log10(emis1[B3]),c=L[B3],cmap='rainbow',alpha=0.8)
		# ax3.scatter(np.log10(Fx[B3]),np.log10(emis1[B3]),c='b',alpha=0.8,label='a = 0.1')
		# ax3.scatter(np.log10(Fx[B3]),np.log10(emis2[B3]),c='r',alpha=0.8,label='a = 10')
		ax3.set_xticklabels([])
		ax3.set_ylim(ylim[0],ylim[1])
		ax3.set_xlim(xlim[0],xlim[1])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.set_xticklabels([])
		ax3.set_yticklabels(yticklabels)
		ax3.set_ylabel(ylabel)

		ax4 = plt.subplot(gs[3])
		ax4.scatter(np.log10(emis2[B4]),np.log10(emis1[B4]),c=L[B4],cmap='rainbow',alpha=0.8)
		# ax4.scatter(np.log10(Fx[B4]),np.log10(emis1[B4]),c='b',alpha=0.8,label='a = 0.1')
		# ax4.scatter(np.log10(Fx[B4]),np.log10(emis2[B4]),c='r',alpha=0.8,label='a = 10')
		ax4.set_xticklabels([])
		ax4.set_ylim(ylim[0],ylim[1])
		ax4.set_xlim(xlim[0],xlim[1])
		ax4.set_xticks(xticks)
		ax4.set_yticks(yticks)
		ax4.set_xticklabels([])
		ax4.set_yticklabels(yticklabels)

		ax5 = plt.subplot(gs[4])
		ax5.scatter(np.log10(emis2[B5]),np.log10(emis1[B5]),c=L[B5],cmap='rainbow',alpha=0.8)
		# ax5.scatter(np.log10(Fx[B5]),np.log10(emis1[B5]),c='b',alpha=0.8,label=r'a = 0.1$\mu$m')
		# ax5.scatter(np.log10(Fx[B5]),np.log10(emis2[B5]),c='r',alpha=0.8,label=r'a = 10$\mu$m')
		ax5.set_ylim(ylim[0],ylim[1])
		ax5.set_xlim(xlim[0],xlim[1])
		ax5.set_xticks(xticks)
		ax5.set_yticks(yticks)
		ax5.set_xticklabels(xticklabels)
		ax5.set_yticklabels(yticklabels)
		ax5.set_xlabel(xlabel)
		ax5.legend()

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()

		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_5panel_{param}_emiss_scatter3.png')
		plt.show()

	def emission_scatter_1bin(self,param,x,y,L,f1,f2,f3,f4,Fx1,Fx2,Fx3,emis1,emis2):

		x = np.asarray(x)
		y = np.asarray(y)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)

		if param == 'hard':
			Fx = Fx1
			xlabel = r'$\nu$F$_{2-10\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'
		elif param == 'soft':
			Fx = Fx2
			xlabel = r'$\nu$F$_{0.5-2\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'
		elif param == 'full':
			Fx = Fx3
			xlabel = r'$\nu$F$_{0.5-10\mathrm{kev}}$/$\nu$F$_{1\mu \mathrm{m}}$'

		xlabel = r'log $\nu$F$_{10\mu \mathrm{m}}$/$\nu$F$_{1\mu \mathrm{m}}$'
		ylabel = r'log $\nu$F$_{0.1\mu \mathrm{m}}$/$\nu$F$_{1\mu \mathrm{m}}$'

		B1 = np.where(np.logical_and(f1 > 0.15, f2 >= -0.15))[0]
		B2 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),f2 >= -0.15))[0]	
		B3 = np.where(np.logical_and(f1 < -0.15, f2 >= -0.15))[0]
		B4 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		plt.rcParams['font.size']=22
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		xlim = [-2.3,2.3]
		ylim = [-2.3,2.3]
		xticks = [-2,-1,0,1,2]
		yticks = [-2,-1,0,1,2]
		xticklabels = ['-2','-1','0','1','2']
		yticklabels = ['-2','-1','0','1','2']

		# fig, ((ax1),(ax2),(ax3),(ax4),(ax5)) = plt.subplots(figsize=(8,15),nrows=5,ncols=1,sharex=True,sharey=True)
		fig, axes = plt.subplots(figsize=(12,12),nrows=5,ncols=1)
		# fig = plt.subplots(figsize=(8,15))
		# gs = gridspec.GridSpec(5, 1)
		# gs.update(hspace=0.05) # set the spacing between axes
		# gs.update(left=0.19,right=0.88,top=0.93,bottom=0.08)

		fig.suptitle('0.9 < z < 1.1',fontsize=22)

		ax1 = plt.subplot(111)
		ax1.scatter(np.log10(emis2[B1]),np.log10(emis1[B1]),c=L[B1],cmap='rainbow',alpha=0.8,s=50)
		# ax1.scatter(np.log10(Fx[B1]),np.log10(emis1[B1]),c='b',alpha=0.8,label='a = 0.1')
		# ax1.scatter(np.log10(Fx[B1]),np.log10(emis2[B1]),c='r',alpha=0.8,label='a = 10')
		# ax1.set_xticklabels([])
		ax1.set_ylim(-0.5,1.5)
		ax1.set_xlim(-0.5,1.5)
		# ax1.set_xticks(xticks)
		# ax1.set_yticks(yticks)
		# ax1.set_xticklabels([])
		# ax1.set_yticklabels(yticklabels)
		ax1.set_ylabel(ylabel)
		ax1.set_xlabel(xlabel)
		ax1.grid()
		
		
		# plt.colorbar()
		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_5panel_{param}_emiss_scatter_1bin.png')
		plt.show()





