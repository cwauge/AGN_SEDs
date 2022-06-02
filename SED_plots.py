import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import LineCollection
from astropy.cosmology import FlatLambdaCDM
from filters import Filters
from SED_v7 import Flux_to_Lum
from collections import OrderedDict
from scipy import interpolate


class Plotter():
	'''A class to plot the properties of each source from the AGN class'''

	def __init__(self,ID,z,wavelength,flux,frac_err,Lx=None,Lbol=None,spec_type=None):
		self.ID = np.asarray(ID)
		self.z = np.asarray(z)
		self.wavelength_array = np.asarray(wavelength)
		self.flux_array = np.asarray(flux)
		self.flux_err = np.asarray(frac_err)*np.asarray(flux)
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

	def hist(self,L,x_label):
		plt.rcParams['font.size']=12
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		new_tick_locations = np.array([43,44,45,46,47,48])
	
		fig = plt.figure(figsize=(8,7))
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twiny()
		plt.hist(L,bins=np.arange(min(L),max(L)+1,0.25))
		ax1.set_xlabel(x_label+r'[erg s$^{-1}$]')
		ax2.set_xlabel(x_label+r'[L$_\odot$]')
		ax1.set_xlim(min(L)-0.25,max(L)+0.1)
		ax2.set_xlim(ax1.get_xlim())
		ax2.set_xticks(new_tick_locations)
		ax2.set_xticklabels([self.tick_function(43),self.tick_function(44),self.tick_function(45),self.tick_function(46),self.tick_function(47),self.tick_function(48)])
		ax1.text(0.85,0.75,f'n = {len(L)}',transform=ax1.transAxes)
		plt.show()


	def PlotSingleSED(self,flux_point=None,wfir=None,ffir=None):
		# print('PLOT')	

		plt.rcParams['font.size']=12
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		self.f_interp = interpolate.interp1d(np.log10(self.wavelength_array[~np.isnan(self.flux_array)]),np.log10(self.flux_array[~np.isnan(self.flux_array)]),kind='linear',fill_value='extrapolate')
		y = 10**self.f_interp(np.log10(self.wavelength_array))

		fig, ax = plt.subplots(figsize=(14,8))
		ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
		ax.set_ylabel(r'$\nu$ F$_\nu$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(5E-5,7E2)
		# ax.set_ylim(1E-2,3E2)
		# ax.set_ylim(7E-5,80)
		ax.set_title(str(self.ID))
		ax.text(0.05,0.7,r'L$_{X}$ = '+str(self.Lx),transform=ax.transAxes)

		plt.grid()
		ax.plot(wfir,ffir,color='green',lw=5,alpha=0.3)

		self.SED_line = ax.plot(self.wavelength_array,self.flux_array)
		self.SED_points = ax.plot(self.wavelength_array,self.flux_array,color='k',marker='x')
		self.SED_errs = ax.errorbar(self.wavelength_array,self.flux_array,yerr=self.flux_err)
		# self.interp = ax.plot(self.wavelength_array,y,'r')
		ax.plot(100,flux_point,'x',color='red')
		# plt.savefig('/Users/connor_auge/Desktop/checkS82X/'+str(self.ID))
		plt.show()


	def PlotSingleNiceSED(self):

		plt.rcParams['font.size'] = 18
		plt.rcParams['axes.linewidth'] = 2.5
		plt.rcParams['xtick.major.size'] = 2.5
		plt.rcParams['xtick.major.width'] = 2.5
		plt.rcParams['ytick.major.size'] = 2.5
		plt.rcParams['ytick.major.width'] = 2.5

		fig, ax = plt.subplots(figsize=(10, 8))
		
		ax.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax.set_ylabel(r'Normalized $\nu$ F$_\nu$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(5E-2, 3E2)
		# ax.set_ylim(1E-2,3E2)
		ax.set_ylim(1E-1, 50)
		# ax.set_title(str(self.ID))
		# ax.text(0.05, 0.7, r'L$_{X}$ = '+str(self.Lx), transform=ax.transAxes)
		ax.text(0.04, 1.02, 'UV', transform=ax.transAxes)
		ax.text(0.19, 1.02, 'Optical', transform=ax.transAxes)
		ax.text(0.4, 1.02, 'NIR', transform=ax.transAxes)
		ax.text(0.58, 1.02, 'MIR', transform=ax.transAxes)
		ax.text(0.85, 1.02, 'FIR', transform=ax.transAxes)

		# secax = ax.secondary_xaxis('top')
		# secax.set_xticklabels([])
		# secax.set_xticklabels(['UV', 'Optical', 'NIR', 'MIR', 'FIR'])
		# secax.set_xlabel('test')

		ax.grid()

		self.SED_line = ax.plot(self.wavelength_array, self.flux_array,color='b',lw=3)
		# self.SED_points = ax.plot(self.wavelength_array, self.flux_array, color='k', marker='x')
		# self.SED_errs = ax.errorbar(self.wavelength_array, self.flux_array, yerr=self.flux_err)
		# ax.plot([0.25,5,100],flux_point,'x',color='red')

		plt.show()


	def plot_median(self,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None):

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		plt.rcParams['font.size']=18
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		fig, ax = plt.subplots(figsize=(8,8))
		ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
		ax.set_ylabel(r'$\nu$ F$_\nu$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(8E-5,7E2)
		ax.set_ylim(3E-2,60)
		ax.set_title('COSMOS Type-2')

		plt.grid()
		plt.text(0.05,10,f'n = {len(median_wavelength)}')
		# all_lines = ax.plot(self.wavelength_array.T,self.flux_array.T,color='green',alpha=0.4)
		percentile_25 = ax.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,25,axis=0),color='blue')
		percentile_75 = ax.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,75,axis=0),color='blue')
		percentile_95 = ax.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,95,axis=0),color='gray')
		percentile_5 = ax.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,5,axis=0),color='gray')
		fill = ax.fill_between(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,75,axis=0),np.nanpercentile(10**median_flux,25,axis=0),color='blue',alpha=0.4)
		fill_955 = ax.fill_between(np.nanmedian(10**median_wavelength,axis=0),np.nanpercentile(10**median_flux,95,axis=0),np.nanpercentile(10**median_flux,5,axis=0),color='gray',alpha=0.25)

		median_line = ax.plot(np.nanmedian(10**median_wavelength,axis=0),np.nanmedian(10**median_flux,axis=0),color='k',lw=3.5)
		median_line_ext = ax.plot(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanmedian(10**median_flux_ext,axis=0),color='k',lw=3.5,ls='--')
		percentile_25_ext = ax.plot(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,25,axis=0),color='blue')
		percentile_75_ext = ax.plot(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,75,axis=0),color='blue')
		percentile_95_ext = ax.plot(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,95,axis=0),color='gray')
		percentile_5_ext = ax.plot(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,5,axis=0),color='gray')
		fill_ext = ax.fill_between(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,75,axis=0),np.nanpercentile(10**median_flux_ext,25,axis=0),color='blue',alpha=0.4)
		fill_ext_955 = ax.fill_between(np.nanmedian(10**median_wavelength_ext,axis=0),np.nanpercentile(10**median_flux_ext,95,axis=0),np.nanpercentile(10**median_flux_ext,5,axis=0),color='gray',alpha=0.25)


		plt.show()

	def plot_multi_SED(self,savestring,x,y,L,median_x,median_y,median_x2=None,median_y2=None,median_x_ext=None,median_y_ext=None,flux_point=None,suptitle=None,norm=None,upper_w=None,upper_lim=None,mark=None,spec_z=None):
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

		plt.rcParams['font.size']=26
		plt.rcParams['axes.linewidth']=2.5
		plt.rcParams['xtick.major.size']=4.5
		plt.rcParams['xtick.major.width'] = 3.5
		plt.rcParams['ytick.major.size']=4.5
		plt.rcParams['ytick.major.width'] = 3.5


		fig, ax = plt.subplots(figsize=(20,15))
		ax.set_aspect(1)
		ax.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		# ax.set_xlim(8E-5,7E2)
		# ax.set_xlim(8E-1,7E2)
		ax.set_ylim(3E-3,120)
		# ax.set_title(suptitle)

		plt.grid()
		ax.text(0.15,0.85,f'n = {len(L)}',transform=ax.transAxes)

	
		upper_seg = np.stack((cosmos_s82x_wave,cosmos_s82x_list),axis=2)
		upper_all = LineCollection(upper_seg,color='gray',alpha=0.3)
		ax.add_collection(upper_all)

		lc = self.multilines(x_data[L >= clim1-0.1],y_data[L >= clim1-0.1],L[L >= clim1-0.1],cmap=cmap,lw=2,alpha=0.85,rasterized=True)
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
		axcb.mappable.set_clim(clim1,clim2)
		axcb.set_label(r'log L$_{0.5-10\mathrm{keV}}$ [erg s$^{-1}$]')
		median_line = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanmedian(10**median_y,axis=0),color='k',lw=5.5)
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

		upper_lim_points = ax.plot(median_upper_wave2,median_upper2,'v',ms=12,color='k')
		upper_lim_line = ax.plot(median_upper_wave2,median_upper2,color='k',lw=5.0)
		# upper_lim_line_25 = ax.plot(median_upper_wave2,upper_25_2,'--',color='k',lw=3.5)
		# upper_lim_line_75 = ax.plot(median_upper_wave2,upper_75_2,'--',color='k',lw=3.5)

		plt.xlim(5E-5,1E3)
		plt.ylim(1E-4,200)
		# flux_point_wave = np.zeros(np.shape(flux_point))
		# flux_point_wave[flux_point_wave == 0] = 100
		# ax.plot(flux_point_wave,flux_point,'x',color='red')
		plt.tight_layout()

		plt.savefig('/Users/connor_auge/Desktop/final_paper_43/Multi_SEDs'+savestring+'.pdf')
		plt.show()

	def plot_multi_SED_ShapeColor(self,savestring,x,y,L,median_x,median_y,median_x_ext=None,median_y_ext=None,flux_point=None,suptitle=None,norm=None,upper_w=None,upper_lim=None,mark=None,spec_z=None,uv_slope=None,mir_slope1=None,mir_slope2=None):
		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		median_x = np.asarray(median_x)
		median_y = np.asarray(median_y)

		x_data, y_data, L_data = [], [], []
		x_upper, y_upper, L_upper = [], [], []

		cosmos_s82x_list = []
		cosmos_s82x_wave = []
		for i in range(len(y)):

			# upper_wave = np.array([1036928.77,1697691.33,2536859.83,3557125.92,5191371.41])
			# rest_upper_w = upper_wave/(1+spec_z[i])
			# rest_upper_w_cgs = rest_upper_w*1E-8
			# rest_upper_w_microns = rest_upper_w*1E-4
			# rest_upper_w_freq = 3E10/rest_upper_w_cgs

			# cosmos_upper_lim_jy = np.array([5000.0,10200.0,8100.0,10700.0,15400.0])*1E-6
			# s82X_upper_lim_jy = np.array([np.nan,np.nan,13000.0,12900.0,14800.0])*1E-6
			# goodsN_upper_lim_jy = np.array([1600.0,3600.0,9000.0,12900.0,12600.0])*1E-6
			# goodsS_upper_lim_jy = np.array([1100.0,3400.0,8300.0,11500.0,11300.0])*1E-6

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



			cosmos_norm = norm[mark == 0]
			s82x_norm = norm[mark == 1]
			goodsN_norm = norm[mark == 2]
			goodsS_norm = norm[mark == 3]


			if mark[i] == 0:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 1:
				if np.isnan(y[i][-8]):
					cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-8], y[i][-7], y[i][-6]])
					cosmos_s82x_list.append(a)
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 2:
				if np.isnan(y[i][-3]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
				else:
					cosmos_s82x_list.append(y[i][-3:])
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			elif mark[i] == 3:
				if np.isnan(y[i][-5]):
					cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
				else:
					a = np.array([y[i][-5], y[i][-4], y[i][-3]])
					cosmos_s82x_list.append(a)
					# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			cosmos_s82x_wave.append(rest_upper_w_microns)

			# if mark[i] == 0:
			# 	if np.isnan(y[i][-3]):
			# 		cosmos_s82x_list.append(Flux_to_Lum(cosmos_nuFnu_upper,spec_z[i])/norm[i])
			# 	else:
			# 		cosmos_s82x_list.append(y[i][-5:])
			# 		# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			# elif mark[i] == 1:
			# 	if np.isnan(y[i][-8]):
			# 		cosmos_s82x_list.append(Flux_to_Lum(s82X_nuFnu_upper,spec_z[i])/norm[i])
			# 	else:
			# 		a = np.array([np.nan, np.nan, y[i][-8], y[i][-7], y[i][-6]])
			# 		cosmos_s82x_list.append(a)
			# 		# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			# elif mark[i] == 2:
			# 	if np.isnan(y[i][-3]):
			# 		cosmos_s82x_list.append(Flux_to_Lum(goodsN_nuFnu_upper,spec_z[i])/norm[i])
			# 	else:
			# 		cosmos_s82x_list.append(y[i][-5:])
			# 		# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			# elif mark[i] == 3:
			# 	if np.isnan(y[i][-5]):
			# 		cosmos_s82x_list.append(Flux_to_Lum(goodsS_nuFnu_upper,spec_z[i])/norm[i])
			# 	else:
			# 		a = np.array([y[i][-7], y[i][-6], y[i][-5], y[i][-4], y[i][-3]])
			# 		cosmos_s82x_list.append(a)
			# 		# cosmos_s82x_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
			# cosmos_s82x_wave.append(rest_upper_w_microns)

		cosmos_s82x_wave = np.asarray(cosmos_s82x_wave)
		cosmos_s82x_list = np.asarray(cosmos_s82x_list)

		median_upper_wave = np.nanmedian(cosmos_s82x_wave,axis=0)
		median_upper = np.nanmedian(cosmos_s82x_list,axis=0)
		upper_25_2 = np.nanpercentile(cosmos_s82x_list,25,axis=0)
		upper_75_2 = np.nanpercentile(cosmos_s82x_list,75,axis=0)

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

		# upper_w = np.vstack(upper_w)
		# upper_lim = np.vstack(upper_lim)

		# print(np.shape(upper_w))
		# print(np.shape(upper_lim))


		clim1 = 42.5
		clim2 = 46

		# x[y > 5E2] = np.nan
		# y[y > 5E2] = np.nan
		# x[y < 1E-4] = np.nan
		# y[y < 1E-4] = np.nan

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		shp = np.zeros(np.shape(L))
		shp[B1] = 45
		shp[B2] = 35
		shp[B3] = 25
		shp[B4] = 15
		shp[B5] = 0.5

		loLx = min(L[~np.isnan(L)])-0.25
		hiLx = max(L[~np.isnan(L)])+0.25

		plt.rcParams['font.size']=26
		plt.rcParams['axes.linewidth']=2.5
		plt.rcParams['xtick.major.size']=4.5
		plt.rcParams['xtick.major.width'] = 3.5
		plt.rcParams['ytick.major.size']=4.5
		plt.rcParams['ytick.major.width'] = 3.5


		cmap = mpl.colors.ListedColormap([c5,c4,c3,c2,c1])
		bounds = [0.0, 10, 20, 30, 40, 50]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


		fig, ax = plt.subplots(figsize=(20,15))
		ax.set_aspect(1)
		ax.set_xlabel(r'Rest Wavelength [$\mu$m]')
		ax.set_ylabel(r'Normalized $\lambda$ L$_\lambda$')
		ax.set_xscale('log')
		ax.set_yscale('log')
		# ax.set_xlim(8E-5,7E2)
		# ax.set_xlim(8E-1,7E2)
		ax.set_ylim(3E-3,120)
		# ax.set_title(suptitle)

		plt.grid()
		ax.text(0.15,0.85,f'n = {len(L)}',transform=ax.transAxes)

		upper_seg = np.stack((cosmos_s82x_wave,cosmos_s82x_list),axis=2)
		upper_all = LineCollection(upper_seg,color='gray',alpha=0.3)
		ax.add_collection(upper_all)

		print(np.shape(x_data))
		print(np.shape(y_data))
		print(np.shape(uv_slope))
		print(np.shape(B1))
		print(np.shape(y_data[B1]))

		# lc1 = plt.plot(x_data[B1].T,y_data[B1].T,c=c1,alpha=0.4,rasterized=True)
		# lc2 = plt.plot(x_data[B2].T,y_data[B2].T,c=c2,alpha=0.4,rasterized=True)
		# lc3 = plt.plot(x_data[B3].T,y_data[B3].T,c=c3,alpha=0.4,rasterized=True)
		# lc4 = plt.plot(x_data[B4].T,y_data[B4].T,c=c4,alpha=0.4,rasterized=True)
		# lc5 = plt.plot(x_data[B5].T,y_data[B5].T,c=c5,alpha=0.4,rasterized=True)

		lc = self.multilines(x_data,y_data,shp,cmap=cmap,alpha=0.4,rasterized=True)
		
		# plt.plot(x_data[L >= clim1-0.1], y_data[L >= clim1-0.1],'.',color='k')
		# upper_lim_points = plt.plot(upper_w,upper_lim,color='k')  
		# lc_upper = self.multilines(upper_w[L >= clim1-0.1],upper_lim[L >= clim1-0.1],L[L >= clim1-0.1],cmap='rainbow',ls='--')

		# upper_lim = plt.plot(upper_wave,mean_upper/norm,'.',color='k')
		# lc2 = self.multilines(x[L < clim1],y[L < clim1],L[L < clim1],cmap='gray')
		# points = plt.plot(x[L >= clim1-0.1],y[L >= clim1-0.1],'.',color='k')
 
		axcb2 = fig.colorbar(lc)
		# axcb2.mappable.set_clim(20,100)
		axcb2.remove()
		test = plt.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(1,50,10),cmap=cmap)
		# axcb = fig.colorbar(test)
		axcb = fig.colorbar(test,cmap=cmap, norm=norm, ticks=[5,15,25,35,45])
		axcb.ax.set_yticklabels(['5','4','3','2','1'])
		# axcb.mappable.set_clim(0,50)
		axcb.set_label('Panel Number')
		median_line = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanmedian(10**median_y,axis=0),color='k',lw=5.5)
		percentile_25 = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,25,axis=0),color='k',ls='--',lw=3.5)
		percentile_75 = ax.plot(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,75,axis=0),color='k',ls='--',lw=3.5)
		xray = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanmedian(y_data[L >= clim1-0.1][:,:2],axis=0),color='k',lw=5.5)
		xray_percentile_25 = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data[L >= clim1-0.1][:,:2],25,axis=0),color='k',ls='--',lw=3.5)
		xray_percentile_75 = ax.plot(np.nanmedian(x_data[L >= clim1-0.1][:,:2],axis=0),np.nanpercentile(y_data[L >= clim1-0.1][:,:2],75,axis=0),color='k',ls='--',lw=3.5)

		# ax.fill_between(np.nanmedian(10**median_x,axis=0),np.nanpercentile(10**median_y,75,axis=0),np.nanpercentile(10**median_y,25,axis=0),color='k',alpha=0.8,zorder=-1)

		FIR_med_x = np.nanmedian(x, axis=0)[np.nanmedian(x, axis=0) > 50]
		FIR_med_y = np.nanmedian(y, axis=0)[np.nanmedian(x, axis=0) > 50]

		# FIR_data_line = ax.plot(FIR_med_x, FIR_med_y, color='k',lw=3.0)

		upper_lim_points = ax.plot(median_upper_wave,median_upper,'v',ms=12,color='k')
		upper_lim_line = ax.plot(median_upper_wave,median_upper,color='k',lw=5.0)
		upper_lim_line_25 = ax.plot(median_upper_wave,upper_25_2,'--',color='k',lw=3.5)
		upper_lim_line_75 = ax.plot(median_upper_wave,upper_75_2,'--',color='k',lw=3.5)

		# plt.xlim(1E-2,7E2)
		plt.ylim(1E-4,200)
		# flux_point_wave = np.zeros(np.shape(flux_point))
		# flux_point_wave[flux_point_wave == 0] = 100
		# ax.plot(flux_point_wave,flux_point,'x',color='red')
		plt.title(r'log L$_{\mathrm{X}}$ > 43 erg/s')
		plt.tight_layout()

		plt.savefig('/Users/connor_auge/Desktop/final_paper1/Shapes_Multi_SEDs'+savestring+'.pdf')
		plt.show()

	def BPT_SEDs(self,savestring,bpt_class,x,y,L,median_x,median_y):

		clim1 = 42
		clim2 = 44.5

		print(L)

		L = np.asarray(L)
		x = np.asarray(x)
		y = np.asarray(y)

		sort = L.argsort()
		x = x[sort]
		y = y[sort]
		L = L[sort]
		bpt_class = bpt_class[sort]

		median_wavelength = np.asarray(median_x)
		median_flux = np.asarray(median_y)

		print(bpt_class)
		print(len(bpt_class))

		B1 = bpt_class == 'sf'
		B2 = bpt_class == 'comp'
		B3 = bpt_class == 'agn'

		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		xticks = [1E-2,1E-1,1,10,100]
		yticks = [0.1,1,10]

		fig = plt.figure(figsize=(18,6))
		gs1 = fig.add_gridspec(nrows=1,ncols=4,top=0.9,bottom=0.15,width_ratios=[1.25,1.25,1.25,0.075],wspace=-0.2)

		ax1 = fig.add_subplot(gs1[0,0])
		x1 = x[B1]
		y1 = y[B1]
		L1 = L[B1]

		test = ax1.scatter(np.ones(10)*-1,np.ones(10)*-1,c=np.linspace(clim1,clim2,10),cmap='rainbow')
		lc1 = self.multilines(x1,y1,L1,cmap='rainbow',lw=1.5)
		axcb1 = fig.colorbar(lc1)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.remove()
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(9E-2,7E2)
		ax1.set_ylim(5E-2,50)
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.text(0.05,0.8,f'n = {len(x1)}',transform=ax1.transAxes)
		ax1.set_title('Star Forming')
		ax1.set_ylabel(r'Normalized $\lambda$L$_{\lambda}$')


		ax2 = fig.add_subplot(gs1[0,1])
		x2 = x[B2]
		y2 = y[B2]
		L2 = L[B2]

		lc2 = self.multilines(x2,y2,L2,cmap='rainbow',lw=1.5)
		axcb2 = fig.colorbar(lc2)
		axcb2.mappable.set_clim(clim1, clim2)
		axcb2.remove()
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(9E-2,7E2)
		ax2.set_ylim(5E-2,50)
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_yticklabels([])
		ax2.text(0.05,0.8,f'n = {len(x2)}',transform=ax2.transAxes)
		ax2.set_title('Composite')
		ax2.set_xlabel(r'Rest wavelength [$\mu$m]')


		ax3 = fig.add_subplot(gs1[0,2])
		x3 = x[B3]
		y3 = y[B3]
		L3 = L[B3]

		lc3 = self.multilines(x3,y3,L3,cmap='rainbow',lw=1.5)
		axcb3 = fig.colorbar(lc3)
		axcb3.mappable.set_clim(clim1,clim2)
		axcb3.remove()
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(9E-2,7E2)
		ax3.set_ylim(5E-2,50)
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.set_yticklabels([])
		ax3.text(0.05,0.8,f'n = {len(x3)}',transform=ax3.transAxes)
		ax3.set_title('AGN')


		ax1.grid()
		ax2.grid()
		ax3.grid()


		cbar_ax = fig.add_subplot(gs1[0,3])
		cb = fig.colorbar(test,cax=cbar_ax)
		cb.set_label(r'log L$_{\mathrm{X}}$ (2-10kev) [erg/s]')		

		plt.savefig('/Users/connor_auge/Desktop/obs_prop/'+savestring+'.pdf')
		plt.show()

	def BPT_panels(self,savestring,bpt_class,x,y,L,uv_slope,mir_slope1,mir_slope2):



		linestyles = OrderedDict(
    	[('solid',               (0, ())),
     	('loosely dotted',      (0, (1, 10))),
     	('dotted',              (0, (1, 5))),
     	('densely dotted',      (0, (1, 1))),

     	('loosely dashed',      (0, (5, 10))),
     	('dashed',              (0, (5, 5))),
     	('densely dashed',      (0, (5, 1))),

    	('loosely dashdotted',  (0, (3, 10, 1, 10))),
   	  	('dashdotted',          (0, (3, 5, 1, 5))),
   	  	('densely dashdotted',  (0, (3, 1, 1, 1))),

     	('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     	('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     	('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

		clim1=42
		clim2=44.5
		z_new = 1.0
		z_new = [0.0,0.8,1.5,2.5,3.0]

		x = np.asarray(x)
		y = np.asarray(y)
		bpt_class = np.asarray(bpt_class)

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.2))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope >  0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope >= -0.3, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]


		# B1 = bpt_class == 'sf'
		# B2 = bpt_class == 'comp'
		# B3 = bpt_class == 'agn'

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		fig = plt.figure(figsize=(12,10))
		ax1 = plt.subplot(111)
		bpt = ax1.scatter(x,y,c=L,cmap='rainbow',edgecolor='black',s=70,alpha=0.5)
		axcb1 = fig.colorbar(bpt)
		axcb1.mappable.set_clim(clim1,clim2)
		axcb1.set_label(r'log L$_{2-10\mathrm{keV}}$ [erg/s]')
		# axcb1.remove()

		# med_pt = ax1.scatter([np.median(x[L < 42.5]), np.median(x[(L > 42.5) & (L < 43.0)]), np.median(x[(L > 43.0) & (L < 43.5)]), np.median(x[(L > 43.5) & (L < 44.0)]), np.median(x[(L > 44.0) & (L < 44.5)])], [
        #             np.median(y[L < 42.5]), np.median(y[(L > 42.5) & (L < 43.0)]), np.median(y[(L > 43.0) & (L < 43.5)]), np.median(y[(L > 43.5) & (L < 44.0)]), np.median(y[(L > 44.0) & (L < 44.5)])], c=[42.25, 42.75, 43.25, 43.75, 44.25], cmap='rainbow', marker='s', s=150, edgecolor='k')
		# axcb2 = fig.colorbar(med_pt)
		# axcb2.mappable.set_clim(clim1,clim2)
		# axcb2.set_label(r'log L$_{2-10\mathrm{keV}}$ [erg/s]')
		# axcb2.remove()

		# ax1.scatter(x[B1],y[B1],c=c1,edgecolor='black',s=70,label='SF')
		# ax1.scatter(x[B2],y[B2],c=c2,edgecolor='black',s=70,label='Comp')
		# ax1.scatter(x[B3],y[B3],c=c3,edgecolor='black',s=70,label='AGN')

		# ax1.scatter(x[B1],y[B1],c=c1,edgecolor='k',s=70,alpha=0.5)
		# ax1.scatter(x[B2],y[B2],c=c2,edgecolor='k',s=70,alpha=0.5)
		# ax1.scatter(x[B3],y[B3],c=c3,edgecolor='k',s=70,alpha=0.5)
		# ax1.scatter(x[B4],y[B4],c=c4,edgecolor='k',s=70,alpha=0.5)
		# ax1.scatter(x[B5],y[B5],c=c5,edgecolor='k',s=70,alpha=0.5)
		# ax1.scatter(np.median(x[B1]),np.median(y[B1]),c=c1,edgecolor='k',s=150,marker='s',label='Panel 1')
		# ax1.scatter(np.median(x[B2]),np.median(y[B2]),c=c2,edgecolor='k',s=150,marker='s',label='Panel 2')
		# ax1.scatter(np.median(x[B3]),np.median(y[B3]),c=c3,edgecolor='k',s=150,marker='s',label='Panel 3')
		# ax1.scatter(np.median(x[B4]),np.median(y[B4]),c=c4,edgecolor='k',s=150,marker='s',label='Panel 4')
		# ax1.scatter(np.median(x[B5]),np.median(y[B5]),c=c5,edgecolor='k',s=150,marker='s',label='Panel 5')


		ax1.set_xlim(-2.0,0.6)
		ax1.set_ylim(-1.2,1.5)
		ax1.set_xlabel(r'log [NII]$\lambda$6584/H$\alpha$')
		ax1.set_ylabel(r'log [OIII]$\lambda$5007/H$\beta$')

		# kewley = plt.plot(np.arange(-4.1,0.4,0.01),(0.61/(np.arange(-4.1,0.4,0.01)-0.47)+1.19),linestyle='dashed',color='k',label='Maximum Starburst z = 0',alpha=0.75)
		# kewley_z = plt.plot(np.arange(-4.1,0.4,0.01),(0.61/(np.arange(-4.1,0.4,0.01)-0.08-0.1833*z_new)+1.1+0.03*z_new),color='r',lw=3,label='Kewley et al. 2013 z = '+str(z_new))

		a, b, c, d, = 0.917, -0.419, -6.090, 0.0
		A, B, C, D = 0.031, 1.441, -0.879, 0.0
		X = np.arange(-4.1,0.4,0.01) 
		kewley_up = a+b*X+c*X**2+d*X**3
		kewley_low = A+B*X+C*X**2+D*X**3
		# plt.plot(X,kewley_up,color='b',ls='--')
		# plt.plot(X,kewley_low,color='b')

		# kauffmann = plt.plot(np.arange(-4.1,0.0,0.01),(0.61/(np.arange(-4.1,0.0,0.01)-0.05)+1.30),color='k',label='Star forming sequence z = 0',alpha=0.75)
		# kewley_z = plt.plot(np.arange(-4.1,-0.3,0.01),(0.61/(np.arange(-4.1,-0.3,0.01)+0.08-0.1833*z_new[0])+1.1+0.03*z_new[0]),color='b',ls='dotted',lw=3,label='Star forming sequence z = '+str(z_new[0]))
		# kewley_z = plt.plot(np.arange(-4.1,-0.1,0.01),(0.61/(np.arange(-4.1,-0.1,0.01)+0.08-0.1833*z_new[1])+1.1+0.03*z_new[1]),color='cyan',ls='--',lw=3,label='Star forming sequence z = '+str(z_new[1]))
		# kewley_z = plt.plot(np.arange(-4.1,0.0,0.01),(0.61/(np.arange(-4.1,0.0,0.01)+0.08-0.1833*z_new[2])+1.1+0.03*z_new[2]),color='green',ls='dashdot',lw=3,label='Star forming sequence z = '+str(z_new[2]))
		# kewley_z = plt.plot(np.arange(-4.1,0.2,0.01),(0.61/(np.arange(-4.1,0.2,0.01)+0.08-0.1833*z_new[3])+1.1+0.03*z_new[3]),color='orange',ls=linestyles['densely dashdotdotted'],lw=3,label='Star forming sequence z = '+str(z_new[3]))
		# kewley_z = plt.plot(np.arange(-4.1,0.3,0.01),(0.61/(np.arange(-4.1,0.3,0.01)+0.08-0.1833*z_new[4])+1.1+0.03*z_new[4]),color='r',lw=3,label='Star forming sequence z = '+str(z_new[4]))

		kewley_z = plt.plot(np.arange(-4.1,-0.1,0.01),(0.61/(np.arange(-4.1,-0.1,0.01)-0.02-0.1833*z_new[0])+1.2+0.03*z_new[0]),color='b',ls='dotted',lw=3,label='Star forming sequence z = '+str(z_new[0]))
		kewley_z = plt.plot(np.arange(-4.1,0.05,0.01),(0.61/(np.arange(-4.1,0.05,0.01)-0.02-0.1833*z_new[1])+1.2+0.03*z_new[1]),color='cyan',ls='--',lw=3,label='Star forming sequence z = '+str(z_new[1]))
		kewley_z = plt.plot(np.arange(-4.1,0.1,0.01),(0.61/(np.arange(-4.1,0.1,0.01)-0.02-0.1833*z_new[2])+1.2+0.03*z_new[2]),color='green',ls='dashdot',lw=3,label='Star forming sequence z = '+str(z_new[2]))
		kewley_z = plt.plot(np.arange(-4.1,0.4,0.01),(0.61/(np.arange(-4.1,0.4,0.01)-0.02-0.1833*z_new[3])+1.2+0.03*z_new[3]),color='orange',ls=linestyles['densely dashdotdotted'],lw=3,label='Star forming sequence z = '+str(z_new[3]))
		kewley_z = plt.plot(np.arange(-4.1,0.4,0.01),(0.61/(np.arange(-4.1,0.4,0.01)-0.02-0.1833*z_new[4])+1.2+0.03*z_new[4]),color='r',lw=3,label='Star forming sequence z = '+str(z_new[4]))



		plt.legend(fontsize=14,loc='lower left')
		plt.savefig('/Users/connor_auge/Desktop/obs_prop/'+savestring+'2.pdf')
		plt.show()

	def plot_SED_bins(self,L,param,median_wavelength,median_flux,median_wavelength_ext=None,median_flux_ext=None,suptitle=None):

		if param == 'Lx':
			label = r'L$_{X}$'
			bins = np.asarray([42,43,44,45.5])
		elif param == 'Lone':
			label = r'L$_{1 \mu m}$'
			bins = np.asarray([43,44,45,46])
		else:
			label = 'L'
			bins = np.asarray([42,44,46,48])

		L1 = np.where(np.logical_and(L > bins[0],L < bins[1]))[0]
		L2 = np.where(np.logical_and(L > bins[1],L < bins[2]))[0]
		L3 = np.where(np.logical_and(L > bins[2],L < bins[3]))[0]

		median_wavelength = np.asarray(median_wavelength)
		median_flux = np.asarray(median_flux)

		plt.rcParams['font.size']=18
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,6))
		fig.suptitle(suptitle)

		ax1.set_ylabel(r'$\nu$ F$_\nu$')
		ax2.set_xlabel(r'Rest Wavelength [$\mu$ m]')
		
		ax1.set_xscale('log')
		ax1.set_yscale('log')
		ax1.set_xlim(1E-2,7E2)
		ax1.set_ylim(4E-2,30)
		ax1.set_title(f'{bins[0]} < log {label} < {bins[1]}')

		ax2.set_xscale('log')
		ax2.set_yscale('log')
		ax2.set_xlim(1E-2,7E2)
		ax2.set_ylim(4E-2,30)
		ax2.set_yticklabels([])
		ax2.set_title(f'{bins[1]} < log  {label} < {bins[2]}')

		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlim(1E-2,7E2)
		ax3.set_ylim(4E-2,30)
		ax3.set_yticklabels([])
		ax3.set_title(f'{bins[2]} < log  {label} < {bins[3]}')

		ax1.plot(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],25,axis=0),color='blue')
		ax1.plot(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],75,axis=0),color='blue')
		ax1.plot(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],95,axis=0),color='gray')
		ax1.plot(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],5,axis=0),color='gray')
		ax1.fill_between(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],75,axis=0),np.nanpercentile(10**median_flux[L1],25,axis=0),color='blue',alpha=0.4)
		ax1.fill_between(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanpercentile(10**median_flux[L1],95,axis=0),np.nanpercentile(10**median_flux[L1],5,axis=0),color='gray',alpha=0.25)

		ax1.plot(np.nanmedian(10**median_wavelength[L1],axis=0),np.nanmedian(10**median_flux[L1],axis=0),color='k',lw=3.5)
		ax1.plot(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanmedian(10**median_flux_ext[L1],axis=0),color='k',lw=3.5,ls='--')

		ax1.plot(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],25,axis=0),color='blue')
		ax1.plot(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],75,axis=0),color='blue')
		ax1.plot(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],95,axis=0),color='gray')
		ax1.plot(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],5,axis=0),color='gray')
		ax1.fill_between(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],75,axis=0),np.nanpercentile(10**median_flux_ext[L1],25,axis=0),color='blue',alpha=0.4)
		ax1.fill_between(np.nanmedian(10**median_wavelength_ext[L1],axis=0),np.nanpercentile(10**median_flux_ext[L1],95,axis=0),np.nanpercentile(10**median_flux_ext[L1],5,axis=0),color='gray',alpha=0.25)
		ax1.text(0.05,10,f'n = {len(median_wavelength[L1])}')

		ax2.plot(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],25,axis=0),color='blue')
		ax2.plot(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],75,axis=0),color='blue')
		ax2.plot(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],95,axis=0),color='gray')
		ax2.plot(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],5,axis=0),color='gray')
		ax2.fill_between(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],75,axis=0),np.nanpercentile(10**median_flux[L2],25,axis=0),color='blue',alpha=0.4)
		ax2.fill_between(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanpercentile(10**median_flux[L2],95,axis=0),np.nanpercentile(10**median_flux[L2],5,axis=0),color='gray',alpha=0.25)

		ax2.plot(np.nanmedian(10**median_wavelength[L2],axis=0),np.nanmedian(10**median_flux[L2],axis=0),color='k',lw=3.5)
		ax2.plot(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanmedian(10**median_flux_ext[L2],axis=0),color='k',lw=3.5,ls='--')

		ax2.plot(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],25,axis=0),color='blue')
		ax2.plot(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],75,axis=0),color='blue')
		ax2.plot(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],95,axis=0),color='gray')
		ax2.plot(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],5,axis=0),color='gray')
		ax2.fill_between(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],75,axis=0),np.nanpercentile(10**median_flux_ext[L2],25,axis=0),color='blue',alpha=0.4)
		ax2.fill_between(np.nanmedian(10**median_wavelength_ext[L2],axis=0),np.nanpercentile(10**median_flux_ext[L2],95,axis=0),np.nanpercentile(10**median_flux_ext[L2],5,axis=0),color='gray',alpha=0.25)
		ax2.text(0.05,10,f'n = {len(median_wavelength[L2])}')


		ax3.plot(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],25,axis=0),color='blue')
		ax3.plot(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],75,axis=0),color='blue')
		ax3.plot(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],95,axis=0),color='gray')
		ax3.plot(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],5,axis=0),color='gray')
		ax3.fill_between(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],75,axis=0),np.nanpercentile(10**median_flux[L3],25,axis=0),color='blue',alpha=0.4)
		ax3.fill_between(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanpercentile(10**median_flux[L3],95,axis=0),np.nanpercentile(10**median_flux[L3],5,axis=0),color='gray',alpha=0.25)

		ax3.plot(np.nanmedian(10**median_wavelength[L3],axis=0),np.nanmedian(10**median_flux[L3],axis=0),color='k',lw=3.5)
		ax3.plot(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanmedian(10**median_flux_ext[L3],axis=0),color='k',lw=3.5,ls='--')

		ax3.plot(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],25,axis=0),color='blue')
		ax3.plot(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],75,axis=0),color='blue')
		ax3.plot(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],95,axis=0),color='gray')
		ax3.plot(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],5,axis=0),color='gray')
		ax3.fill_between(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],75,axis=0),np.nanpercentile(10**median_flux_ext[L3],25,axis=0),color='blue',alpha=0.4)
		ax3.fill_between(np.nanmedian(10**median_wavelength_ext[L3],axis=0),np.nanpercentile(10**median_flux_ext[L3],95,axis=0),np.nanpercentile(10**median_flux_ext[L3],5,axis=0),color='gray',alpha=0.25)
		ax3.text(0.05,10,f'n = {len(median_wavelength[L3])}')

		ax3.grid()
		ax1.grid()
		ax2.grid()
		# plt.tight_layout()
		plt.savefig(f'/Users/connor_auge/Desktop/Type1_{param}_bins_15z20.png')
		plt.show()


		
		
	def plot_hist_emission(self,savestring,param,param2,Lx,x,y,Fx1,Fx2,Fx3,emis1,emis2,emis3,F1=None,f1=None,f2=None,f3=None):
		x = np.asarray(x)
		y = np.asarray(y)
		Fx1 = np.asarray(Fx1)
		Fx2 = np.asarray(Fx2)
		Fx3 = np.asarray(Fx3)
		emis1 = np.asarray(emis1)
		emis2 = np.asarray(emis2)
		emis3 = np.asarray(emis3)

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
			legend3 = r'a = 100$\mu$m'

			c1 = 'blue'
			c2 = 'red'
			c3 = 'green'

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

		L1 = np.log10(emis1/Fx)
		L2 = np.log10(emis2/Fx)
		L3 = np.log10(emis3/Fx)

		Lx = np.asarray([10**i for i in Lx])
		Lx = Lx/F1

		L1 = np.log10(emis1/Lx)
		L2 = np.log10(emis2/Lx)
		L3 = np.log10(emis3/Lx)


		xticks = [-2,-1,0,1,2,3,4,5]


		plt.rcParams['font.size']=18
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3


		xlim=[-2,5]
		bin_size1 = np.arange(-3,5,0.25)
		bin_size2 = np.arange(-3,5,0.25)

		# bin_size1 = np.arange(55,60,0.25)
		# bin_size2 = np.arange(55,60,0.25)



		fig = plt.figure(figsize=(8,6))
		ax1 = plt.subplot(111)

		ax1.hist(L1,bins=bin_size1,histtype='step',color=c1,alpha=0.75,label=legend1,lw=2)
		ax1.axvline(np.nanmedian(L1),color=c1,ls='--',lw=2)
		ax1.hist(L2,bins=bin_size2,histtype='step',color=c2,alpha=0.95,label=legend2,lw=2)
		ax1.axvline(np.nanmedian(L2),color=c2,ls='--',lw=2)
		ax1.hist(L3,bins=bin_size2,histtype='step',color=c3,alpha=0.75,label=legend3,lw=2)
		ax1.axvline(np.nanmedian(L3),color=c3,ls='--',lw=2)

		# ax1.set_xticklabels([])
		ax1.set_ylim(0,15)
		ax1.set_xlim(xlim[0],xlim[1])
		ax1.set_xticks(xticks)
		ax1.set_xlabel(xlabel)
		ax1.legend()
		ax1.grid()


		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_5panel_'+savestring+'.png')
		plt.show()

	
	
	def plot_hist_bins(self,param,x,y,L,f1,f2,f3,f4):
		L = np.asarray(L)

		if param == 'Lx':
			xlabel = r'log L$_{\mathrm{X}}$ [erg s$^{-1}$]'
			xticks = [42,43,44,45,46]

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
			
			# def tick_function(X):
			# 	V = X*0.8
			# 	mass = np.log10(V)
			# 	return '%.2f' % mass

			# new_tick_locations = np.array([43,44,45,46,47])
			# xlabel2 = r'log M$_\Big\star$ [M$_\odot$]'


		x = np.asarray(x)
		y = np.asarray(y)
		L = np.asarray(L)
		B1 = np.where(np.logical_and(f1 > 0.15, f2 > 0.15))[0]
		B2 = np.where(np.logical_and(f1 > 0.15, np.logical_and(f2 <= 0.1, f2 >= -0.15)))[0]
		B3 = np.where(np.logical_and(np.logical_and(f1 <=0.15, f1 >= -0.15),np.logical_and(f2 <=0.15, f2 >= -0.15)))[0]	
		B4 = np.where(np.logical_and(np.logical_and(f1 <=0.1, f1 >= -0.15),f2 >0.15))[0]		
		B5 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 <=0.15, f2 >= -0.15)))[0]
		B6 = np.where(np.logical_and(f1 < -0.15, f2 > 0.15))[0] 
		B7 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 >= f2)))[0]
		B8 = np.where(np.logical_and(f1 < -0.15, np.logical_and(f2 < -0.15,f4 <= f2)))[0]

		plt.rcParams['font.size']=18
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size']=4
		plt.rcParams['ytick.major.width'] = 3

		bin_size = np.arange(min(L)-0.5,max(L)+0.25,0.25)

		fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(figsize=(9,12),nrows=4,ncols=2,sharex=True,sharey=True)

		ax1 = plt.subplot(421)
		# ax0 = ax1.twiny()
		ax1.hist(L[B1],bins=bin_size,color='gray',alpha=0.8)
		ax1.axvline(np.nanmedian(L[B1]),color='k',ls='--',lw=3)

		ax1.set_xticklabels([])
		ax1.set_ylim(0,30)
		ax1.set_xticks(xticks)
		# ax0.set_xlabel(xlabel2)
		# ax0.set_xlim(ax1.get_xlim())
		# ax0.set_xticks(new_tick_locations)
		# ax0.set_xticklabels([tick_function(new_tick_locations[0]),tick_function(new_tick_locations[1]),tick_function(new_tick_locations[2]),tick_function(new_tick_locations[3]),tick_function(new_tick_locations[4])])

		ax2 = plt.subplot(422)
		# ax00 = ax2.twiny()
		ax2.hist(L[B2],bins=bin_size,color='gray',alpha=0.8)
		ax2.axvline(np.nanmedian(L[B2]),color='k',ls='--',lw=3)

		ax2.set_xticklabels([])
		ax2.set_yticklabels([])
		ax2.set_ylim(0,30)
		ax2.set_xticks(xticks)
		# ax00.set_xlabel(xlabel2)
		# ax00.set_xlim(ax2.get_xlim())
		# ax00.set_xticks(new_tick_locations)
		# ax00.set_xticklabels([tick_function(new_tick_locations[0]),tick_function(new_tick_locations[1]),tick_function(new_tick_locations[2]),tick_function(new_tick_locations[3]),tick_function(new_tick_locations[4])])


		ax3 = plt.subplot(423)
		ax3.hist(L[B3],bins=bin_size,color='gray',alpha=0.8)
		ax3.axvline(np.nanmedian(L[B3]),color='k',ls='--',lw=3)

		ax3.set_xticklabels([])
		ax3.set_ylim(0,30)
		ax3.set_xticks(xticks)

		ax4 = plt.subplot(424)
		ax4.hist(L[B4],bins=bin_size,color='gray',alpha=0.8)
		ax4.axvline(np.nanmedian(L[B4]),color='k',ls='--',lw=3)

		ax4.set_xticklabels([])
		ax4.set_yticklabels([])
		ax4.set_ylim(0,30)
		ax4.set_xticks(xticks)

		ax5 = plt.subplot(425)
		ax5.hist(L[B5],bins=bin_size,color='gray',alpha=0.8)
		ax5.axvline(np.nanmedian(L[B5]),color='k',ls='--',lw=3)

		ax5.set_xticklabels([])
		ax5.set_ylim(0,30)
		ax5.set_xticks(xticks)

		ax6 = plt.subplot(426)
		ax6.hist(L[B6],bins=bin_size,color='gray',alpha=0.8)
		ax6.axvline(np.nanmedian(L[B6]),color='k',ls='--',lw=3)

		ax6.set_yticklabels([])
		ax6.set_xticklabels([])
		ax6.set_ylim(0,30)
		ax6.set_xticks(xticks)

		ax7 = plt.subplot(427)
		ax7.hist(L[B7],bins=bin_size,color='gray',alpha=0.8)
		ax7.axvline(np.nanmedian(L[B7]),color='k',ls='--',lw=3)

		ax7.set_xlabel(xlabel)
		ax7.set_ylim(0,30)
		ax7.set_xticks(xticks)

		ax8 = plt.subplot(428)
		ax8.hist(L[B8],bins=bin_size,color='gray',alpha=0.8)
		ax8.axvline(np.nanmedian(L[B8]),color='k',ls='--',lw=3)

		ax8.set_xlabel(xlabel)
		ax8.set_yticklabels([])
		ax8.set_ylim(0,30)
		ax8.set_xticks(xticks)

		ax1.grid()
		ax2.grid()
		ax3.grid()
		ax4.grid()
		ax5.grid()
		ax6.grid()
		ax7.grid()
		ax8.grid()

		plt.savefig(f'/Users/connor_auge/Desktop/SEDshape_{param}_hist.png')
		plt.show()

	def plot_emission_scatter_bins(self,L,F1,F2,F3,F4=None):

		L = np.asarray(L)
		F1 = np.asarray(F1)
		F2 = np.asarray(F2)
		F3 = np.asarray(F3)
		ticks = [-2,-1,0,1,2]
		ticklabels = ['-2','-1','0','1','2']


		fig, (ax1, ax2, ax3) = plt.subplots(figsize=(19,6),nrows=1,ncols=3)

		a = ax1.scatter(np.log10(F1),np.log10(F2),c=L,cmap='rainbow')
		ax1.set_xlabel(r'Log L$_{0.25 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax1.set_ylabel(r'Log L$_{5.0 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax1.set_xlim(-2,2)
		ax1.set_ylim(-2,2)
		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_xticklabels(ticklabels)
		ax1.set_yticklabels(ticklabels)
		# ax1.text(0.1,0.7,f'n = {len(L[L1])}',transform=ax1.transAxes)
		# ax1.grid()

		ax2.scatter(np.log10(F1),np.log10(F3),c=L,cmap='rainbow')
		ax2.set_xlabel(r'Log L$_{0.25 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax2.set_ylabel(r'Log L$_{100 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax2.set_xlim(-2,2)
		ax2.set_ylim(-2,2)
		ax2.set_xticks(ticks)
		ax2.set_yticks(ticks)
		ax2.set_xticklabels(ticklabels)
		ax2.set_yticklabels(ticklabels)
		# ax2.text(0.1,0.7,f'n = {len(L[L2])}',transform=ax2.transAxes)
		# ax2.grid()

		ax3.scatter(np.log10(F2),np.log10(F3),c=L,cmap='rainbow')
		ax3.set_xlabel(r'Log L$_{5 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax3.set_ylabel(r'Log L$_{100 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax3.set_xlim(-2,2)
		ax3.set_ylim(-2,2)
		ax3.set_xticks(ticks)
		ax3.set_yticks(ticks)
		ax3.set_xticklabels(ticklabels)
		ax3.set_yticklabels(ticklabels)
		# ax3.text(0.1,0.7,f'n = {len(L[L3])}',transform=ax3.transAxes)
		# ax3.grid()

		fig.subplots_adjust(right=0.84)
		cbar_ax = fig.add_axes([0.86, 0.05, 0.025, 0.88])
		cb = fig.colorbar(a,cax=cbar_ax)
		cb.set_label(r'log $L_{\mathrm{X}}$ [erg/s]')
		# plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/3panel_scatter.png')
		plt.show()


	def plot_emission_Lx_scatter_bins(self,L,spec_type,F1,F2,F3,F4=None):

		print(len(L))

		L = np.asarray(L)
		F1 = np.asarray(F1)
		F2 = np.asarray(F2)
		F3 = np.asarray(F3)
		ticks = [-2,-1,0,1,2]
		ticklabels = ['-2','-1','0','1','2']


		fig, (ax1, ax2, ax3) = plt.subplots(figsize=(19,6),nrows=1,ncols=3)

		a = ax1.scatter(np.log10(F1),L,c=L,cmap='rainbow')
		ax1.set_ylabel(r'Log L$_{\mathrm{X}}$ [erg s$^{-1}$]',fontsize=22)
		ax1.set_xlabel(r'Log L$_{0.25 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		# ax1.set_xlim(-2,2)
		ax1.set_xlim(-2,2)
		ax1.set_xticks(ticks)
		ax1.set_xticklabels(ticklabels)
		# ax1.text(0.1,0.7,f'n = {len(L[L1])}',transform=ax1.transAxes)
		# ax1.grid()

		ax2.scatter(np.log10(F2),L,c=L,cmap='rainbow')
		# ax2.set_ylabel(r'Log L$_{\mathrm{X}}$ [erg s$^{-1}$]',fontsize=22)
		ax2.set_xlabel(r'Log L$_{5.0 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		# ax2.set_xlim(-2,2)
		ax2.set_xlim(-2,2)
		ax2.set_xticks(ticks)
		ax2.set_xticklabels(ticklabels)
		# ax2.text(0.1,0.7,f'n = {len(L[L2])}',transform=ax2.transAxes)
		# ax2.grid()

		ax3.scatter(np.log10(F3),L,c=L,cmap='rainbow')
		# ax3.set_xlabel(r'Log L$_{5 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax3.set_xlabel(r'Log L$_{100 \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		# ax3.set_xlim(-2,2)
		ax3.set_xlim(-2,2)
		ax3.set_xticks(ticks)
		ax3.set_xticklabels(ticklabels)
		# ax3.text(0.1,0.7,f'n = {len(L[L3])}',transform=ax3.transAxes)
		# ax3.grid()

		fig.subplots_adjust(right=0.84)
		cbar_ax = fig.add_axes([0.86, 0.05, 0.025, 0.88])
		cb = fig.colorbar(a,cax=cbar_ax)
		cb.set_label(r'log $L_{\mathrm{X}}$ [erg/s]')
		# cb.set_label(r'Spec Type')
		# plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/3panel_Lx_scatter.png')
		plt.show()


	def plot_emission_hist_bins(self,L,F1,F2,F3,F4=None):

		L = np.asarray(L)
		F1 = np.asarray(F1)
		F2 = np.asarray(F2)
		F3 = np.asarray(F3)

		L1 = np.where(L < 43.5)[0]
		L2 = np.where(np.logical_and(L >= 43.5, L <=44.5))[0]
		L3 = np.where(L > 44.5)[0]

		fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14,8),nrows=1,ncols=3,sharey=True,sharex=True)

		ax1.hist(np.log10(F1[L1]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='blue',lw=2.5,label='a = 0.25')
		ax1.hist(np.log10(F1[L1]),bins=np.arange(-2.5,2.5,0.25),color='blue',alpha=0.25)
		ax1.hist(np.log10(F2[L1]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='red',lw=2.5,label='a = 5.0')
		ax1.hist(np.log10(F2[L1]),bins=np.arange(-2.5,2.5,0.25),color='red',alpha=0.25)
		ax1.hist(np.log10(F3[L1]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='green',lw=2.5,label='a = 100')
		ax1.hist(np.log10(F3[L1]),bins=np.arange(-2.5,2.5,0.25),color='green',alpha=0.25)
		ax1.axvline(np.nanmedian(np.log10(F1[L1])),ls='--',lw=3.0,color='blue')
		ax1.axvline(np.nanmedian(np.log10(F2[L1])),ls='--',lw=3.0,color='red')
		ax1.axvline(np.nanmedian(np.log10(F3[L1])),ls='--',lw=3.0,color='green')
		# ax1.set_xlabel(r'Log L$_{\mathrm{a} \; \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax1.text(0.1,0.7,f'n = {len(L[L1])}',transform=ax1.transAxes)
		ax1.set_title(r'log L$_\mathrm{X}$ < 43.5')
		# ax1.grid()

		ax2.hist(np.log10(F1[L2]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='blue',lw=2.5,label='a = 0.25')
		ax2.hist(np.log10(F1[L2]),bins=np.arange(-2.5,2.5,0.25),color='blue',alpha=0.25)
		ax2.hist(np.log10(F2[L2]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='red',lw=2.5,label='a = 5.0')
		ax2.hist(np.log10(F2[L2]),bins=np.arange(-2.5,2.5,0.25),color='red',alpha=0.25)
		ax2.hist(np.log10(F3[L2]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='green',lw=2.5,label='a = 100')
		ax2.hist(np.log10(F3[L2]),bins=np.arange(-2.5,2.5,0.25),color='green',alpha=0.25)
		ax2.axvline(np.nanmedian(np.log10(F1[L2])),ls='--',lw=3.0,color='blue')
		ax2.axvline(np.nanmedian(np.log10(F2[L2])),ls='--',lw=3.0,color='red')
		ax2.axvline(np.nanmedian(np.log10(F3[L2])),ls='--',lw=3.0,color='green')
		ax2.set_xlabel(r'Log L$_{\mathrm{a} \; \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax2.text(0.1,0.7,f'n = {len(L[L2])}',transform=ax2.transAxes)
		ax2.set_title(r'43.5 < log L$_\mathrm{X}$ < 44.5')
		# ax2.grid()

		ax3.hist(np.log10(F1[L3]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='blue',lw=2.5,label='a = 0.25')
		ax3.hist(np.log10(F1[L3]),bins=np.arange(-2.5,2.5,0.25),color='blue',alpha=0.25)
		ax3.hist(np.log10(F2[L3]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='red',lw=2.5,label='a = 5.0')
		ax3.hist(np.log10(F2[L3]),bins=np.arange(-2.5,2.5,0.25),color='red',alpha=0.25)
		ax3.hist(np.log10(F3[L3]),bins=np.arange(-2.5,2.5,0.25),histtype='step',color='green',lw=2.5,label='a = 100')
		ax3.hist(np.log10(F3[L3]),bins=np.arange(-2.5,2.5,0.25),color='green',alpha=0.25)
		ax3.axvline(np.nanmedian(np.log10(F1[L3])),ls='--',lw=3.0,color='blue')
		ax3.axvline(np.nanmedian(np.log10(F2[L3])),ls='--',lw=3.0,color='red')
		ax3.axvline(np.nanmedian(np.log10(F3[L3])),ls='--',lw=3.0,color='green')
		# ax3.set_xlabel(r'Log L$_{\mathrm{a} \; \mu\mathrm{m}}$/L$_{1.0 \mu\mathrm{m}}$',fontsize=18)
		ax3.text(0.1,0.7,f'n = {len(L[L3])}',transform=ax3.transAxes)
		ax3.set_title(r'44.5 < log L$_\mathrm{X}$')
		# ax3.grid()
		ax3.legend()

		plt.tight_layout()
		plt.show()


	def plot_SF_MS(self,z,z2=None,M_in=None,SFR_in=None,labels=None):
		'''
		Function to plot a specific source (or an array of sources) on a star-forming main sequence 
		with the main sequence line for the corresponding redshift 
		'''

		plt.rcParams['font.size']=18
		plt.rcParams['axes.linewidth']=2
		plt.rcParams['xtick.major.size']=2
		plt.rcParams['xtick.major.width'] = 2
		plt.rcParams['ytick.major.size']=2
		plt.rcParams['ytick.major.width'] = 2

		color = ['b','purple','green','orange','red','gray']
		if any(M_in) == None:
			M_in = np.nan
			SFR_in = np.nan
			# labels = np.nan
		if z2 == None:
			z2 = np.nan

		M_star = 10**np.arange(8.5,11.5,0.25)
		s_0 = 0.448 + 1.220*z - 0.174*z**2
		M_0 = 10**(9.458 + 0.865*z - 0.132*z**2)
		gamma = 1.091

		s_02 = 0.448 + 1.220*z2 - 0.174*z2**2
		M_02 = 10**(9.458 + 0.865*z2 - 0.132*z2**2)

		SFR = s_0 - np.log10(1 + (M_star/M_0)**-gamma)
		SFR2 = s_02 - np.log10(1 + (M_star/M_02)**-gamma)

		fig = plt.figure(figsize=(10,8))

		plt.plot(np.log10(M_star),SFR,color='k',lw=3,label='MS z = 1.0')
		plt.xlabel(r'log $(M_{\bigstar}/M_{\odot})$')
		plt.ylabel('log SFR')
		plt.grid()
		

		if len(M_in) > 1:
			for i in range(len(M_in)):
				plt.scatter(M_in[i],np.log10(SFR_in[i]),c=color[i],s=100,label=labels[i])
			plt.ylim(min(np.log10(SFR_in))-0.25,3.0)
			plt.xlim(8,11.5)

		else:
			plt.scatter(M_in[0],np.log10(SFR_in[0]),c='b',s=100,label=labels[0])
			plt.xlim(7.75,11.75)
			plt.ylim(-0.5,2.7)

		if np.isnan(z2) == False:
			plt.plot(np.log10(M_star),SFR2,color='k',ls='--',lw=3,label='MS z = 0.02')



		if any(labels) != None:
			plt.legend(loc='upper left')
			plt.show()
		else:
			plt.show()








