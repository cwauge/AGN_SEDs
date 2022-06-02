'''
Connor Auge - Latest Edits on May 23, 2022
AGN class
    Create SEDs
    Calculate Lbol
    Create file for X-CIGALE
Applicable to any data set (i.e., COSMOS, S82X, GOODS, or GOALS)
'''
import wave
from xml.dom.expatbuilder import FILTER_INTERRUPT
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate 
from scipy import integrate
from filters import Filters


def main(ID,z,filter_name,obs_flux,obs_flux_err,abs_corr=None):
	source = AGN(ID,z,filter_name,obs_flux,obs_flux_err,abs_corr)
	source.MakeSED()
	source.write_xcigale_input()


class AGN():
	'''A class to utilize the photometric data of an AGN to constructs an SED and determine additional properties'''

	def __init__(self,ID,z,filter_name,obs_flux,obs_flux_err,abs_corr=None):
		self.filter_name = filter_name
		self.ID = ID					# object ID
		self.z = z						# best measured redshift
		self.name = filter_name 		# name of observation filter from filter_list.dat file
		self.obs_f = obs_flux 			# observed flux given in micro Janskeys
		self.obs_f_err = obs_flux_err	# error in observed flux given in micro Janskeys
		self.abs_corr = abs_corr        # absorption correction factor for X-ray flux

		filter_list = ascii.read('filter_list.dat')
		self.all_names = np.asarray(filter_list['Name'])
		self.name_err = np.asarray(filter_list['Name_err'])
		self.cent_wavelength = np.asarray(filter_list['central_wavelength'])
		self.upper_lim = np.asarray(filter_list['upper_lim'])

		self.obs_w = Filters('filter_list.dat').pull_filter(self.filter_name,'central wavelength')
		

	def MakeSED(self):
		'''Function to make the SED in the restframe'''

		rest_w = self.obs_w/(1+self.z)
		obs_w_cgs = self.obs_w*1E-8
		self.rest_w_cgs = rest_w*1E-8
		self.rest_w_microns = rest_w*1E-4
		self.rest_freq = 3E10/self.rest_w_cgs

		self.flux_jy = self.obs_f*1E-6
		self.flux_jy_err = self.obs_f_err*1E-6
		self.flux_jy[self.flux_jy <= 0] = np.nan
		self.flux_jy_err[self.flux_jy_err <= 0] = np.nan
		self.flux_jy[np.isnan(self.flux_jy_err)] = np.nan
		self.flux_jy[self.flux_jy_err/self.flux_jy > 0.475] = np.nan # Remove data points that have a high fractional error in flux

		self.flux_cgs = self.flux_jy*1E-23 # flux in cgs: erg s^-1 cm^-2 Hz^-1
		self.Fnu = self.flux_cgs
		self.Flambda = self.flux_cgs*(3E10/obs_w_cgs**2)

		self.nuF_nu = self.rest_freq*self.Fnu
		self.lambdaF_lambda = obs_w_cgs*self.Flambda
		self.lambdaF_lambda_ext = obs_w_cgs*self.Flambda

		self.lambdaL_lambda = self.Flux_to_Lum(self.lambdaF_lambda,self.z)
		self.nuL_nu = self.Flux_to_Lum(self.nuF_nu,self.z)
		# print('6: ', self.nuL_nu)
		# print(self.z)
		self.f_interp = interpolate.interp1d(np.log10(self.rest_w_microns[~np.isnan(self.nuL_nu)]),np.log10(self.nuL_nu[~np.isnan(self.nuL_nu)]),kind='linear',fill_value='extrapolate')

	def CheckSED(self,check_w,check_span=None):
		# Check for an observational data point within check_span microns of a desired wavelength value (check_w)
		# If check_span is not specified use 2 microns 
		max_w = check_w + 15
		if check_span is None:
			min_w = check_w - 2 # 2 microns
		else:
			min_w = check_w - check_span


		wave_range = (self.rest_w_microns <= max_w) & (self.rest_w_microns >= min_w)
		check_flux = self.flux_jy[wave_range]
		# print(self.flux_jy[wave_range])
		if len(check_flux[np.isnan(check_flux)]) == len(check_flux):
			check_return = 'BAD'
		elif any(self.flux_jy[wave_range]) > 0:
			check_return = 'GOOD'
		elif any(self.flux_jy_err[wave_range] > 0):
			check_return = 'GOOD'
		else:
			check_return = 'BAD'
		# print(check_return)
		# check_return = 'GOOD'

		return check_return


	def FIR_extrapolation(self,w):
		self.regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')

		fir_w = self.obs_w[self.regime == 'FIR']
		rest_fir_w = fir_w/(1+self.z)
		rest_fir_w_cgs = rest_fir_w*1E-8
		rest_fir_w_microns = rest_fir_w*1E-4
		rest_fir_freq = 3E10/self.rest_w_cgs

		fir_flux_jy = self.obs_f[self.regime == 'FIR']*1E-6
		fir_flux_jy[fir_flux_jy <= 0] = np.nan

		flux_upper = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')*1E-29 # 3σ upper limits in cgs
		flux_upper /= 3
		nuFnu_upper = flux_upper*rest_fir_freq
		fir_nuFnu_upper = nuFnu_upper[self.regime == 'FIR']

		fir_nuLnu_upper = self.Flux_to_Lum(fir_nuFnu_upper,self.z)

		fir_nuFnu_upper_interp  = interpolate.interp1d(np.log10(rest_fir_w_microns),np.log10(fir_nuLnu_upper),kind='linear',fill_value='extrapolate')
		F100_upper = 10**fir_nuFnu_upper_interp(np.log10(w))

		ext_F100 = 10**self.f_interp(np.log10(w))

		if np.isnan(fir_flux_jy[-3]) == False:
			F100_use = ext_F100
			# self.upper_check = 0

		elif np.isnan(fir_flux_jy[-2]) == False:
			F100_use = ext_F100
			# self.upper_check = 0

		elif np.isnan(fir_flux_jy[-1]) == False:
			F100_use = ext_F100
			# self.upper_check = 0

		elif np.isnan(fir_flux_jy[0]) == False and np.isnan(fir_flux_jy[1]) == False:
			if ext_F100 > F100_upper:
				F100_use = F100_upper
				# self.upper_check = 1
			else:
				F100_use = ext_F100
				# self.upper_check = 0

		else:
			F100_use = F100_upper
			# self.upper_check = 1

		return F100_use


	def median_FIR_filter(self,filtername,Find_value=np.nan):
		upper_lims_flux = Filters('filter_list.dat').pull_filter(filtername,'upper limit')*1E-6 #3σ upper limits in Jy
		upper_lims_flux /= 3 # 1σ upper limits
		wavelength = Filters('filter_list.dat').pull_filter(filtername,'central wavelength')

		rest_w = wavelength/(1+self.z)
		rest_w_cgs = rest_w*1E-8
		rest_w_microns = rest_w*1E-4
		rest_freq = 3E10/rest_w_cgs

		flux_jy = []
		for i in range(len(filtername)):
			if self.flux_jy[self.obs_w == wavelength[i]] > 0:
				flux_jy.append(self.flux_jy[self.obs_w == wavelength[i]][0])
			elif np.isnan(self.flux_jy[self.obs_w == wavelength[i]]):
				flux_jy.append(upper_lims_flux[i])
			else:
				flux_jy.append(upper_lims_flux[i])

		flux_jy = np.asarray(flux_jy)
		flux_cgs = flux_jy*1E-23
		nuFnu = flux_cgs*rest_freq
		nuLnu = self.Flux_to_Lum(nuFnu,self.z)

		nuLnu_interp  = interpolate.interp1d(np.log10(rest_w_microns[~np.isnan(nuLnu)]),np.log10(nuLnu[~np.isnan(nuLnu)]),kind='linear',fill_value='extrapolate')
		nuLnu_data_and_upper = 10**nuLnu_interp(np.log10(rest_w_microns))
		nuLnu_data = 10**self.f_interp(np.log10(rest_w_microns))
		
		ext_F100 = 10**self.f_interp(np.log10(Find_value))
		upper_F100 = 10**nuLnu_interp(np.log10(Find_value))

		if self.flux_jy[self.obs_w == wavelength[-3]] > 0:
			nuLnu_out = nuLnu_data
			F100_out = ext_F100
			# self. upper_check = 0
		elif self.flux_jy[self.obs_w == wavelength[-3]] < 0:
			# if nuLnu_data[-1] < nuLnu_data_and_upper[-1]:
			# 	nuLnu_out = nuLnu_data
			# 	F100_out = ext_F100
			# 	self.upper_check = 0
			# else:
				nuLnu_out = nuLnu_data_and_upper
				F100_out = upper_F100
				# self.upper_check = 1
		else:
			nuLnu_out = nuLnu_data_and_upper
			F100_out = upper_F100
			# self.upper_check = 1

		ind = np.where(self.flux_jy[self.rest_w_microns > 100] > 0)[0]
		if len(ind) > 0:
			self.upper_check = 0
		else:
			self.upper_check = 1

		wave_out = rest_w_microns
		self.F100 = F100_out

		self.FIRnuLnu_out2, self.FIRwave_out2 = np.delete(nuLnu_out,0), np.delete(wave_out,0) 

		if np.isnan(Find_value):
			# print(Find_value)
			return self.FIRnuLnu_out2, self.FIRwave_out2
		else:
			# print(Find_value)
			return self.FIRnuLnu_out2, self.FIRwave_out2, F100_out



	def Find_BPT_class(self,x,y):

		kewley_line = (0.61/(x-0.47)) + 1.19
		kauffmann_line = (0.61/(x-0.05)) +1.30

		if y > kewley_line:
			out = 'agn'
		elif y < kauffmann_line:
			out = 'sf'
		elif (y > kauffmann_line) & (y < kewley_line):
			out = 'comp'
		else:
			out = 'None'
		
		return out


	def Find_Lbol(self):
		x = self.rest_w_cgs
		y = self.nuL_nu

		# F100 = self.FIR_extrapolation(100.0)

		# x = np.append(x,0.01)
		# y = np.append(y,self.jjF100)
		x = np.append(x,self.FIRwave_out2*1E-4)
		y = np.append(y,self.FIRnuLnu_out2)

		sort = x.argsort()
		x,y = x[sort], y[sort]


		Lbol_interp = interpolate.interp1d(np.log10(x[~np.isnan(y)]),np.log10(y[~np.isnan(y)]),kind='linear',fill_value='extrapolate')

		x_interp = np.linspace(np.log10(min(x)),np.log10(max(x)))
		y_interp = 10**Lbol_interp(x_interp)

		x_interp_FIR = np.linspace(np.log10(min(self.FIRwave_out2*1E-4)),np.log10(max(self.FIRwave_out2*1E-4)))
		y_interp_FIR = 10**Lbol_interp(x_interp_FIR)

		# x_interp_FIR = np.linspace(np.log10(0.003),np.log10(0.01))
		# y_interp_FIR = 10**Lbol_interp(x_interp_FIR)
	

		# plt.plot((10**x_interp)*1E4,y_interp)
		# plt.plot((10**x_interp_FIR)*1E4,y_interp_FIR)
		# plt.plot(self.FIRwave_out2,self.FIRnuLnu_out2,'v',color='k')
		# plt.plot(x*1E4,y,'x',color='k')
		# plt.plot(0.01*1E4,self.F100,'x',color='r')
		# plt.xscale('log')
		# plt.yscale('log')
		# plt.show()

		x_interp, y_interp  = x_interp[::-1], y_interp[::-1]
		x_interp_FIR, y_interp_FIR  = x_interp_FIR[::-1], y_interp_FIR[::-1]

		freq = 3E10/10**x_interp
		freq_FIR = 3E10/10**x_interp_FIR

		y_interp = y_interp/freq
		y_interp_FIR = y_interp_FIR/freq_FIR

		f_bol = integrate.trapz(y_interp,freq)
		f_bol_fir = integrate.trapz(y_interp_FIR,freq_FIR)

		cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
		dl = cosmo.luminosity_distance(self.z).value
		dl_cgs = dl*3.086E24

		# L_bol = f_bol*4*np.pi*dl_cgs**2
		# L_bol_FIR = f_bol_fir*4*np.pi*dl_cgs**2

		L_bol = f_bol
		L_bol_FIR = f_bol_fir

		# if max(x) <= 0.01:
			# print('yes')
			# L_bol += L_bol_FIR
		# else:
			# print('no')
		self.L_bol = L_bol
		self.L_bol_FIR = L_bol_FIR
		self.Lbol = L_bol

		return L_bol

	def find_Lum_range(self,xmin,xmax):
		x = self.rest_w_cgs
		y = self.nuL_nu

		x = np.append(x, self.FIRwave_out2*1E-4)
		y = np.append(y, self.FIRnuLnu_out2)

		sort = x.argsort()
		x, y = x[sort], y[sort]

		Lbol_interp = interpolate.interp1d(np.log10(x[~np.isnan(y)]),np.log10(y[~np.isnan(y)]),kind='linear',fill_value='extrapolate')

		x_interp = np.linspace(np.log10(xmin*1E-4), np.log10(xmax*1E-4))
		y_interp = 10**Lbol_interp(x_interp)

		# plt.figure(figsize=(9,6))
		# plt.plot((10**x_interp)*1E4, y_interp, color='b',lw=4,alpha=0.25)
		# plt.plot(self.FIRwave_out2, self.FIRnuLnu_out2, 'v', color='k')
		# plt.plot(x*1E4, y, 'x', color='k')
		# plt.xscale('log')
		# plt.yscale('log')
		# plt.show()

		x_interp, y_interp = x_interp[::-1], y_interp[::-1]
		freq = 3E10/10**x_interp
		y_interp = y_interp/freq
		L_region = integrate.trapz(y_interp, freq)

		return L_region




	def FIR_frac(self):
		return self.L_bol_FIR


	def Lbol_corrections(self,Lx):
		a = 15.33
		b = 11.48
		c = 16.20

		# L_x = [10**i for i in Lx]
		# L_x = 10**Lx
		L_bol = a*(1+(np.log10(Lx/3.8E33)/b)**c)*Lx

		return np.log10(L_bol)


	def SFR(self):
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
		self.regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')

		fir_w = self.obs_w[self.regime == 'FIR']
		rest_fir_w = fir_w/(1+self.z)
		rest_fir_w_cgs = rest_fir_w*1E-8
		rest_fir_w_microns = rest_fir_w*1E-4
		rest_fir_freq = 3E10/self.rest_w_cgs

		fir_flux_jy = self.obs_f[self.regime == 'FIR']*1E-6
		fir_flux_jy[fir_flux_jy <= 0] = np.nan

		flux_upper = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')*1E-29 # 3σ upper limits in cgs
		nuFnu_upper = flux_upper*rest_fir_freq
		fir_nuFnu_upper = nuFnu_upper[self.regime == 'FIR']


	def pull_plot_info(self):
		F1 = self.Find_nuFnu(1.0)
		norm_nuF_nu = self.nuF_nu/F1

		norm_nuL_nu = self.nuL_nu/F1

		norm_lambdaF_lambda = self.lambdaF_lambda/F1

		norm_lambdaL_lambda = self.Flux_to_Lum(norm_lambdaF_lambda,self.z)

		x_out = np.linspace(min(self.rest_w_microns),max(self.rest_w_microns))
		y_out = 10**self.f_interp(np.log10(x_out))/F1

		try:
			return self.ID, self.z, self.rest_w_microns, norm_nuL_nu, self.flux_jy_err/self.flux_jy, self.upper_check
		except AttributeError:
			return self.ID, self.z, self.rest_w_microns, norm_nuL_nu, self.flux_jy_err/self.flux_jy



	def median_SED(self,filt_1,filt_2):
		try:
			w1 = Filters('filter_list.dat').pull_filter(filt_1, 'central wavelength')/(1+self.z)*1E-4
			w2 = Filters('filter_list.dat').pull_filter(filt_2, 'central wavelength')/(1+self.z)*1E-4			

			self.med_x = np.linspace(np.log10(w1), np.log10(w2), 1000)
			self.med_x = np.linspace(np.log10(w1), np.log10(w2), 1000)
			med_interp = interpolate.interp1d(np.log10(self.rest_w_microns),np.log10(self.nuL_nu/self.Find_nuFnu(1.0)),kind='linear',fill_value='extrapolate')
			self.med_y = med_interp(self.med_x)

		except ValueError:
			self.med_x = np.linspace(np.log10(w1), np.log10(w2), 1000)
			self.med_y = np.zeros(1000)
			self.med_y[self.med_y == 0] = np.nan

		return self.med_x, self.med_y




	def extrapolated_median_SED(self):
		regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')
		upper_lims_flux = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')*1E-29
		fir_w = self.obs_w
		rest_w = fir_w/(1+self.z)
		rest_w_cgs = rest_w*1E-8
		rest_w_microns = rest_w*1E-4
		rest_w_fir_microns = rest_w_microns[regime == 'FIR']
		rest_freq = 3E10/self.rest_w_cgs
		upper_lims_nuF_nu = upper_lims_flux*rest_freq

		extrap_FIR_nuF_nu = 10**self.f_interp(rest_w_fir_microns)
		bad_fir = np.where(np.logical_and(np.isnan(self.nuF_nu_ext) == True, regime == 'FIR'))[0]	
		# print('bad fir',bad_fir)
		# print(len(extrap_FIR_nuF_nu))
		# print(len(regime[regime != 'FIR']))

		if extrap_FIR_nuF_nu[1] > upper_lims_nuF_nu[1]:
			self.nuF_nu_ext[bad_fir] = upper_lims_nuF_nu[bad_fir]
		elif extrap_FIR_nuF_nu[1] < 1E-1:
			self.nuF_nu_ext[bad_fir] = np.nanmedian(self.nuF_nu_ext[regime == 'FIR'])
		else: 
			# self.nuF_nu_ext[bad_fir] = extrap_FIR_nuF_nu[bad_fir-(len(regime[regime != 'FIR']))]
			self.nuF_nu_ext[bad_fir] = extrap_FIR_nuF_nu[bad_fir-35]

		try:
			med_x_ext = np.linspace(1.02,2.4,100)
			med_interp_ext = interpolate.interp1d(np.log10(self.rest_w_microns),np.log10(self.nuF_nu_ext/self.Find_nuFnu(1.0)),kind='linear',fill_value='extrapolate')
			med_y_ext = med_interp_ext(med_x_ext)
		except ValueError:
			med_x_ext = np.linspace(1.02,2.4,100)
			med_y_ext = np.zeros(100)
			med_y_ext[med_y_ext == 0] = np.nan

		med_x_ext_out = np.append(self.med_x[-1],med_x_ext)
		med_y_ext_out = np.append(self.med_y[-1],med_y_ext)

		return med_x_ext_out, med_y_ext_out



	def Find_nuFnu(self,wave):
		try:
			if wave == 100.0:
				nuFnu = self.FIR_extrapolation(100.0)

				nuFnu_1 = self.FIR_extrapolation(60.0)
				self.FIRwave_out2 = np.asarray([60,100])
				self.FIRnuLnu_out2 = np.asarray([nuFnu_1, nuFnu])
				self.upper_check = 1
			else:
				nuFnu = 10**self.f_interp(np.log10(wave))
			return nuFnu
		except AttributeError:
			nuFnu = np.nan
			print(f'Object {self.ID} does not have enough good data points to find nuF_nu at {wave} microns.')
			return nuFnu



	def Find_nuFnu_xray(self,band,obs_f_Fx,abs_corr=None,correction=False):
		try:
			if band == 'hard':
				name = ['Fx_hard']
	
			elif band == 'soft':
				name = ['Fx_soft']
	
			elif band == 'full':
				name = ['Fx_full']

			elif band == 'hard_27':
				name = ['Fx_hard_27']

			obs_w_Fx = Filters('filter_list.dat').pull_filter(name,'central wavelength')

			rest_w_Fx = obs_w_Fx/(1+self.z)
			rest_w_cgs_Fx = rest_w_Fx*1E-8
			obs_w_cgs_Fx = obs_w_Fx*1E-8
			rest_w_microns_Fx = rest_w_Fx*1E-4
			rest_freq_Fx = 3E10/rest_w_cgs_Fx
	
			if correction == False:
				flux_jy_Fx = obs_f_Fx*1E-6
			elif correction == True:
				flux_jy_Fx = obs_f_Fx*1E-6
				flux_jy_Fx /= abs_corr
	
			flux_cgs_Fx = flux_jy_Fx*1E-23 # flux in cgs: erg s^-1 cm^-2 Hz^-1
			nuF_nu = rest_freq_Fx*flux_cgs_Fx
			F_lambda = flux_cgs_Fx*(3E10/obs_w_cgs_Fx**2)

			lambdaF_lambda = obs_w_cgs_Fx*F_lambda

			lambdaL_lambda = self.Flux_to_Lum(lambdaF_lambda,self.z)

			return lambdaL_lambda

		except ValueError:
			nuFnu = np.nan
			print('Enter X-ray band. Options are: hard, soft, or full.')

	def Find_nuFnu_xray2(self,w_max,w_min,abs_corr=None,correction=False):

		rest_w_cgs_Fx = self.rest_w_cgs[0:2]
		obs_w_cgs_Fx = self.obs_w[0:2]*1E-8
		rest_w_microns_Fx = rest_w_cgs_Fx*1E4
		rest_freq_Fx = 3E10/rest_w_cgs_Fx
	
		if correction == False:
			flux_jy_Fx = self.obs_f[0:2]*1E-6
		elif correction == True:
			flux_jy_Fx = self.obs_f[0:2]*1E-6
			flux_jy_Fx /= abs_corr

		flux_cgs_Fx = flux_jy_Fx*1E-23 # flux in cgs: erg s^-1 cm^-2 Hz^-1
		nuF_nu = rest_freq_Fx*flux_cgs_Fx
		F_lambda = flux_cgs_Fx*(3E10/obs_w_cgs_Fx**2)

		lambdaF_lambda = obs_w_cgs_Fx*F_lambda
		lambdaL_lambda = self.Flux_to_Lum(lambdaF_lambda,self.z)

		xray_interp = interpolate.interp1d(np.log10(rest_w_microns_Fx),np.log10(lambdaL_lambda),kind='linear',fill_value='extrapolate')

		target_energy = (w_max + w_min)/2
		target_energy *= 1000

		target_wave = 1.23984193/target_energy # 1.23984193 is hc in units of eV * μm
		target_wave = target_wave/(1+self.z)


		lambdaL_lambda_out = 10**xray_interp(np.log10(target_wave))

		return lambdaL_lambda_out 


	def Flux_to_Lum(self,F,z):
		cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

		dl = cosmo.luminosity_distance(z).value # luminosity distance in Mpc
		dl_cgs = dl*(3.086E24)

		# convert flux to luminoisty
		L = F*4*np.pi*dl_cgs**2
		return L

	def Find_slope(self,wi,wf):
		fi = self.Find_nuFnu(wi)/self.Find_nuFnu(1.0)
		ff = self.Find_nuFnu(wf)/self.Find_nuFnu(1.0)

		slope = (np.log10(ff) - np.log10(fi))/(np.log10(wf) - np.log10(wi))

		return slope

	def morph(self,type_of_fit,COSMOS_ID):
		self.name = type_of_fit
		self.ID = COSMOS_ID

		if self.name == 'bd': # deVaucoulers bulge (n=4) + exponential disk (n=1)
			inf = ascii.read('/Users/connor_auge/Research/Disertation/catalogs/COSMOS_Morph/cosmos_hst_bd.csv')
			source = inf['LaigleID']
			BT = inf['BT_I']

		return np.asarray(BT[source == COSMOS_ID])

	def filter_check(self,filt_name):
		if np.isnan(self.obs_f[self.filter_name == filt_name]) == False:

			if self.obs_f[self.filter_name == filt_name] > 0:
				out = 'detection'
			else:
				out = 'no detection'

		else:
			out = 'no detection'

		return out

	def SED_output(self,fname,opt):
		flux_flux_err = np.empty(self.flux_jy.size+self.flux_jy_err.size, dtype=self.flux_jy.dtype)
		filter_name_err = np.asarray([i+'_err' for i in self.filter_name])
		filter_filter_err = np.empty(self.filter_name.size+filter_name_err.size, dtype=filter_name_err.dtype)

		flux_flux_err[0::2] = self.flux_jy
		flux_flux_err[1::2] = self.flux_jy_err

		filter_filter_err[0::2] = self.filter_name
		filter_filter_err[1::2] = filter_name_err

		flux_flux_err = np.append(self.ID,flux_flux_err)
		filter_filter_err = np.append('ID',filter_filter_err)

		t = Table(data=flux_flux_err, names=filter_filter_err)

		if 'w' in opt:
			try:
				fin = fits.open('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname)
				fdata = fin[1].data
				fcols = fin[1].columns.names
				fin.close()
				tin = Table(data=fdata,names=fcols)
				tin.add_row(flux_flux_err)

				tin.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fits',overwrite=True)

			except FileNotFoundError:
				t.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fits',overwrite=True)


		elif 'a' in opt:
				fin = fits.open('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname)
				fdata = fin[1].data
				fcols = fin[1].columns.names
				fin.close()
				tin = Table(data=fdata, names=fcols)
				tin.add_row(flux_flux_err)

				tin.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname, format='fits', overwrite=True)

	def AGN_output(self,fname,Lx,Nh,Lone,opt):
		
		
		data_out = np.asarray([self.ID,self.z,Lx,Nh,self.Lbol,Lone])
		cols_out = np.asarray(['ID','z','Lx','Nh','Lbol','Lone'])

		t = Table(data=data_out, names=cols_out)

		if 'w' in opt:
			try:
				fin = fits.open('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname)
				fdata = fin[1].data
				fcols = fin[1].columns.names
				tin = Table(data=fdata,names=fcols)
				tin.add_row(data_out)

				tin.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fits',overwrite=True)

			except FileNotFoundError:
				t.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fits',overwrite=True)


		elif 'a' in opt:
				fin = fits.open('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname)
				fdata = fin[1].data
				fcols = fin[1].columns.names
				tin = Table(data=fdata, names=fcols)
				tin.add_row(data_out)

				tin.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname, format='fits', overwrite=True)


	def upper_limits(self):
		upper_lims = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')*(1E-23)*(1E-6)/3 #1σ upper limits in cgs
		regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')
		upper_w = self.rest_w_microns[regime == 'FIR']

		upper_lims = upper_lims[regime == 'FIR']

		upper_lims *= self.rest_freq[regime == 'FIR']
		upper_lims_nuLnu = self.Flux_to_Lum(upper_lims,self.z)

		upper_lim_check = []
		upper_lim_out = []
		upper_w_out = []
		for i in range(len(self.nuL_nu[regime == 'FIR'])):
			if np.isnan(self.nuL_nu[regime == 'FIR'][i]):
				upper_lim_out.append(upper_lims_nuLnu[i]/self.Find_nuFnu(1.0))
				upper_w_out.append(upper_w[i])
				upper_lim_check.append(1.)
			else:
				upper_lim_out.append(np.nan)
				upper_w_out.append(upper_w[i])
				upper_lim_check.append(0.)

		nan_array = np.array([np.nan,np.nan])
		if len(upper_w_out) < 7:
			upper_w_out = np.append(nan_array,np.asarray(upper_w_out))
			upper_lim_out = np.append(nan_array,np.asarray(upper_lim_out))

		return np.asarray(upper_w_out),np.asarray(upper_lim_out)


	def write_xcigale_input(self,name):
		# x_cigale_filters = Filters('filter_list.dat').pull_filter(self.good_filter_name,'xcigale name')
		# x_cigale_filters_all = Filters('filter_list.dat').pull_filter(self.filter_name,'xcigale name')
		# regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')
		upper_lims = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')/1E3 # upper limits in mjy

		header = np.asarray(['# id','redshift'])

		# for i in range(len(x_cigale_filters_all)):
		# 	header = np.append(header,x_cigale_filters_all[i])
		# 	header = np.append(header,x_cigale_filters_all[i]+'_err')
		# print(self.obs_f[0],self.obs_f[1])
		# print(self.abs_corr[0],self.abs_corr[2])
		# print((self.obs_f[0]/1E3)/self.abs_corr[0],(self.obs_f[1]/1E3)/self.abs_corr[2])
		# print(self.obs_f_err[0],self.obs_f_err[1])
		
		# data = np.asarray([self.ID,self.z,(self.obs_f[0]/1E3)/self.abs_corr[1],self.obs_f_err[0]/1E3,(self.obs_f[1]/1E3)/self.abs_corr[1],self.obs_f_err[1]/1E3])
		# data = np.asarray([str(self.ID),self.z,self.obs_f[0]/1E3,self.obs_f_err[0]/1E3,self.obs_f[1]/1E3,self.obs_f_err[1]/1E3])
		data = np.asarray([str(self.ID),self.z])
		
		# for i in range(len(self.obs_f)-2):
		# 	if self.obs_f[i+2][regime[i+2] == 'FIR'] < 0:
		# 		self.obs_f[i+2] = upper_lims[i+2]
		# 		self.obs_f_err[i+2] = -9000.
		# 		header = np.append(header,x_cigale_filters_all[i+2])
		# 		header = np.append(header,x_cigale_filters_all[i+2]+'_err')

		# 	if self.obs_f_err[i+2]/self.obs_f[i+2] <= 0.2:
		# 		data = np.append(data,self.obs_f[i+2]/1E3)
		# 		data = np.append(data,self.obs_f_err[i+2]/1E3)
		# 	# elif self.obs_f_err[i+2]/self.obs_f[i+2] > 0.2:
				
		# 	else:
		# 		data = np.append(data,-9999.)
		# 		data = np.append(data,-9999.)

		for i in range(len(self.obs_f)):
			if self.obs_f[i] > 0:
				data = np.append(data,self.obs_f[i]/1E3)
				data = np.append(data,self.obs_f_err[i]/1E3)
			elif self.obs_f[i] <= 1E-20:
				data = np.append(data,upper_lims[i])
				data = np.append(data,-9000.)
			elif np.isnan(self.obs_f[i]) == True:
				data = np.append(data,-9999.)
				data = np.append(data,-9999.)
		# print(self.obs_f)	
		# print(data)
		# print(len(self.obs_f))
		# print('Data:',len(data))
		# print(data)

		# bad_data = np.where(np.logical_or(data <= 0, np.isnan(data) == True))[0]
		# print(data[bad_data])
		# data[bad_data] = -9999.

		with open(f'../xcigale/cigale-new_xcig/pcigale/data/{name}','ab') as f:
			f.write(b'\n')
			np.savetxt(f,data,fmt='%s',delimiter='    ',newline=' ')

		# np.savetxt('../xcigale/cigale-xray/pcigale/data/XCIGALE_COSMOS_test5.mag',(header,data),fmt='%s',delimiter='    ',newline=' ')


		# with open('../xcigale/cigale-xray/pcigale/data/XCIGALE_COSMOS_test5.mag','ab') as f:
			
			# np.savetxt(f,data,fmt='%s',delimiter='    ',newline='\n')

def Lum_to_Flux(L,z):
	cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

	dl = cosmo.luminosity_distance(z).value # luminosity distance in Mpc
	dl_cgs = dl*(3.086E24)

	# convert flux to luminoisty
	F = (L)/(4*np.pi*dl_cgs**2)
	return F

def Flux_to_Lum(F,z):
	cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

	dl = cosmo.luminosity_distance(z).value # luminosity distance in Mpc
	dl_cgs = dl*(3.086E24)

	# convert flux to luminoisty
	L = F*(4*np.pi*dl_cgs**2)
	return L



def Median_Lbol(rest_w,nuLnu,z,F1,F100):
	x = rest_w*1E-4
	# y = self.nuF_nu
	y = nuLnu*F1
	F100 = F100*F1

	# x = np.append(x,0.01)
	# y = np.append(y,F100)
	# sort = x.argsort()
	# x,y = x[sort], y[sort]

	Lbol_interp = interpolate.interp1d(np.log10(x[~np.isnan(y)]),np.log10(y[~np.isnan(y)]),kind='linear',fill_value='extrapolate')

	x_interp = np.linspace(np.log10(min(x)),np.log10(max(x)))
	y_interp = 10**Lbol_interp(x_interp)


	x_interp_FIR = np.linspace(np.log10(0.003),np.log10(0.01))
	y_interp_FIR = 10**Lbol_interp(x_interp_FIR)

	plt.plot(10**x_interp,y_interp)
	plt.plot(10**x_interp_FIR,y_interp_FIR)
	plt.plot(x,y,'x',color='k')
	plt.plot(0.01,F100,'x',color='r')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

	x_interp, y_interp  = x_interp[::-1], y_interp[::-1]
	x_interp_FIR, y_interp_FIR  = x_interp_FIR[::-1], y_interp_FIR[::-1]

	freq = 3E10/10**x_interp
	freq_FIR = 3E10/10**x_interp_FIR

	y_interp = y_interp/freq
	y_interp_FIR = y_interp_FIR/freq_FIR

	f_bol = integrate.trapz(y_interp,freq)
	f_bol_fir = integrate.trapz(y_interp_FIR,freq_FIR)

	cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
	dl = cosmo.luminosity_distance(z).value
	dl_cgs = dl*3.086E24

	# L_bol = f_bol*4*np.pi*dl_cgs**2
	# L_bol_FIR = f_bol_fir*4*np.pi*dl_cgs**2

	L_bol = f_bol
	L_bol_FIR = f_bol_fir

	if max(x) <= 0.01:
		# print('yes')
		L_bol += L_bol_FIR
	# else:
		# print('no')

	return L_bol

def Find_Lbol(x,y,z):

		x = 10**x
		print(x)
		print(y)
		print(z)

		Lbol_interp = interpolate.interp1d(np.log10(x[~np.isnan(y)]),np.log10(y[~np.isnan(y)]),kind='linear',fill_value='extrapolate')

		x_interp = np.linspace(np.log10(min(x)),np.log10(max(x)))
		y_interp = 10**Lbol_interp(x_interp)


		# x_interp_FIR = np.linspace(np.log10(0.003),np.log10(0.01))
		# y_interp_FIR = 10**Lbol_interp(x_interp_FIR)

		plt.plot(10**x_interp,y_interp)
		# plt.plot(10**x_interp_FIR,y_interp_FIR)
		plt.plot(x,y,'x',color='k')
		plt.xscale('log')
		plt.yscale('log')
		plt.show()

		x_interp, y_interp  = x_interp[::-1], y_interp[::-1]
		# x_interp_FIR, y_interp_FIR  = x_interp_FIR[::-1], y_interp_FIR[::-1]

		freq = 3E10/10**x_interp
		# freq_FIR = 3E10/10**x_interp_FIR

		y_interp = y_interp/freq
		# y_interp_FIR = y_interp_FIR/freq_FIR

		f_bol = integrate.trapz(y_interp,freq)
		# f_bol_fir = integrate.trapz(y_interp_FIR,freq_FIR)

		cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
		dl = cosmo.luminosity_distance(z).value
		dl_cgs = dl*3.086E24

		# L_bol = f_bol*4*np.pi*dl_cgs**2
		# L_bol_FIR = f_bol_fir*4*np.pi*dl_cgs**2

		L_bol = f_bol
		# L_bol_FIR = f_bol_fir

		# if max(x) <= 0.01:
			# print('yes')
			# L_bol += L_bol_FIR
		# else:
			# print('no')
		# L_bol = L_bol
		# L_bol_FIR = L_bol_FIR

		print('GOALS Lbol:',L_bol)

		return L_bol
