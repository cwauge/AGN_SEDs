'''
Connor Auge - Created Arpil 28, 2020
Photometric filter database to be utilized in constructing SEDs
'''

import numpy as np 
from astropy.io import ascii
from collections import OrderedDict
from operator import itemgetter


class Filters():
	''' A class defining a photometric filter with a given central wavelength'''

	def __init__(self,file):
		self.file = file
		inf = ascii.read(self.file)
		self.name = np.asarray(inf['Name'])
		self.name_err = np.asarray(inf['Name_err'])
		self.xcigale_name = np.asarray(inf['XCIGALE_Name'])
		self.cent_wave = np.asarray(inf['central_wavelength'])
		self.upper_lim = np.asarray(inf['upper_lim'])
		self.wave_regime = np.asarray(inf['wavelength_range']) 

		self.name_err_od = OrderedDict(zip(self.name,self.name_err))
		self.xcigale_name_od = OrderedDict(zip(self.name,self.xcigale_name))
		self.wave_od = OrderedDict(zip(self.name,self.cent_wave))
		self.upper_od = OrderedDict(zip(self.name,self.upper_lim))
		self.regime_od = OrderedDict(zip(self.name,self.wave_regime))

	def pull_filter(self,filter_name,out):
		if out == 'error name':
			pull = np.asarray(itemgetter(*filter_name)(self.name_err_od))
		elif out == 'xcigale name':
			pull = np.asarray(itemgetter(*filter_name)(self.xcigale_name_od))
		elif out == 'central wavelength':
			pull = np.asarray(itemgetter(*filter_name)(self.wave_od))
		elif out == 'upper limit':
			pull = np.asarray(itemgetter(*filter_name)(self.upper_od))
		elif out == 'wavelength range':
			pull = np.asarray(itemgetter(*filter_name)(self.regime_od))
		else:
			print('Specify pull_filter output. Options are: error name, xcigale name, central wavelength, or upper limit. Must be entered as strings.')
		return pull

	def add_filter(self,new_filter_name,new_regime,new_cent_w,new_xcigale_name,new_upper_lim):
		name = np.append(self.name,new_filter_name)
		name_err = np.append(self.name_err,new_filter_name+'_err')
		xcigale_name = np.append(self.xcigale_name,new_xcigale_name)
		cent_wave = np.append(self.cent_wave,new_cent_w)
		upper_lim = np.append(self.upper_lim,new_upper_lim)
		wave_regime = np.append(self.wave_regime,new_regime)

		sort = cent_wave.argsort()
		name, name_err, xcigale_name, cent_wave, upper_lim, wave_regime = name[sort],name_err[sort],xcigale_name[sort],cent_wave[sort],upper_lim[sort],wave_regime[sort]

		outf = open(self.file,'w')
		outf.writelines('# Name    Name_err    XCIGALE_Name    central_wavelength    upper_lim    wavelength_range\n')
		for i in range(len(name)):
			outf.writelines('%s    %s    %s    %s    %s    %s\n' % (name[i],name_err[i],xcigale_name[i],cent_wave[i],upper_lim[i],wave_regime[i]))
		outf.close()







# name = np.asarray(['Fx_hard','Fx_full','Fx_soft','FLUX_GALEX_FUV','MAG_FUV','FLUX_GALEX_NUV','MAG_NUV','u_FLUX_APER2','U','B_FLUX_APER2','IB464_FLUX_APER2','G','IA484_FLUX_APER2','IB505_FLUX_APER2','IA527_FLUX_APER2','V_FLUX_APER2','IB574_FLUX_APER2','IA624_FLUX_APER2','R','r_FLUX_APER2','IA679_FLUX_APER2','IB709_FLUX_APER2','NB711_FLUX_APER2','IA738_FLUX_APER2','I','IA767_FLUX_APER2','ip_FLUX_APER2','NB816_FLUX_APER2','IB827_FLUX_APER2','zp_FLUX_APER2','Z','zpp_FLUX_APER2','yHSC_FLUX_APER2','Y_FLUX_APER2','JVHS','J_FLUX_APER2','Hw_FLUX_APER2','HVHS','H_FLUX_APER2','KVHS','Ks_FLUX_APER2','Ksw_FLUX_APER2','W1','CH1_SPIES','CH2_SPIES','SPLASH_1_FLUX','SPLASH_2_FLUX','W2','SPLASH_3_FLUX','SPLASH_4_FLUX','W3','IRAS1','W4','FLUX_24','IRAS2','IRAS3','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_350','FLUX_500','SCUBA1','SCUBA2','VLA1','VLA2','nan'])
# name_err = np.asarray(['Fx_hard_err','Fx_full_err','Fx_soft_err','FLUXERR_GALEX_FUV','MAGERR_FUV','FLUXERR_GALEX_NUV','MAGERR_NUV','u_FLUXERR_APER2','U_ERR','G_ERR','B_FLUXERR_APER2','IB464_FLUXERR_APER2','IA484_FLUXERR_APER2','IB505_FLUXERR_APER2','IA527_FLUXERR_APER2','V_FLUXERR_APER2','IB574_FLUXERR_APER2','IA624_FLUXERR_APER2','R_ERR','r_FLUXERR_APER2','IA679_FLUXERR_APER2','IB709_FLUXERR_APER2','NB711_FLUXERR_APER2','IA738_FLUXERR_APER2','I_ERR','IA767_FLUXERR_APER2','ip_FLUXERR_APER2','NB816_FLUXERR_APER2','IB827_FLUXERR_APER2','zp_FLUXERR_APER2','Z_ERR','zpp_FLUXERR_APER2','yHSC_FLUXERR_APER2','Y_FLUXERR_APER2','JVHS_ERR','J_FLUXERR_APER2','Hw_FLUXERR_APER2','HVHS_ERR','H_FLUXERR_APER2','KVHS_ERR','Ks_FLUXERR_APER2','Ksw_FLUXERR_APER2','W1_ERR','CH1_SPIES_ERR','CH2_SPIES_ERR','SPLASH_1_FLUX_ERR','SPLASH_2_FLUX_ERR','W2_ERR','SPLASH_3_FLUX_ERR','SPLASH_4_FLUX_ERR','W3_ERR','IRAS1_ERR','W4_ERR','FLUXERR_24','IRAS2_ERR','IRAS3_ERR','MIPS2_ERR','FLUXERR_100','FLUXERR_160','FLUXERR_250','FLUXERR_350','FLUXERR_500','SCUBA1_ERR','SCUBA2_ERR','VLA1_ERR','VLA2_ERR','nan_err'])
# x_cigale_name = np.asarray(['xray_boxcar_2to10keV','xray_boxcar_0p5to10keV','xray_boxcar_0p5to2keV','FUV','FUV','NUV','NUV','CFHT_u','u_prime','SUBARU_B','subaru.suprime.IB464','g_prime','subaru.suprime.IB484','subaru.suprime.IB505','subaru.suprime.IB527','V_B90','subaru.suprime.IB574','subaru.suprime.IB624','r_prime','subaru.suprime.r','subaru.suprime.IB679','subaru.suprime.IB709','subaru.suprime.NB711','subaru.suprime.IB738','i_prime','subaru.suprime.IB767','subaru.suprime.i','subaru.suprime.NB816','subaru.suprime.IB827','subaru.suprime.z','z_prime','subaru.suprime.zpp','subaru.hsc.y','vista.vircam.Y','vista.vircam.J','2mass.J','ukirt.H','vista.vircam.H','2mass.H','vista.vircam.Ks','cfht.wircam.Ks','2mass.Ks','WISE1','spitzer.irac.ch1','spitzer.irac.ch2','spitzer.irac.ch1','spitzer.irac.ch2','WISE2','spitzer.irac.ch3','spitzer.irac.ch4','WISE3','IRAS1','WISE4','spitzer.mips.24','IRAS2','IRAS3','MISP2','herschel.pacs.100','spitzer.mips.160','herschel.spire.PSW','herschel.spire.PMW','herschel.spire.PLW','scuba.450','scuba.850','VLA_C','VLA_L','nan'])
# cent_wavelength = np.asarray([2.07,2.36,9.92,1539.78,1539.78,2313.89,2313.89,3823.29,3550,4458.3,4635.13,4800,4849.20,5062.15,5261.13,5477.83,5764.76,6233.09,6240,6288.71,6781.18,7073.63,7119.88,7361.56,7660,7684.89,7683.88,8149.39,8244.53,9036.88,9080,9105.72,9779.93,10214.19,12520,12534.65,16311.41,16430,16453.41,21520,21539.88,21590.44,33680,35560,35020,35634.2,45110.1,46170,57593.4,79594.9,120690,115979.9,221950,236747.51,238775.3,614849.6,708650.0,1036928.77,1697691.33,2536859.83,3557125.92,5191371.41,4500470.0,8632180.0,57094290.0,197784502.0,np.nan])
# upper_lim = np.asarray([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,14.6,np.nan,np.nan,np.nan,71,np.nan,np.nan,np.nan,5E3,10.2E3,8.1E3,10.7E3,15.4E3,np.nan,np.nan,np.nan,np.nan,np.nan])
# regime = np.asarray(['X-ray','X-ray','X-ray','UV','UV','UV','UV','UV','UV','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','Radio','Radio','Radio','Radio','nan'])
# print(len(name),len(name_err),len(x_cigale_name),len(cent_wavelength),len(upper_lim),len(regime))

# outf = open('filter_list.dat','w')
# outf.writelines('# Name    Name_err    XCIGALE_Name    central_wavelength    upper_lim    wavelength_range\n')
# for i in range(len(name)):
# 	outf.writelines('%s    %s    %s    %s    %s    %s\n' % (name[i],name_err[i],x_cigale_name[i],cent_wavelength[i],upper_lim[i],regime[i]))
# outf.close()












name = np.asarray(['Fx_hard','Fx_hard_27','Fx_full','Fx_soft','FLUX_GALEX_FUV','MAG_FUV','FLUX_GALEX_NUV','MAG_NUV','u_FLUX_APER2','U','B_FLUX_APER2','IB464_FLUX_APER2','G','IA484_FLUX_APER2','IB505_FLUX_APER2','IA527_FLUX_APER2','V_FLUX_APER2','IB574_FLUX_APER2','IA624_FLUX_APER2','R','r_FLUX_APER2','IA679_FLUX_APER2','IB709_FLUX_APER2','NB711_FLUX_APER2','IA738_FLUX_APER2','I','IA767_FLUX_APER2','ip_FLUX_APER2','NB816_FLUX_APER2','IB827_FLUX_APER2','zp_FLUX_APER2','Z','zpp_FLUX_APER2','yHSC_FLUX_APER2','Y_FLUX_APER2','JVHS','J_FLUX_APER2','Hw_FLUX_APER2','HVHS','H_FLUX_APER2','KVHS','Ks_FLUX_APER2','Ksw_FLUX_APER2','W1','CH1_SPIES','CH2_SPIES','SPLASH_1_FLUX','SPLASH_2_FLUX','W2','SPLASH_3_FLUX','SPLASH_4_FLUX','W3','IRAS1','W4','FLUX_24','IRAS2','IRAS3','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_250_s82x','FLUX_350','FLUX_350_s82x','FLUX_500','FLUX_500_s82x','SCUBA1','SCUBA2','VLA1','VLA2','nan'])
name_err = np.asarray(['Fx_hard_err','Fx_hard_27_err','Fx_full_err','Fx_soft_err','FLUXERR_GALEX_FUV','MAGERR_FUV','FLUXERR_GALEX_NUV','MAGERR_NUV','u_FLUXERR_APER2','U_ERR','G_ERR','B_FLUXERR_APER2','IB464_FLUXERR_APER2','IA484_FLUXERR_APER2','IB505_FLUXERR_APER2','IA527_FLUXERR_APER2','V_FLUXERR_APER2','IB574_FLUXERR_APER2','IA624_FLUXERR_APER2','R_ERR','r_FLUXERR_APER2','IA679_FLUXERR_APER2','IB709_FLUXERR_APER2','NB711_FLUXERR_APER2','IA738_FLUXERR_APER2','I_ERR','IA767_FLUXERR_APER2','ip_FLUXERR_APER2','NB816_FLUXERR_APER2','IB827_FLUXERR_APER2','zp_FLUXERR_APER2','Z_ERR','zpp_FLUXERR_APER2','yHSC_FLUXERR_APER2','Y_FLUXERR_APER2','JVHS_ERR','J_FLUXERR_APER2','Hw_FLUXERR_APER2','HVHS_ERR','H_FLUXERR_APER2','KVHS_ERR','Ks_FLUXERR_APER2','Ksw_FLUXERR_APER2','W1_ERR','CH1_SPIES_ERR','CH2_SPIES_ERR','SPLASH_1_FLUX_ERR','SPLASH_2_FLUX_ERR','W2_ERR','SPLASH_3_FLUX_ERR','SPLASH_4_FLUX_ERR','W3_ERR','IRAS1_ERR','W4_ERR','FLUXERR_24','IRAS2_ERR','IRAS3_ERR','MIPS2_ERR','FLUXERR_100','FLUXERR_160','FLUXERR_250','FLUXERR_250_s82x','FLUXERR_350','FLUXERR_350_s82x','FLUXERR_500','FLUXERR_500_s82x','SCUBA1_ERR','SCUBA2_ERR','VLA1_ERR','VLA2_ERR','nan_err'])
x_cigale_name = np.asarray(['xray_boxcar_2to10keV','xray_boxcar_2to7keV','xray_boxcar_0p5to10keV','xray_boxcar_0p5to2keV','FUV','FUV','NUV','NUV','CFHT_u','u_prime','SUBARU_B','subaru.suprime.IB464','g_prime','subaru.suprime.IB484','subaru.suprime.IB505','subaru.suprime.IB527','V_B90','subaru.suprime.IB574','subaru.suprime.IB624','r_prime','subaru.suprime.r','subaru.suprime.IB679','subaru.suprime.IB709','subaru.suprime.NB711','subaru.suprime.IB738','i_prime','subaru.suprime.IB767','subaru.suprime.i','subaru.suprime.NB816','subaru.suprime.IB827','subaru.suprime.z','z_prime','subaru.suprime.zpp','subaru.hsc.y','vista.vircam.Y','vista.vircam.J','2mass.J','ukirt.H','vista.vircam.H','2mass.H','vista.vircam.Ks','cfht.wircam.Ks','2mass.Ks','WISE1','spitzer.irac.ch1','spitzer.irac.ch2','spitzer.irac.ch1','spitzer.irac.ch2','WISE2','spitzer.irac.ch3','spitzer.irac.ch4','WISE3','IRAS1','WISE4','spitzer.mips.24','IRAS2','IRAS3','MISP2','herschel.pacs.100','spitzer.mips.160','herschel.spire.PSW','herschel.spire.PSW','herschel.spire.PMW','herschel.spire.PMW','herschel.spire.PLW','herschel.spire.PLW','scuba.450','scuba.850','VLA_C','VLA_L','nan'])
cent_wavelength = np.asarray([2.07,2.76,2.36,9.92,1539.78,1539.78,2313.89,2313.89,3823.29,3550,4458.3,4635.13,4800,4849.20,5062.15,5261.13,5477.83,5764.76,6233.09,6240,6288.71,6781.18,7073.63,7119.88,7361.56,7660,7684.89,7683.88,8149.39,8244.53,9036.88,9080,9105.72,9779.93,10214.19,12520,12534.65,16311.41,16430,16453.41,21520,21539.88,21590.44,33680,35560,35020,35634.2,45110.1,46170,57593.4,79594.9,120690,115979.9,221950,236747.51,238775.3,614849.6,708650.0,1036928.77,1697691.33,2536859.83,2536859.83,3557125.92,3557125.92,5191371.41,5191371.41,4500470.0,8632180.0,57094290.0,197784502.0,np.nan])
upper_lim = np.asarray([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,14.6,650,np.nan,2600,71,np.nan,np.nan,np.nan,5E3,10.2E3,8.1E3,13.0E3,10.7E3,12.9E3,15.4E3,14.8E3,np.nan,np.nan,np.nan,np.nan,np.nan])
regime = np.asarray(['X-ray','X-ray','X-ray','X-ray','UV','UV','UV','UV','UV','UV','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','Optical','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','IR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','FIR','Radio','Radio','Radio','Radio','nan'])
# # print(len(name),len(name_err),len(x_cigale_name),len(cent_wavelength),len(upper_lim),len(regime))

# outf = open('filter_list.dat','w')
# outf.writelines('# Name    Name_err    XCIGALE_Name    central_wavelength    upper_lim    wavelength_range\n')
# for i in range(len(name)):
# 	outf.writelines('%s    %s    %s    %s    %s    %s\n' % (name[i],name_err[i],x_cigale_name[i],cent_wavelength[i],upper_lim[i],regime[i]))
# outf.close()
