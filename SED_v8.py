import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import astropy.constants as const
import argparse
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate
from scipy import integrate
from filters import Filters
from bootstrap_err import BootStrap
import error_prop

def main(ID,z,filter_name,obs_flux,obs_flux_err):
    source = AGN(ID,z,filter_name,obs_flux,obs_flux_err)

class AGN():

    def __init__(self,ID,z,filter_name,obs_flux,obs_flux_err):
        self.ID = ID
        self.filter_name = filter_name
        self.z = z
        self.obs_f = obs_flux
        self.obs_f_err = obs_flux_err
        
        filter_list = ascii.read('filter_list.dat')
        self.f_name = np.asarray(filter_list['Name'])
        self.f_name_err = np.asarray(filter_list['Name_err'])
        self.cent_wavelength = np.asarray(filter_list['central_wavelength'])
        self.upper_lim = np.asarray(filter_list['upper_lim'])

        self.obs_w = Filters('filter_list.dat').pull_filter(self.filter_name,'central wavelength')

    def MakeDict(self,list1,list2):
        out_dict = {list1[i]:list2[i] for i in range(len(list1))}
        return out_dict

    def MakeSED(self,data_replace_filt='None'):
        '''Function to make the SED in the restframe'''
        self.c = const.c.to('cm/s').value #speed of light in cgs

        # unit conversions and rest-frame corrections for wavelength values
        rest_w = self.obs_w/(1+self.z) # bring the wavelength to the rest frame
        self.obs_w_cgs = self.obs_w*1E-8 # observed wavelength from Angstroms to cm
        self.rest_w_cgs = rest_w*1E-8 # rest wavelength from Angstroms to cm
        self.rest_w_microns = rest_w*1E-4 # rest wavelength  from Angstroms to microns
        self.rest_freq = self.c/self.rest_w_cgs # convert rest wavelength to a frequency
        self.obs_freq = self.c/self.obs_w_cgs # convert obs wavelength to a frequency

        # unit converstion and quality check for flux values
        self.flux_jy = self.obs_f*1E-6 # convert the flux values from microJy to Jy
        self.flux_jy_err = self.obs_f_err*1E-6 # convert the flux errors from microJyto Jy
        self.flux_jy[self.flux_jy <= 0] = np.nan # replace negative or zero flux values with nan
        self.flux_jy_err[self.flux_jy_err <= 0] = np.nan # replace negative or zero error values with nan
        self.flux_jy[np.isnan(self.flux_jy_err)] = np.nan # replace flux values with no errors with nan
        self.flux_jy[self.flux_jy_err/self.flux_jy >= 0.5] = np.nan # Remove flux values with frac error > 50%
        self.flux_jy_err[np.isnan(self.flux_jy)] = np.nan
        if data_replace_filt != 'None':
            for i in range(len(data_replace_filt)):
                value_replace = self.data_replace([data_replace_filt[i]])
                self.flux_jy[self.filter_name == data_replace_filt[i]] =  value_replace

        # convert flux from frequency space to wavelength
        self.Fnu = self.flux_jy*1E-23 # convert flux from Jy to cgs: erg s^-1 cm^-2 Hz^-1
        self.Fnu_err = self.flux_jy_err*1E-23
        self.Flambda = self.Fnu*(self.c/self.obs_w_cgs**2) 
        self.Flambda_err = self.Fnu_err*(self.c/self.obs_w_cgs**2)

        self.lambdaF_lambda = self.obs_w_cgs*self.Flambda # convert units to erg s^-1 cm^-2
        self.lambdaF_lambda_err = self.obs_w_cgs*self.Flambda
        self.lambdaL_lambda = self.Flux_to_Lum(self.lambdaF_lambda,self.z) # convert flux to luminosity [erg s^-1]
        self.lambdaL_lambda_err = self.Flux_to_Lum(self.lambdaF_lambda_err,self.z)
        self.nuF_nu = self.obs_freq*self.Fnu
        self.nuF_nu_err = self.obs_freq*self.Fnu
        self.nuL_nu = self.Flux_to_Lum(self.nuF_nu,self.z)
        self.nuL_nu_err = self.Flux_to_Lum(self.nuF_nu_err,self.z)

        # Remove data points that do not have a valid y value and interpolate
        # x = np.log10(self.rest_w_microns[~np.isnan(self.lambdaL_lambda)])
        # y = np.log10(self.lambdaL_lambda[~np.isnan(self.lambdaL_lambda)])
        x = np.log10(self.rest_w_microns[~np.isnan(self.nuL_nu)])
        y = np.log10(self.nuL_nu[~np.isnan(self.nuL_nu)])
        yerr = error_prop.log_err(self.nuL_nu[~np.isnan(self.nuL_nu)],self.nuL_nu_err[~np.isnan(self.nuL_nu)])
        self.f_interp = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate') 
        self.boot = BootStrap(x,y,None,yerr,1000)

    def Lum_filter(self,nfilter):
        outL = self.lambdaL_lambda[self.filter_name == nfilter]
        return outL

    def Int_SED(self,xmin=1E-1,xmax=1E1):
        '''Function to determine the interpolated SED''' 
        x_out = np.arange(xmin,xmax,0.05)
        y_out = self.f_interp(np.log10(x_out))
        return x_out, y_out

    def Find_value(self,wave,limit_factor=2,boot=False):
        '''
        Function to find the value of the SED at a given wavelength (wave in microns)
        limit_factor is a scale factor to look for nearby data in Check_nearest function. 
        If no nearby data is found, then no interpolated value is returned. 
        Default factor is twice the wavelength
        '''
        limit = wave*limit_factor
        if wave > 12:
            # wfir, ffir, value = self.Int_SED_FIR(Find_value=wave,discreet=True,boot=boot)
            value = self.L_FIR_value_out
            if boot:
                # wfir, ffir, value, value_boot = self.Int_SED_FIR(Find_value=wave,discreet=True,boot=boot)
                value = self.L_FIR_value_out_boot

        else:
            # if self.Check_nearest(wave,limit):
                try:
                    value = 10**self.f_interp(np.log10(wave)) # Use interpolation function defined in self.MakeSED() to find value
                    if boot:
                        value_boot = self.boot.iterate_interp(wave,1000,log=True)
                except AttributeError: # Return warning and NaN value if interpolation fails
                    value = np.nan
                    # print(f'Object {self.ID} does not have enought valid flux values to find Luminosity at {wave} through SED interpolation.')
            # else:
                # value = np.nan
                # print(f'Object {self.ID} does not ha ve enought valid flux values to find Luminosity at {wave} through SED interpolation.')
        if boot:
            return value, value_boot
        else:
            return value

    def Find_nearest(self,wave):
        '''Function to find the nearest data points to specified wavelength'''
        wave_short, wave_long = [], []
        check_wave = self.rest_w_microns[~np.isnan(self.nuL_nu)]
        for i, j in enumerate(check_wave):
            if j < wave:
                continue
            elif j > wave and check_wave[i-1] < wave:
                wave_short.append(check_wave[i-1])
                wave_long.append(check_wave[i])
            elif j == wave:
                wave_short.append(j)
                wave_long.append(j)
            else:
                continue
        if len(wave_short) == 0:
            wave_short.append(np.nan)
            wave_long.append(np.nan)

        return np.asarray(wave_short), np.asarray(wave_long)
    
    def Check_nearest(self,wave,limit):
        '''
        Function to check if the distance of to the nearest data points greater than and less than wave are within limit
        Limit is set by a scale factor of wave in Find_value function (default factor is 2)
        '''
        ws, wl = self.Find_nearest(wave)
        if np.isnan(ws):
            out = False
        else:
            distance = wl - ws
            if distance < limit:
                out = True
            else:
                out = False
        return out
    
    def Find_slope(self,wi,wf):
        '''Function to find the slope of the SED between to given wavelength values, wi, wf'''
        fi = self.Find_value(wi)/self.Find_value(1.0) # Use Find_value funtion to find SED value at wi
        ff = self.Find_value(wf)/self.Find_value(1.0) # Use Find_value funtion to find SED value at wf
        slope = (np.log10(ff) - np.log10(fi))/(np.log10(wf) - np.log10(wi)) # Calculate the slope in log space

        return slope

    def SED_shape(self,Uv1=0.2,Uv2=1.0,Mir11=1.0,Mir12=6.,Mir21=6.0,Mir22=10):
        '''Find the "shape" of the SED as defined by 1 of 5 predefined bins'''
        uv_slope = self.Find_slope(Uv1, Uv2)  # Find the slope of the SED in the UV range
        mir_slope1 = self.Find_slope(Mir11, Mir12) # Find the slope of the SED in the NIR-MIR range
        mir_slope2 = self.Find_slope(Mir21, Mir22) # Find the slope of the SED in the MIR range

        # Pre-defined conditions. Check slope values to determine SED shape bin and return bin
        if (uv_slope < -0.3) & (mir_slope1 >= -0.4):
            bin = 1
        elif (uv_slope >= -0.3) & (uv_slope <= 0.21) & (mir_slope1 >= -0.42):
            bin = 2 
        elif (uv_slope > 0.21) & (mir_slope1 >= -0.4):
            bin = 3
        elif (uv_slope >= 0.2) & (mir_slope1 < -0.4) & (mir_slope2 > 0.0):
            bin = 4
        elif (uv_slope > 0.21) & (mir_slope1 < -0.4) & (mir_slope2 <= 0.0):
            bin = 5
        else:
            bin = -99.
        self.shape = bin
        
        return bin
    
    def pull_plot_info(self,norm_w,norm=True):
        '''
        Function to return the arrays/values necessary for SED plotting
        Input is wavelength to use for SED normalization 
        Default function will normalize by norm_w
        can set norm=False to return non-normalized SED
        '''
        if norm:
            norm_f = self.Find_value(norm_w) # Get normalization value
            # norm_lambdaL_lambda = self.lambdaL_lambda/norm_f # Normalize SED y values
            norm_lambdaL_lambda = self.nuL_nu/norm_f
            for i in range(len(norm_lambdaL_lambda)):
                if (self.rest_w_microns[i] > 0.25) & (self.rest_w_microns[i] < 1) & (norm_lambdaL_lambda[i] < 1E-2):
                    norm_lambdaL_lambda[i] = np.nan
                elif (self.rest_w_microns[i] > 0.5) & (self.rest_w_microns[i] < 1) & (norm_lambdaL_lambda[i] < 0.1):
                    norm_lambdaL_lambda[i] = np.nan
        else: 
            # norm_lambdaL_lambda = self.lambdaL_lambda
            norm_lambdaL_lambda = self.nuL_nu

        try:
            return self.ID, self.z, self.rest_w_microns, norm_lambdaL_lambda, self.flux_jy_err/self.flux_jy, self.upper_check
        except AttributeError:
            return self.ID, self.z, self.rest_w_microns, norm_lambdaL_lambda, self.flux_jy_err/self.flux_jy

    def calc_Lbol(self,xin=None,yin=None,xmax=None,sub=False,Lscale=None,Lnorm=None,temp_x=None,temp_y=None,Data=True):
        if xin is None:
            x = self.rest_w_microns*1E-4
            y = self.nuL_nu
        else:
            x = xin*1E-4
            y = yin

        if Data:
            x = np.append(x, self.FIR_wave*1E-4)
            y = np.append(y, self.FIR_lambdaL_lambda)
        sort = x.argsort()
        x,y = x[sort], y[sort]

        if xmax != None:
            y = y[x < xmax*1E-4]
            x = x[x < xmax*1E-4]    

        if sub:
            y_sub = self.Find_Lbol_temp_sub(Lscale,Lnorm,temp_x,temp_y,x,y)
            y = y_sub
            
        x = np.log10(x[~np.isnan(y)])
        y = np.log10(y[~np.isnan(y)])

        self.Lbol_interp = interpolate.interp1d(x[y>0],y[y>0],kind='linear',fill_value='extrapolate')

        x_interp = np.linspace(min(x),max(x))
        y_interp = 10**self.Lbol_interp(x_interp)

        # if ~sub:
        #     plt.figure(figsize=(8,8))
        #     plt.title(self.ID)
        #     plt.plot(10**x_interp*1E4,y_interp)
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.show()

        x_interp, y_interp = x_interp[::-1], y_interp[::-1]

        freq = self.c/10**x_interp
        y_interp = y_interp/freq

        Lbol = integrate.trapz(y_interp,freq)
        # if sub:
        #     return Lbol_sub
        # else:
        return Lbol

    def Find_Lbol(self, xmax=None, xin=None, yin=None, sub=False, Lscale=None, Lnorm=None, temp_x=None, temp_y=None, Data=True):
        if xmax == None:
            if sub:
                self.Lbol_sub = self.calc_Lbol(Lscale=Lscale,Lnorm=Lnorm,temp_x=temp_x,temp_y=temp_y,sub=True)
                return self.Lbol_sub
            else:
                self.Lbol = self.calc_Lbol()
                return self.Lbol
        else:
            if sub:
                self.Lbol_sub = self.calc_Lbol(Lscale=Lscale,Lnorm=Lnorm,temp_x=temp_x,temp_y=temp_y,sub=True)
                return self.Lbol_sub
            else:
                self.Lbol = self.calc_Lbol(xmax=xmax)
                return self.Lbol

    def Find_Lbol_temp_sub(self,scale_L,Lnorm,temp_x,temp_y,x,y,xmax=None, sed=False, redshift=False):
        try:
            Lone_temp = temp_y[temp_x == 1.0050][0]
        except IndexError:
            Lone_temp = temp_y[np.round(temp_x, 5) == 1.0][0]
        if self.z <= 0.6:
            if Lnorm < scale_L[0]:
                scale = Lnorm/Lone_temp
            else:
                scale = scale_L[0]/Lone_temp
        elif (self.z > 0.6) & (self.z < 0.9):
            if Lnorm < scale_L[1]:
                scale = Lnorm/Lone_temp
            else:
                scale = scale_L[1]/Lone_temp
        else:
            if Lnorm < scale_L[2]:
                scale = Lnorm/Lone_temp
            else:
                scale = scale_L[2]/Lone_temp
        
            # if redshift:
            #     scale = 1
            #     flux_y = self.Lum_to_Flux(temp_y,0.0009)
            #     temp_y = self.Flux_to_Lum(flux_y, self.z)
            # else:
                
                # scale = Lnorm/Lone_temp
        scale_y = temp_y*scale

        temp_interp = interpolate.interp1d(np.log10(temp_x*1E-4), np.log10(scale_y),kind='linear',fill_value='extrapolate')
        y_interp = 10**temp_interp(np.log10(x))
        # Lbol_temp = self.calc_Lbol(xin=self.rest_w_microns[y_interp > 0],yin=y_interp[y_interp > 0],Data=False)
        # return self.Lbol - Lbol_temp

        y_sub = y - y_interp
        return y_sub
        # return self.calc_Lbol(xin=x[y_sub > 0], yin=y_sub[y_sub > 0])
        # if sed:
        #     return self.Find_Lbol(xin=x[y_sub > 0], yin=y_sub[y_sub > 0]), temp_x, scale_y, self.rest_w_microns, y_sub
        # # if xmax is None:
        #     # return self.Find_Lbol(xin=self.rest_w_microns[y_sub > 0], yin=y_sub[y_sub > 0])
        # else:
        #     return self.Find_Lbol(xin=x[y_sub > 0], yin=y_sub[y_sub > 0],xmax=xmax)

    def find_Lum_range(self,xmin,xmax):
        x = np.log10(self.rest_w_cgs[~np.isnan(self.lambdaL_lambda)])
        # y = np.log10(self.lambdaL_lambda[~np.isnan(self.lambdaL_lambda)])
        y = np.log10(self.nuL_nu[~np.isnan(self.nuL_nu)])

        # Lbol_interp = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate')

        x_interp = np.linspace(np.log10(xmin*1E-4),np.log10(xmax*1E-4))
        y_interp = 10**self.Lbol_interp(x_interp)

        x_interp, y_interp = x_interp[::-1], y_interp[::-1]
        freq = self.c/10**x_interp
        y_interp = y_interp/freq
        L_region = integrate.trapz(y_interp,freq)

        return L_region

    def stack_check(self):
        F250 = self.flux_jy[self.filter_name == 'FLUX_250_s82x']
        L22 = self.Lum_filter('W4')

        bin1 = (np.isnan(F250)) & (self.z <= 0.45) & (np.log10(L22[0]) <= 44.1)
        bin2 = (np.isnan(F250)) & (self.z <= 0.45) & (np.logical_and(np.log10(L22[0]) <= 47.0, np.log10(L22[0]) > 44.1))
        bin3 = (np.isnan(F250)) & (np.logical_and(self.z > 0.45, self.z <= 0.7)) & (np.log10(L22[0]) <= 44.7)
        bin4 = (np.isnan(F250)) & (np.logical_and(self.z > 0.45, self.z <= 0.7)) & (np.logical_and(np.log10(L22[0]) <= 47.0, np.log10(L22[0]) > 44.7))
        bin5 = (np.isnan(F250)) & (np.logical_and(self.z > 0.7, self.z <= 1.25)) & (np.log10(L22[0]) <= 45.3)
        bin6 = (np.isnan(F250)) & (np.logical_and(self.z > 0.7, self.z <= 1.25)) & (np.logical_and(np.log10(L22[0]) <= 47.0, np.log10(L22[0]) > 45.3))

        bin_out = np.asarray([bin1,bin2,bin3,bin4,bin5,bin6])
        loc = np.where(bin_out)[0]

        if len(loc) > 0:
            return loc[0]
        else:
            return 7
 
    def flux_ratio_lower(self):
        flux_1 = self.flux_jy[self.filter_name == 'JVHS']
        flux_100 = flux_1*15
        w_100 = 100*1E-4
        freq_100 = self.c/w_100
        fnu_100 = flux_100*1E-23
        nufnu_100 = fnu_100*freq_100

        nuLnu_100 = self.Flux_to_Lum(nufnu_100,self.z)
        return nuLnu_100

    def stack_FIR_set(self,fname):
        flux_upper = Filters('filter_list.dat').pull_filter(fname,'upper limit')*1E-6 # 3 sigma upper limits from the Filters read file

        stack_bin1_ID = np.asarray([2387,   2667,   2700,   2731,   2803,   2832,   2846,   2850,   2868,   2873,
                            2960,   3015,   3029,   3092,   3131,   3171,   3318,   3335,   3388,   3398,
                            3485,   3504,   3739,   3783,   3840,   3851,   3854,   3936,   3939,   3976,
                            4007,   4034,   4053,   4214,   4222,   4276,   4290,   4295,   4334,   4387,
                            4409,   4414,   4592,   4596,   4602,   4739,   4747,   4898,   4964,   5028,
                            5062,   5089,   5135,   5143,   5151,   5172,    417,    434,    492,    514,
                            520,    521,  57494,  89316, 129876, 129885])
        stack_bin2_ID = np.asarray([2360,   2463,   2471,   2525,   2563,   2598,   2635,   2693,   2728,   2782,
                            2811,   2831,   2840,   2906,   3053,   3241,   3246,   3259,   3264,   3327,
                            3427,   3488,   3540,   3547,   3626,   3628,   3647,   3708,   3763,   3810,
                            3831,   3846,   3861,   3872,   3884,   3909,   3966,   3979,   3982,   4010,
                            4028,   4031,   4051,   4073,   4087,   4139,   4159,   4264,   4272,   4407,
                            4418,   4422,   4424,   4437,   4456,   4512,   4696,   4758,   4791,   4838,
                            4867,   5031,   5087,    405,    425,  57498, 129884, 129887])
        stack_bin3_ID = np.asarray([2363,   2388,   2420,   2442,   2446,   2482,   2522,   2536,   2673,   2675,
                            2702,   2711,   2753,   2845,   2878,   2886,   2925,   2935,   2940,   2948,
                            2949,   3037,   3106,   3116,   3179,   3232,   3291,   3304,   3305,   3312,
                            3339,   3354,   3408,   3868,   3912,   3921,   3929,   3934,   3949,   3975,
                            3983,   4019,   4021,   4029,   4060,   4158,   4174,   4273,   4278,   4287,
                            4306,   4321,   4442,   4467,   4510,   4557,   4558,   4591,   4624,   4630,
                            4766,   4781,   4799,   4816,   4833,   4836,   4845,   4853,   4877,   4881,
                            4902,   4913,   4939,   4957,   4991,   5005,   5025,   5064,   5068,   5078,
                            5213,    411,    418,    488,    491,  15292,  15296,  15306,  42255,  50025,
                            107987, 129802, 180997])
        stack_bin4_ID = np.asarray([2379,   2407,   2413,   2469,   2524,   2587,   2741,   2871,   2971,   3005,
                            3085,   3147,   3194,   3209,   3211,   3219,   3316,   3555,   3557,   3603,
                            3636,   3652,   3654,   3719,   3808,   4001,   4054,   4096,   4134,   4136,
                            4194,   4220,   4235,   4292,   4329,   4398,   4495,   4544,   4625,   4626,
                            4645,   4666,   4771,   4920,   5043,   5076,   5079,   5093,   5094,   5163,
                            399,    505,  15297,  50021, 129821])
        stack_bin5_ID = np.asarray([2458,   2474,   2521,   2542,   2555,   2600,   2622,   2627,   2686,   2691,
                            2788,   2793,   2847,   2893,   2895,   2901,   2974,   2988,   3050,   3059,
                            3185,   3258,   3274,   3281,   3306,   3322,   3361,   3381,   3382,   3410,
                            3482,   3492,   3511,   3608,   3627,   3705,   3711,   3744,   3761,   3794,
                            3816,   3823,   3835,   3837,   3853,   3859,   3873,   3880,   3908,   3914,
                            3937,   4020,   4074,   4112,   4118,   4122,   4127,   4196,   4211,   4212,
                            4217,   4231,   4258,   4260,   4267,   4284,   4303,   4342,   4365,   4395,
                            4420,   4441,   4449,   4488,   4585,   4692,   4707,   4716,   4728,   4778,
                            4807,   4810,   4811,   4824,   4825,   4831,   4832,   4870,   4884,   4918,
                            4934,   4940,   4982,   5007,   5012,   5041,   5100,   5139,   5148,   5155,
                            489,  89309, 105728, 129814, 180999])
        stack_bin6_ID = np.asarray([2368,  2476,  2570,  2606,  2660,  2707,  2708,  2808,  2909,  3027,  3212,  3228,
                            3268,  3376,  3487,  3544,  3610,  3633,  3660,  3709,  3726,  3862,  3865,  3915,
                            4111,  4225,  4421,  4446,  4531,  4598,  4615,  4676,  4678,  4784,  4793,  4850,
                            4851,  4915,  5106,   503, 50029])
        stack_bin7_ID = np.asarray([2445,   2516,   2533,   2637,   2654,   2706,   2729,   2829,   2936,   3070,
                            3076,   3099,   3168,   3247,   3249,   3517,   3552,   3674,   3692,   3766,
                            3768,   3800,   3803,   3822,   3836,   3946,   3953,   3965,   3967,   4059,
                            4147,   4241,   4304,   4448,   4476,   4504,   4704,   4705,   4735,   4865,
                            4998,   5101,   5158,   5162,   5211,    493,  89310, 107991, 129819])
        stack_bin8_ID = np.asarray([2685,   2748,   2774,   2795,   2931,   2964,   3129,   3201,   3296,   3451,
                            3531,   4349,   4435,   4453,   4490,   4857,   4872,   5136,   5142,   5178,
                            5185, 105703])
        if self.ID in stack_bin1_ID:
            F250_upper_out = 7.98/1000
            # print('yes 1')
        elif self.ID in stack_bin2_ID:
            F250_upper_out = 10.17/1000
            # print('yes 2')
        elif self.ID in stack_bin3_ID:
            F250_upper_out = 6.12/1000
            # print('yes 3')        
        elif self.ID in stack_bin4_ID:
            F250_upper_out = 9.14/1000
            # print('yes 4')
        elif self.ID in stack_bin5_ID:
            F250_upper_out = 4.26/1000
            # print('yes 5')
        elif self.ID in stack_bin6_ID:
            F250_upper_out = 14.93/1000
            # print('yes 6')
        elif self.ID in stack_bin7_ID:
            F250_upper_out = 9.92/1000
            # print('yes 7')
        elif self.ID in stack_bin8_ID:
            F250_upper_out = 15.74/1000
            # print('yes 8')
        else:
            F250_upper_out = flux_upper

        return F250_upper_out

    def data_replace(self,filtername):
        flux_upper = Filters('filter_list.dat').pull_filter(filtername,'upper limit')*1E-6 # 3 sigma upper limits from the Filters read file
        # flux_upper /= 3 # 1 sigma upper limit

        ind = np.where(self.filter_name == filtername)[0]
        ind_use = np.array([ind-1,ind,ind+1])
        check = []
        for i in range(len(ind_use)):
            check.append(self.flux_jy[ind_use[i][0]])
        check = np.asarray(check) 
        if any(~np.isnan(check)):
            value = self.flux_jy[self.filter_name == filtername]
        else:
            value = flux_upper

        # if np.isnan(self.flux_jy[self.filter_name == filtername]):
        #     value = flux_upper

        # elif self.flux_jy[self.filter_name == filtername] <= 0:
        #     value = flux_upper
        
        # else:
        #     value = self.flux_jy[self.filter_name == filtername]

        return value

    def FIR_extrap(self,filtername,stack=False):
        '''
        Function to find the FIR SED either based on data or the upper limits
        Input: list of observational filter names over which to construct SED
        Optional input: wavelength to find FIR SED value at (input in microns)
        '''
        self.FIR_filt = filtername
        regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range') # Pull the specified wavelength regime from the Filters read file
        flux_upper = Filters('filter_list.dat').pull_filter(filtername,'upper limit')*1E-6 # 3 sigma upper limits from the Filters read file
        
        if stack:
            f250_upper = self.stack_FIR_set(['FLUX_250_s82x'])
            # print(flux_upper[-3])
            flux_upper[-3] = f250_upper
            # print(flux_upper[-3])


        flux_upper /= 3 # 1 sigma upper limits
        # flux_upper = (flux_upper/3)*2 # 2 sigma upper limits

        self.filt_rest_w_mircons = np.asarray([self.rest_w_microns[self.filter_name == i][0] for i in filtername]) # make array of the rest wavelength in microns for spcified input filters
        filt_rest_w_cgs = np.asarray([self.rest_w_cgs[self.filter_name == i][0] for i in filtername])
        filt_obs_w_cgs = np.asarray([self.obs_w_cgs[self.filter_name == i][0] for i in filtername])
        filt_rest_freq = self.c/filt_rest_w_cgs
        filt_obs_freq = self.c/filt_obs_w_cgs
        self.fir_flux_jy = self.flux_jy[regime == 'FIR']*1E-6 # Flux values for the FIR filters

        flux_jy = []
        for i in range(len(filtername)):
            if self.flux_jy[self.filter_name == filtername[i]] > 0:
                flux_jy.append(self.flux_jy[self.filter_name == filtername[i]][0])
            elif np.isnan(self.flux_jy[self.filter_name == filtername[i]][0]):
                flux_jy.append(flux_upper[i])
            else:
                flux_jy.append(flux_upper[i])
        flux_jy = np.asarray(flux_jy)
        flux_cgs = flux_jy*1E-23

        Flambda = flux_cgs*(self.c/filt_rest_w_cgs**2)
        lambdaF_lambda = filt_obs_w_cgs*Flambda
        lambdaL_lambda = self.Flux_to_Lum(lambdaF_lambda,self.z)

        nuF_nu = flux_cgs*filt_obs_freq
        nuL_nu = self.Flux_to_Lum(nuF_nu,self.z)

        # x = np.log10(filt_rest_w_mircons[~np.isnan(lambdaL_lambda)])
        # y = np.log10(lambdaL_lambda[~np.isnan(lambdaL_lambda)])
        x = np.log10(self.filt_rest_w_mircons[~np.isnan(nuL_nu)])
        y = np.log10(nuL_nu[~np.isnan(nuL_nu)])
        self.upper_f_interp = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        
    def Int_SED_FIR(self,Find_value=np.nan,discreet=False,boot=False):
        xmin = min(self.filt_rest_w_mircons)
        xmax = max(self.filt_rest_w_mircons)
        if discreet:
            xfir_out = np.linspace(xmin, xmax, 13)
        else:
            xfir_out = self.filt_rest_w_mircons

        yfir_upper = 10**self.upper_f_interp(np.log10(xfir_out))
        yfir_data = 10**self.f_interp(np.log10(xfir_out))
        value_upper = 10**self.upper_f_interp(np.log10(Find_value))
        value_data = 10**self.f_interp(np.log10(Find_value))

        if any(~np.isnan(self.fir_flux_jy)):
            yfir_out = yfir_data 
            self.L_FIR_value_out = value_data
            self.upper_check = 0

        elif ~np.isnan(self.fir_flux_jy[0]) and ~np.isnan(self.fir_flux_jy[1]):
            if value_data > value_upper:
                yfir_out = yfir_upper
                yfir_out = yfir_data
                self.L_FIR_value_out = value_upper
                self.upper_check = 0
            else:
                yfir_out = yfir_data
                self.L_FIR_value_out = value_data
                self.upper_check = 0
        else:
            yfir_out = yfir_upper
            self.L_FIR_value_out = value_upper
            self.upper_check = 1

        yfir_out = yfir_upper
        self.L_FIR_value_out = value_upper
        # self.L_FIR_value_out = value_data
        if boot:
            self.L_FIR_value_out_boot = self.boot.iterate_interp(Find_value, 1000, log=True)
        
        self.FIR_lambdaL_lambda, self.FIR_wave = yfir_out, xfir_out
        if np.isnan(Find_value):
            return self.FIR_wave, self.FIR_lambdaL_lambda
        else:
            if boot:
                return self.FIR_wave, self.FIR_lambdaL_lambda, self.L_FIR_value_out, self.L_FIR_value_out_boot
            else:
                return self.FIR_wave, self.FIR_lambdaL_lambda, self.L_FIR_value_out

    def check_SED(self,check_w,check_span=None):
        # Check for an observational data popint within check_span microns of a desired wavelength value (check_w)
        # If check span is not specified use 2 microns
        max_w = check_w + 2.75 # 15 microns
        if check_span is None:
            min_w = check_w - 3 # 2 microns
        else:
            min_w = check_w - check_span

        wave_range = (self.rest_w_microns <= max_w) & (self.rest_w_microns >= min_w)
        check_flux = self.flux_jy[wave_range] # get flux values in specified wavelength range
        
        # Check for good data in wavelength range
        if len(check_flux[np.isnan(check_flux)]) == len(check_flux):
            check_return = 'BAD'
        elif any(check_flux) > 0:
            check_return = 'GOOD'
        elif any(self.flux_jy_err[wave_range]) > 0:
            check_return == 'GOOD'
        else:
            check_return == 'BAD'

        return check_return
    
    def check_SED10(self, check_w, check_span=None):
            # Check for an observational data popint within check_span microns of a desired wavelength value (check_w)
            # If check span is not specified use 2 microns
        max_w = check_w + 15  # 15 microns
        if check_span is None:
            min_w = check_w - 3  # 2 microns
        else:
            min_w = check_w - check_span

        wave_range = (self.rest_w_microns <= max_w) & (
            self.rest_w_microns >= min_w)
        # get flux values in specified wavelength range
        check_flux = self.flux_jy[wave_range]

        # Check for good data in wavelength range
        if len(check_flux[np.isnan(check_flux)]) == len(check_flux):
            check_return = 'BAD'
        elif any(check_flux) > 0:
            check_return = 'GOOD'
        elif any(self.flux_jy_err[wave_range]) > 0:
            check_return == 'GOOD'
        else:
            check_return == 'BAD'

        return check_return

    def IR_colors(self,filter1,filter2,filter3,filter4):
        x1 = self.obs_f[self.filter_name == filter1]
        x2 = self.obs_f[self.filter_name == filter2]
        x3 = self.obs_f[self.filter_name == filter3]
        x4 = self.obs_f[self.filter_name == filter4]

        x = np.log10(x3/x1)
        y = np.log10(x4/x2)

        ir_agn = (x >=  0.08) & (y >= 0.15) & (y >= 1.21*x - 0.27) & (y <= 1.21*x +0.27) & (x4 > x3) & (x3 > x2) & (x2 > x1)

        return x, y, ir_agn[0]

    def Source_output(self,fname,Lx,Nh,opt,Lone):
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

                tin.write('/Useres/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fites',overwrite=True)
                fin.close()

            except FileNotFoundError:
                t.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname,format='fits',overwrite=True)

        elif 'a' in opt:
            fin = fits.open('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname)
            fdata = fin[1].data
            fcols = fin[1].columns.names
            tin = Table(data=fdata, names=fcols)
            tin.add_row(data_out)

            tin.write('/Users/connor_auge/Research/Disertation/catalogs/output/'+fname, format='fits', overwrite=True)
            fin.close()

    def Flux_to_Lum(self,F,z):
        '''Function to convert flux to luminosity'''
        cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

        dl = cosmo.luminosity_distance(z).value # Distance in Mpc
        dl_cgs = dl*(3.0856E24) # Distance from Mpc to cm

        # convert flux to luminosity 
        L = F*4*np.pi*dl_cgs**2

        return L
    
    def Lum_to_Flux(self, L, z):
        '''Function to convert flux to luminosity'''
        cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

        dl = cosmo.luminosity_distance(z).value  # Distance in Mpc
        dl_cgs = dl*(3.0856E24)  # Distance from Mpc to cm

        # convert flux to luminosity
        F = L/(4*np.pi*dl_cgs**2)

        return F

    def mix_loc(self,xrange,yrange):
        fi_x = self.Find_value(xrange[0])
        ff_x = self.Find_value(xrange[1])

        fi_y = self.Find_value(yrange[0])
        ff_y = self.Find_value(yrange[1])

        # print(xrange[0],xrange[1])

        # print(np.log10(fi_x),np.log10(ff_x))

        slope_x = (np.log10(ff_x) - np.log10(fi_x))/(xrange[1] - xrange[0])
        slope_y = (np.log10(ff_y) - np.log10(fi_y))/(yrange[1] - yrange[0])

        x = self.Find_slope(xrange[0],xrange[1])
        y = self.Find_slope(yrange[0],yrange[1])

        # return slope_x, slope_y
        return x*-1, y*-1

    def output_properties(self,field,xid,ra,dec,Lx,Nh):
        '''
        Function to output the properties of each source
        Output:
            AHA field
            X-ray ID
            Phot ID
            RA
            DEC
            Lx
            z
            Nh
            Lbol
            SED Shape
        '''
        cols = ['Field','x_ID','phot_id','RAJ2000','DEJ2000','L0510_c','z_spec','Nh','Lbol','SED_shape']
        data = [field,xid,int(self.ID),ra,dec,round(np.log10(Lx),3),self.z,round(np.log10(Nh),3),round(np.log10(self.Lbol_sub),3),self.shape]
        dtype_out = ['str','str','str','str','str','float','float','float','float','float']
        return cols, data, dtype_out

    def output_phot(self,field,filtername_tot,filtername_field):
        '''
        Function to output the photometry measurements of each source
        Output:
            Phot ID
            Flux measurements and errors for all possible photometry bands 
        '''
        cols = np.asarray(['Field','phot_id'])
        data = np.asarray([field,int(self.ID)])
        tot_flux_err_name = np.asarray([i+'_err' for i in filtername_tot])
        field_flux_err_name = np.asarray([i+'_err' for i in filtername_field])
        col_names = np.empty(filtername_tot.size+tot_flux_err_name.size, dtype=tot_flux_err_name.dtype)
        field_names = np.empty(filtername_field.size+field_flux_err_name.size, dtype=field_flux_err_name.dtype)
        flux_err_data = np.empty(self.flux_jy.size+self.flux_jy_err.size,dtype=self.flux_jy.dtype)

        col_names[0::2] = filtername_tot
        col_names[1::2] = tot_flux_err_name
        # cols_out = np.append(cols,col_names)

        field_names[0::2] = filtername_field
        field_names[1::2] = field_flux_err_name
        # field_names = np.append(cols,field_names)

        flux_err_data[0::2] = self.flux_jy*1E6
        flux_err_data[1::2] = self.flux_jy_err*1E6
        # flux_err_data[0::2] = self.flux_jy
        # flux_err_data[1::2] = self.flux_jy_err
        # data = np.append(data,flux_err_data)

        data_in = np.zeros(col_names.size)
        data_in[data_in == 0] = -99.99

        flux_err_data[np.isnan(flux_err_data)] = -99.99

        data_out = self.match_filters(field_names, col_names, flux_err_data, data_in)
        cols_out = np.append(cols,col_names)
        data_out = np.append(data,data_out)
        dtyps = []
        for i in range(len(cols_out)):
            if cols_out[i] == 'Field':
                dtyps.append('str')
            elif cols_out[i] == 'phot_id':
                dtyps.append('str')
            else:
                dtyps.append('float')
        dtype_out = np.asarray(dtyps)

        return cols_out, data_out, dtype_out

    def match_filters(self,filter1,filter2,data1,data2):
        '''
        Function to fill an array with suplemental data based on matched ID
        filter1,2 - arrays of the IDs used to match 
        data1 - data to fill into data2 array
        data2 - array to be filled with supplemental data
        '''

        for i,j in enumerate(filter1):
            ind = np.where(j == filter2)[0] # where in filter2 does an entry in filter1 match with filter2
            if len(ind) == 1: # check if a match is found. Single match: len(ind) == 1, no match len(ind) == 0
                data2[ind] = data1[i] # replace data2 entry with data1 entry at location of match
            else:
                continue

        return data2

    def write_output_file(self,fname,data_in,cols,dtype_in,opt='w',phot=False):
        '''
        Function to wirte a fits file of the data contained in the output functions
        '''
        cols = np.asarray(cols,dtype=str)
        data_in = np.asarray(data_in,dtype=str)
        t = Table(data=data_in,names=cols)

        if 'w' in opt:
            try:
                # fin = fits.open(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}')
                # fdata = fin[1].data
                # fcols = fin[1].columns.names
                # fin.close()
                fin = pd.read_csv(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}')
                fdata = fin.values
                fcols = fin.columns
                # print(fcols)
                # print(np.shape(fdata),np.shape(fcols),np.shape(data_in))
                tin = Table(data=fdata, names=fcols, dtype=(dtype_in))
                if phot:
                    for i in range(len(fcols)-2):
                        if '_err' in fcols[i+2]:
                        # print(fcols[i+2],fdata[i+2])
                            tin[fcols[i+2]].format = '{:.4g}'
                        else:
                            if fcols[i+2] == 'Fxh':
                                tin[fcols[i+2]].format = '{:.3e}'
                            elif fcols[i+2] == 'Fxs':
                                tin[fcols[i+2]].format = '{:.3e}'
                            else:
                                tin[fcols[i+2]].format = '{:.5g}'
                tin.add_row(data_in)
                # print(tin)

                tin.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='ascii.csv',overwrite=True)

            except FileNotFoundError:
                t.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='ascii.csv',overwrite=True)

        elif 'a' in opt:
            fin = fits.open(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}')
            fdata = fin[1].data
            fcols = fin[1].columns.names
            fin.close()
            tin = Table(data=fdata,names=fcols,dtype=(dtype_in))
            tin.add_row(data_in)

            tin.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='fits',overwrite=True)

    def write_cigale_file2(self,fname,filts,flux_dict,flux_dict_err,int_fx=[np.nan],int_fx_err=np.nan):
        upper_lims = Filters('filter_list.dat').pull_filter(filts,'upper limit')/1E3
        region = Filters('filter_list.dat').pull_filter(filts,'wavelength range')

        header = np.asarray(['# id','redshift'])

        data = np.asarray([str(self.ID),self.z])
        for i in range(len(filts)):
            if filts[i] == 'Fx_hard':
                data = np.append(data, int_fx)
                if ~np.isnan(int_fx_err):
                    data = np.append(data,int_fx_err)
                else:
                    data = np.append(data, flux_dict_err[filts[i]]/1E3)
            elif flux_dict[filts[i]] > 0:
                data = np.append(data, flux_dict[filts[i]]/1E3)
                data = np.append(data, flux_dict_err[filts[i]]/1E3)

            elif flux_dict[filts[i]] <= 1E-20:
                data = np.append(data, upper_lims[i])
                data = np.append(data, upper_lims[i]*(-0.5))

            elif np.isnan(flux_dict[filts[i]]):
                if region[i] == 'FIR':
                    data = np.append(data, upper_lims[i])
                    data = np.append(data, upper_lims[i]*(-0.5))
                else:
                    data = np.append(data, -9999.)
                    data = np.append(data, -9999.)
            else:
                data = np.append(data, -9999.)
                data = np.append(data, -9999.)

        with open(f'../xcigale/cigale-master/pcigale/data/AHA_input_final2/{fname}', 'ab') as f:
            f.write(b'\n')
            np.savetxt(f, data, fmt='%s', delimiter='    ', newline=' ')

    def write_cigale_file(self,fname,int_fx=[np.nan,np.nan],use_int_fx=True,use_upper=False):
        upper_lims = Filters('filter_list.dat').pull_filter(self.filter_name,'upper limit')/1E3
        region = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')
        header = np.asarray(['# id','redshift'])

        f = self.flux_jy*1E6
        ferr = self.flux_jy_err*1E6

        data = np.asarray([str(self.ID),self.z]) 
        for i in range(len(self.filter_name)):
            if self.filter_name[i] == 'nan':
                continue
            elif self.filter_name[i] == 'Fx_hard':
                if use_int_fx:
                    data = np.append(data,int_fx[0])
                    data = np.append(data,ferr[i]/1E3)
                else:
                    data = np.append(data,f[i]/1E3)
                    data = np.append(data,ferr[i]/1E3)
            elif self.filter_name[i] == 'Fx_soft':
                if use_int_fx:
                    data = np.append(data,int_fx[1])
                    data = np.append(data,ferr[i]/1E3)
                else:
                    data = np.append(data,f[i]/1E3)
                    data = np.append(data,ferr[i]/1E3)
            elif f[i] > 0:
                data = np.append(data,f[i]/1E3)
                data = np.append(data,ferr[i]/1E3)
            elif f[i] <= 1E-20:
                data = np.append(data,upper_lims[i])
                data = np.append(data,-9000.)
            elif np.isnan(f[i]) == True:
                if region[i] == 'FIR':
                    if use_upper:
                        data = np.append(data,upper_lims[i])
                        data = np.append(data,-9000)
                    else:
                        data = np.append(data, -9999.)
                        data = np.append(data, -9999.)
                else:
                    data = np.append(data,-9999.)
                    data = np.append(data,-9999.)

        with open(f'../xcigale/data_input/{fname}','ab') as f:
            f.write(b'\n')
            np.savetxt(f,data,fmt='%s',delimiter='    ',newline=' ')

def Flux_to_Lum(F, z, d=np.nan, distance=False):
    '''Function to convert flux to luminosity'''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

    dl = cosmo.luminosity_distance(z).value  # Distance in Mpc
    if distance:
        dl = d/1E6
    dl_cgs = dl*(3.0856E24)  # Distance from Mpc to cm

    # convert flux to luminosity
    L = F*4*np.pi*dl_cgs**2

    return L


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Long Class to generate restframe SED for AGN target. Can output additional info, suchas as lambdaL_lambda as specified wavelength, slope of SED in given range, general SED shape, Lbol, luminosity under specific region of SED, etc.')
    parser.add_argument('ID', help='Source ID', type=str)
    parser.add_argument('--redshift','-z',help='best redshift measurement', type=float)
    parser.add_argument('--filter','-fn',help='array of names of the obs filter used')
    parser.add_argument('--observed_flux','-obs_flux',help='array of observed flux values for each filter')
    parser.add_argument('--observed_flux_error','-obs_flux_err',help='array of observed flux errors for each filter')

    args = parser.parse_args()
    main(args.ID,args.redshift,args.filter,args.observed_flux,args.observed_flux_error)