import numpy as np
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

    def MakeSED(self):
        '''Function to make the SED in the restframe'''
        self.c = const.c.to('cm/s').value #speed of light in cgs

        # unit conversions and rest-frame corrections for wavelength values
        rest_w = self.obs_w/(1+self.z) # bring the wavelength to the rest frame
        self.obs_w_cgs = self.obs_w*1E-8 # observed wavelength from Angstroms to cm
        self.rest_w_cgs = rest_w*1E-8 # rest wavelength from Angstroms to cm
        self.rest_w_microns = rest_w*1E-4 # rest wavelength  from Angstroms to microns
        self.rest_freq = self.c/self.rest_w_cgs # convert rest wavelength to a frequency

        # unit converstion and quality check for flux values
        self.flux_jy = self.obs_f*1E-6 # convert the flux values from microJy to Jy
        self.flux_jy_err = self.obs_f_err*1E-6 # convert the flux errors from microJyto Jy
        self.flux_jy[self.flux_jy <= 0] = np.nan # replace negative or zero flux values with nan
        self.flux_jy_err[self.flux_jy_err <= 0] = np.nan # replace negative or zero error values with nan
        self.flux_jy[np.isnan(self.flux_jy_err)] # replace flux values with no errors with nan
        self.flux_jy[self.flux_jy_err/self.flux_jy >= 0.5] = np.nan # Remove flux values with frac error > 50%

        # convert flux from frequency space to wavelength
        self.Fnu = self.flux_jy*1E-23 # convert flux from Jy to cgs: erg s^-1 cm^-2 Hz^-1
        self.Flambda = self.Fnu*(self.c/self.obs_w_cgs**2) 

        self.lambdaF_lambda = self.obs_w_cgs*self.Flambda # convert units to erg s^-1 cm^-2
        self.lambdaL_lambda = self.Flux_to_Lum(self.lambdaF_lambda,self.z) # convert flux to luminosity [erg s^-1]
        self.nuF_nu = self.rest_freq*self.Fnu
        self.nuL_nu = self.Flux_to_Lum(self.nuF_nu,self.z)

        # Remove data points that do not have a valid y value and interpolate
        # x = np.log10(self.rest_w_microns[~np.isnan(self.lambdaL_lambda)])
        # y = np.log10(self.lambdaL_lambda[~np.isnan(self.lambdaL_lambda)])
        x = np.log10(self.rest_w_microns[~np.isnan(self.nuL_nu)])
        y = np.log10(self.nuL_nu[~np.isnan(self.nuL_nu)])
        self.f_interp = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate') 

    def Int_SED(self,xmin=1E-1,xmax=1E1):
        '''Function to determine the interpolated SED''' 
        x_out = np.arange(xmin,xmax,0.01)
        y_out = self.f_interp(np.log10(x_out))
        return x_out, y_out

    def Find_value(self,wave):
        '''Function to find the value of the SED at a given wavelength'''
        try:
            value = 10**self.f_interp(np.log10(wave)) # Use interpolation function defined in self.MakeSED() to find value
            return value
        except AttributeError: # Return warning and NaN value if interpolation fails
            value = np.nan
            print(f'Object {self.ID} does not have enought valid flux values to find Luminosity at {wave} through SED interpolation.')
            return value

    def Find_slope(self,wi,wf):
        '''Function to find the slope of the SED between to given wavelength values, wi, wf'''
        fi = self.Find_value(wi) # Use Find_value funtion to find SED value at wi
        ff = self.Find_value(wf) # Use Find_value funtion to find SED value at wf
        slope = np.log10(ff/fi)/np.log10(wf/wi) # Calculate the slope in log space

        return slope

    def SED_shape(self):
        '''Find the "shape" of the SED as defined by 1 of 5 predefined bins'''
        uv_slope = self.Find_slope(
            0.15, 1.0)  # Find the slope of the SED in the UV range
        mir_slope1 = self.Find_slope(1.0, 6.5) # Find the slope of the SED in the NIR-MIR range
        mir_slope2 = self.Find_slope(6.5, 10) # Find the slope of the SED in the MIR range

        # Pre-defined conditions. Check slope values to determine SED shape bin and return bin
        if (uv_slope < -0.3) & (mir_slope1 >= 0.2):
            bin = 1
        elif (uv_slope >= 0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2):
            bin = 2 
        elif (uv_slope > 0.2) & (mir_slope1 >= -0.2):
            bin = 3
        elif (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0):
            bin = 4
        elif (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0):
            bin = 5
        else:
            bin = 6
        
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
        else: 
            # norm_lambdaL_lambda = self.lambdaL_lambda
            norm_lambdaL_lambda = self.nuL_nu

        try:
            return self.ID, self.z, self.rest_w_microns, norm_lambdaL_lambda, self.flux_jy_err/self.flux_jy, self.upper_check
        except AttributeError:
            return self.ID, self.z, self.rest_w_microns, norm_lambdaL_lambda, self.flux_jy_err/self.flux_jy

    def Find_Lbol(self,xin=None,yin=None):
        if xin is None:
            x = self.rest_w_microns*1E-4
            # y = self.lambdaL_lambda
            y = self.nuL_nu
        else:
            x = xin*1E-4
            y = yin

        x = np.append(x, self.FIR_wave*1E-4)
        y = np.append(y, self.FIR_lambdaL_lambda)
        sort = x.argsort()
        x,y = x[sort], y[sort]

        x = np.log10(x[~np.isnan(y)])
        y = np.log10(y[~np.isnan(y)])

        Lbol_interp = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate')

        x_interp = np.linspace(min(x),max(x))
        y = 10**Lbol_interp(x_interp)

        x_interp, y_interp = x_interp[::-1], y_interp[::-1]

        freq = self.c/10**x_interp
        y_interp = y_interp/freq

        self.Lbol = integrate.trapz(y_interp,freq)

        return self.Lbol

    def find_Lum_range(self,xmin,xmax):
        x = np.log10(self.rest_w_cgs[~np.isnan(self.lambdaL_lambda)])
        # y = np.log10(self.lambdaL_lambda[~np.isnan(self.lambdaL_lambda)])
        y = np.log10(self.nuL_nu[~np.isnan(self.nuL_nu)])

        Lbol_interp = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate')

        x_interp = np.linspace(np.log10(xmin*1E-4),np.log10(xmax*1E-4))
        y_interp = 10**Lbol_interp(x_interp)

        x_interp, y_interp = x_interp[::-1], y_interp[::-1]
        freq = self.c/10**x_interp
        y_interp = y_interp/freq
        L_region = integrate.trapz(y_interp,freq)

        return L_region

    def FIR_extrapolation(self,w):
        regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range')
        flux_upper = Filters('filter_list.dat').pull_filter(self.filter_name,' upper limit')*1E-29 # 3 sigma upper limits in cgs
        
        fir_flux_jy = self.flux_jy[regime == 'FIR']*1E-6 # Flux values for the FIR filters
        fir_rest_w_microns = self.rest_w_microns[regime == 'FIR']
        flux_upper /= 3 # 1 sigma upper limit
        Flambda_upper = flux_upper*(self.c/self.obs_w_cgs*2)
        lambdaF_lambda_upper = self.obs_w_cgs*Flambda_upper
        FIR_lambdaF_lambda_upper = lambdaF_lambda_upper[self.regime == 'FIR']

        FIR_lambdaL_lambda_upper = self.Flux_to_Lum(FIR_lambdaF_lambda_upper,self.z)

        FIR_lambdaL_lambda_upper_interp = interpolate.interp1d(np.log10(fir_rest_w_microns),np.log10(FIR_lambdaL_lambda_upper),kind='linear',fill_value='extrapolate')
        Fw_upper = 10**FIR_lambdaL_lambda_upper_interp(np.log10(w))
        Fw = 10**self.f_interp(np.log10(w))

        if ~np.isnan(fir_flux_jy[-3]):
            Fw_use = Fw
            self.upper_check = 0
        elif ~np.isnan(fir_flux_jy[-2]):
            Fw_use = Fw
            self.upper_check = 0
        elif ~np.isnan(fir_flux_jy[-1]):
            Fw_use = Fw
            self.upper_check = 0

        elif ~np.isnan(fir_flux_jy[0]) and ~np.isnan(fir_flux_jy[1]):
            if Fw > Fw_upper:
                Fw_use = Fw_upper
                self.upper_check = 1
            else:
                Fw_use = Fw
                self.upper_check = 0
        else:
            Fw_use = Fw_upper
            self.upper_check = 1
        return Fw_use
        
    def median_FIR(self,filtername,Find_value=np.nan):
        '''
        Function to find the FIR SED either based on data or the upper limits
        Input: list of observational filter names over which to construct SED
        Optional input: wavelength to find FIR SED value at (input in microns)
        '''
        regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range') # Pull the specified wavelength regime from the Filters read file
        flux_upper = Filters('filter_list.dat').pull_filter(filtername,'upper limit')*1E-6 # 3 sigma upper limits from the Filters read file
        flux_upper /= 3 # 1 sigma upper limits

        filt_rest_w_mircons = np.asarray([self.rest_w_microns[self.filter_name == i][0] for i in filtername]) # make array of the rest wavelength in microns for spcified input filters
        filt_rest_w_cgs = np.asarray([self.rest_w_cgs[self.filter_name == i][0] for i in filtername])
        filt_rest_freq = self.c/filt_rest_w_cgs
        fir_flux_jy = self.flux_jy[regime == 'FIR']*1E-6 # Flux values for the FIR filters

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
        lambdaF_lambda = filt_rest_w_cgs*Flambda
        lambdaL_lambda = self.Flux_to_Lum(lambdaF_lambda,self.z)

        nuF_nu = flux_cgs*filt_rest_freq
        nuL_nu = self.Flux_to_Lum(nuF_nu,self.z)

        # x = np.log10(filt_rest_w_mircons[~np.isnan(lambdaL_lambda)])
        # y = np.log10(lambdaL_lambda[~np.isnan(lambdaL_lambda)])
        x = np.log10(filt_rest_w_mircons[~np.isnan(nuL_nu)])
        y = np.log10(nuL_nu[~np.isnan(nuL_nu)])
        upper_f_interp = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        
        lambdaL_lambda_upper = 10**upper_f_interp(np.log10(filt_rest_w_mircons))
        lambdaL_lambda_data = 10**self.f_interp(np.log10(filt_rest_w_mircons))
        value_upper = 10**upper_f_interp(np.log10(Find_value))
        value_data = 10**self.f_interp(np.log10(Find_value))

        if len(fir_flux_jy) >= 5:
            if ~np.isnan(fir_flux_jy[-5]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-4]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-3]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-2]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-1]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0

            elif ~np.isnan(fir_flux_jy[0]) and ~np.isnan(fir_flux_jy[1]):
                if value_data > value_upper:
                    lambdaL_lambda_out = lambdaL_lambda_upper
                    L_value_out = value_upper
                    self.upper_check = 1
                else:
                    lambdaL_lambda_out = lambdaL_lambda_data
                    L_value_out = value_data
                    self.upper_check = 0
            else:
                lambdaL_lambda_out = lambdaL_lambda_upper
                L_value_out = value_upper
                self.upper_check = 1

        else:
            if ~np.isnan(fir_flux_jy[-3]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-2]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0
            elif ~np.isnan(fir_flux_jy[-1]):
                lambdaL_lambda_out = lambdaL_lambda_data
                L_value_out = value_data
                self.upper_check = 0

            elif ~np.isnan(fir_flux_jy[0]) and ~np.isnan(fir_flux_jy[1]):
                if value_data > value_upper:
                    lambdaL_lambda_out = lambdaL_lambda_upper
                    L_value_out = value_upper
                    self.upper_check = 1
                else:
                    lambdaL_lambda_out = lambdaL_lambda_data
                    L_value_out = value_data
                    self.upper_check = 0
            else:
                lambdaL_lambda_out = lambdaL_lambda_upper
                L_value_out = value_upper
                self.upper_check = 1
        
        self.FIR_lambdaL_lambda, self.FIR_wave = np.delete(lambdaL_lambda_out,0), np.delete(filt_rest_w_mircons,0)
        self.FIR_lambdaL_lambda, self.FIR_wave = lambdaL_lambda_out, filt_rest_w_mircons
        if np.isnan(Find_value):
            return self.FIR_lambdaL_lambda, self.FIR_wave
        else:
            return self.FIR_lambdaL_lambda, self.FIR_wave, L_value_out

    def check_SED(self,check_w,check_span=None):
        # Check for an observational data popint within check_span microns of a desired wavelength value (check_w)
        # If check span is not specified use 2 microns
        max_w = check_w + 15 # 15 microns
        if check_span is None:
            min_w = check_w - 2 # 2 microns
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

    def Source_output(self,fname,Lx,Nh,opt):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Long Class to generate restframe SED for AGN target. Can output additional info, suchas as lambdaL_lambda as specified wavelength, slope of SED in given range, general SED shape, Lbol, luminosity under specific region of SED, etc.')
    parser.add_argument('ID', help='Source ID', type=str)
    parser.add_argument('--redshift','-z',help='best redshift measurement', type=float)
    parser.add_argument('--filter','-fn',help='array of names of the obs filter used')
    parser.add_argument('--observed_flux','-obs_flux',help='array of observed flux values for each filter')
    parser.add_argument('--observed_flux_error','-obs_flux_err',help='array of observed flux errors for each filter')

    args = parser.parse_args()
    main(args.ID,args.redshift,args.filter,args.observed_flux,args.observed_flux_error)