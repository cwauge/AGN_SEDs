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
        self.obs_freq = self.c/self.obs_w_cgs # convert obs wavelength to a frequency

        # unit converstion and quality check for flux values
        self.flux_jy = self.obs_f*1E-6 # convert the flux values from microJy to Jy
        self.flux_jy_err = self.obs_f_err*1E-6 # convert the flux errors from microJyto Jy
        self.flux_jy[self.flux_jy <= 0] = np.nan # replace negative or zero flux values with nan
        self.flux_jy_err[self.flux_jy_err <= 0] = np.nan # replace negative or zero error values with nan
        self.flux_jy[np.isnan(self.flux_jy_err)] = np.nan # replace flux values with no errors with nan
        self.flux_jy[self.flux_jy_err/self.flux_jy >= 0.5] = np.nan # Remove flux values with frac error > 50%

        # convert flux from frequency space to wavelength
        self.Fnu = self.flux_jy*1E-23 # convert flux from Jy to cgs: erg s^-1 cm^-2 Hz^-1
        self.Flambda = self.Fnu*(self.c/self.obs_w_cgs**2) 

        self.lambdaF_lambda = self.obs_w_cgs*self.Flambda # convert units to erg s^-1 cm^-2
        self.lambdaL_lambda = self.Flux_to_Lum(self.lambdaF_lambda,self.z) # convert flux to luminosity [erg s^-1]
        self.nuF_nu = self.obs_freq*self.Fnu
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

    def Find_value(self,wave,limit_factor=2):
        '''
        Function to find the value of the SED at a given wavelength (wave in microns)
        limit_factor is a scale factor to look for nearby data in Check_nearest functin. 
        If no nearby data is found, then no interpolated value is returned. 
        Default factor is twice the wavelength
        '''
        limit = wave*limit_factor
        if wave > 12:
            value = self.L_FIR_value_out
        else:
            if self.Check_nearest(wave,limit):
                try:
                    value = 10**self.f_interp(np.log10(wave)) # Use interpolation function defined in self.MakeSED() to find value
                except AttributeError: # Return warning and NaN value if interpolation fails
                    value = np.nan
                    # print(f'Object {self.ID} does not have enought valid flux values to find Luminosity at {wave} through SED interpolation.')
            else:
                value = np.nan
                # print(f'Object {self.ID} does not have enought valid flux values to find Luminosity at {wave} through SED interpolation.')
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
        fi = self.Find_value(wi) # Use Find_value funtion to find SED value at wi
        ff = self.Find_value(wf) # Use Find_value funtion to find SED value at wf
        slope = np.log10(ff/fi)/np.log10(wf/wi) # Calculate the slope in log space

        return slope

    def SED_shape(self):
        '''Find the "shape" of the SED as defined by 1 of 5 predefined bins'''
        uv_slope = self.Find_slope(0.15, 1.0)  # Find the slope of the SED in the UV range
        mir_slope1 = self.Find_slope(1.0, 6.5) # Find the slope of the SED in the NIR-MIR range
        mir_slope2 = self.Find_slope(6.5, 10) # Find the slope of the SED in the MIR range

        # Pre-defined conditions. Check slope values to determine SED shape bin and return bin
        if (uv_slope < -0.3) & (mir_slope1 >= 0.2):
            bin = 1
        elif (uv_slope >= -0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2):
            bin = 2 
        elif (uv_slope > 0.2) & (mir_slope1 >= -0.2):
            bin = 3
        elif (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0):
            bin = 4
        elif (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0):
            bin = 5
        else:
            bin = 6
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
        y_interp = 10**Lbol_interp(x_interp)

        x_interp, y_interp = x_interp[::-1], y_interp[::-1]

        freq = self.c/10**x_interp
        y_interp = y_interp/freq

        self.Lbol = integrate.trapz(y_interp,freq)

        return self.Lbol

    def Find_Lbol_temp_sub(self,scale_L,temp_x,temp_y):
        Lone_temp = temp_y[temp_x == 1.0050][0]
        if self.z <= 0.6:
            scale = scale_L[0]/Lone_temp
        elif (self.z > 0.6) & (self.z < 0.9):
            scale = scale_L[1]/Lone_temp
        else:
            scale = scale_L[2]/Lone_temp
        scale_y = temp_y*scale

        temp_interp = interpolate.interp1d(np.log10(temp_x), np.log10(scale_y),kind='linear',fill_value='extrapolate')
        y_interp = 10**temp_interp(np.log10(self.rest_w_microns))

        y_sub = self.nuL_nu - y_interp

        return self.Find_Lbol(xin=self.rest_w_microns[y_sub > 0], yin=y_sub[y_sub > 0])

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

    def FIR_extrap(self,filtername):
        '''
        Function to find the FIR SED either based on data or the upper limits
        Input: list of observational filter names over which to construct SED
        Optional input: wavelength to find FIR SED value at (input in microns)
        '''
        self.FIR_filt = filtername
        regime = Filters('filter_list.dat').pull_filter(self.filter_name,'wavelength range') # Pull the specified wavelength regime from the Filters read file
        flux_upper = Filters('filter_list.dat').pull_filter(filtername,'upper limit')*1E-6 # 3 sigma upper limits from the Filters read file
        flux_upper /= 3 # 1 sigma upper limits

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
        
    def Int_SED_FIR(self,Find_value=np.nan,discreet=False):
        xmin = min(self.filt_rest_w_mircons)
        xmax = max(self.filt_rest_w_mircons)
        if discreet:
            xfir_out = np.linspace(xmin, xmax, 12)
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
                self.L_FIR_value_out = value_upper
                self.upper_check = 1
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
        
        self.FIR_lambdaL_lambda, self.FIR_wave = yfir_out, xfir_out
        if np.isnan(Find_value):
            return self.FIR_wave, self.FIR_lambdaL_lambda
        else:
            return self.FIR_wave, self.FIR_lambdaL_lambda, self.L_FIR_value_out

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
            Nh
            Lbol
            SED Shape
        '''
        cols = ['Field','x_ID','phot_id','RAJ2000','DEJ2000','L0510_c','Nh','Lbol','SED_shape']
        data = [field,xid,self.ID,ra,dec,Lx,Nh,self.Lbol,self.shape]

        return cols, data

    def output_phot(self,field,filtername_tot,filtername_field):
        '''
        Function to output the photometry measurements of each source
        Output:
            Phot ID
            Flux measurements and errors for all possible photometry bands 
        '''
        cols = np.asarray(['Field','phot_id'])
        data = np.asarray([field,self.ID])
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

        flux_err_data[0::2] = self.flux_jy
        flux_err_data[1::2] = self.flux_jy_err
        # data = np.append(data,flux_err_data)

        data_in = np.zeros(col_names.size)
        data_in[data_in == 0] = -99.99

        flux_err_data[np.isnan(flux_err_data)] = -99.99

        data_out = self.match_filters(field_names, col_names, flux_err_data, data_in)
        cols_out = np.append(cols,col_names)
        data_out = np.append(data,data_out)

        return cols_out, data_out

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

    def write_output_file(self,fname,data_in,cols,opt='w'):
        '''
        Function to wirte a fits file of the data contained in the output functions
        '''
        cols = np.asarray(cols,dtype=str)
        data_in = np.asarray(data_in,dtype=str)
        t = Table(data=data_in,names=cols)

        if 'w' in opt:
            try:
                fin = fits.open(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}')
                fdata = fin[1].data
                fcols = fin[1].columns.names
                fin.close()
                tin = Table(data=fdata,names=fcols)
                tin.add_row(data_in)

                tin.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='fits',overwrite=True)

            except FileNotFoundError:
                t.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='fits',overwrite=True)

        elif 'a' in opt:
            fin = fits.open(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}')
            fdata = fin[1].data
            fcols = fin[1].columns.names
            fin.close()
            tin = Table(data=fdata,names=fcols)
            tin.add_row(data_in)

            tin.write(f'/Users/connor_auge/Research/Disertation/catalogs/output/{fname}',format='fits',overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Long Class to generate restframe SED for AGN target. Can output additional info, suchas as lambdaL_lambda as specified wavelength, slope of SED in given range, general SED shape, Lbol, luminosity under specific region of SED, etc.')
    parser.add_argument('ID', help='Source ID', type=str)
    parser.add_argument('--redshift','-z',help='best redshift measurement', type=float)
    parser.add_argument('--filter','-fn',help='array of names of the obs filter used')
    parser.add_argument('--observed_flux','-obs_flux',help='array of observed flux values for each filter')
    parser.add_argument('--observed_flux_error','-obs_flux_err',help='array of observed flux errors for each filter')

    args = parser.parse_args()
    main(args.ID,args.redshift,args.filter,args.observed_flux,args.observed_flux_error)