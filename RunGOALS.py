import time
import numpy as np
import matplotlib.pyplot as plt
import os
import Lit_functions
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from SED_v8 import AGN
from SED_plots_v2 import Plotter
from filters import Filters
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from SED_shape_plots import SED_shape_Plotter
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr


ti = time.perf_counter() # Start timer
path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

Lx_min = 42.5

'''
Read in data files for GOALS sources
Data comes from several different files for both the X-ray and photometry data:
    Photometry
        The catalog from Vivian U (U2012_GOALS2.fits)
        Gather individual from NED by myself (NED_GOALS.fits)
    X-ray 
        The C-GOALS II catalog - combination of Table 1 and A.1. (CGOALS_Xray.txt) - contains hard and soft X-ray flux - potentially dont need
        Selection of sources from the Ricci 2021 paper (ULIRG_Xray.csv) - contains observed and intrinsic Lx values and flux values
'''

# Read in the Photometry data data
with fits.open('../catalogs/U2012_GOALS2.fits') as hdul:
    U_phot_data = hdul[1].data 

with fits.open('../catalogs/GOALS_total.fits') as hdul:
    Ned_phot_data = hdul[1].data 

U_phot_ID = U_phot_data['ID']
Ned_phot_ID = Ned_phot_data['ID']

# Read in the X-ray data
cgoals = ascii.read('../catalogs/CGOALS_Xray.txt',guess=False,delimiter=',',encoding='utf-8')
ricci = ascii.read('../catalogs/ULIRG_Xray2.csv',guess=False,delimiter=',')

cgoals_ID = cgoals['ID']
cgoals_z = cgoals['z']
cgoals_Lx = cgoals['Lhx']*1E40
cgoals_Lir = cgoals['LIR']
cgoals_Fhx = cgoals['Fhx']*1E-14
cgoals_Fsx = cgoals['Fsx']*1E-14

ricci_ID = np.asarray(ricci['ID'])
ricci_z = np.asarray(ricci['z'])
ricci_Lx_hard = np.asarray(ricci['Lx2_10'])
ricci_Lx_full = np.asarray(ricci['Lx2_10']) + np.log10(1.64)
ricci_Lx_hard_obs = np.asarray(ricci['Lx2_10_obs'])
ricci_Fhx = np.asarray(ricci['Fx2_10'])
ricci_Fsx = np.asarray(ricci['Fx05_2'])
ricci_Nh = np.asarray(ricci['Nh'])
ricci_LIR = np.asarray(ricci['LIR'])


goals_cond = ricci_Lx_full > Lx_min
ricci_ID = ricci_ID[goals_cond]
ricci_z = ricci_z[goals_cond]
ricci_Lx_hard = ricci_Lx_hard[goals_cond]
ricci_Lx_full = ricci_Lx_full[goals_cond]
ricci_Lx_hard_obs = ricci_Lx_hard_obs[goals_cond]
ricci_Fhx = ricci_Fhx[goals_cond]
ricci_Fsx = ricci_Fsx[goals_cond]
ricci_Nh = ricci_Nh[goals_cond]
ricci_LIR = ricci_LIR[goals_cond]

ix, iy = match(ricci_ID,U_phot_ID)
ricci_ID_match_U = ricci_ID[ix]
ricci_z_match_U = ricci_z[ix]
ricci_Lx_hard_match_U = ricci_Lx_hard[ix]
ricci_Lx_full_match_U = ricci_Lx_full[ix]
ricci_Lx_hard_obs_match_U = ricci_Lx_hard_obs[ix]
ricci_Fhx_match_U = ricci_Fhx[ix]
ricci_Fsx_match_U = ricci_Fsx[ix]
ricci_Nh_match_U = ricci_Nh[ix]
ricci_LIR_match_U = ricci_LIR[ix]

U_phot_ID_match = U_phot_ID[iy]

ricci_Fx_hard_match_mjy_U = ricci_Fhx_match_U*4.136E8/(10-2)
ricci_Fx_soft_match_mjy_U = ricci_Fsx_match_U*4.136E8/(2-0.5)

goals_nan_array = np.zeros(np.shape(U_phot_data['HX'][iy]))
goals_nan_array[goals_nan_array == 0] = np.nan

U_goals_flux = np.asarray([
    ricci_Fx_hard_match_mjy_U*1000,
    ricci_Fx_soft_match_mjy_U*1000,
    # U_phot_data['HX'][iy]*1E6,
    # U_phot_data['SX'][iy]*1E6,
    goals_nan_array,
    U_phot_data['FUV'][iy]*1E6,
    U_phot_data['NUV'][iy]*1E6,
    U_phot_data['U'][iy]*1E6,
    U_phot_data['B'][iy]*1E6,
    U_phot_data['V'][iy]*1E6,
    U_phot_data['R'][iy]*1E6,
    U_phot_data['I'][iy]*1E6,
    U_phot_data['J'][iy]*1E6,
    U_phot_data['H'][iy]*1E6,
    U_phot_data['Ks'][iy]*1E6,
    U_phot_data['IRAC1'][iy]*1E6,
    U_phot_data['IRAC2'][iy]*1E6,
    U_phot_data['IRAC3'][iy]*1E6,
    U_phot_data['IRAC4'][iy]*1E6,
    U_phot_data['IRAS1'][iy]*1E6,
    U_phot_data['MIPS1'][iy]*1E6,
    U_phot_data['IRAS2'][iy]*1E6,
    U_phot_data['IRAS3'][iy]*1E6,
    U_phot_data['MIPS2'][iy]*1E6,
    U_phot_data['PACS2'][iy]*1E6,
    U_phot_data['MIPS3'][iy]*1E6,
    U_phot_data['F250'][iy]*1E6,
    U_phot_data['F350'][iy]*1E6,
    U_phot_data['F500'][iy]*1E6
    # U_phot_data['SCUBA2'][iy]*1E6,
    # U_phot_data['VLA1'][iy]*1E6,
    # U_phot_data['VLA2'][iy]*1E6
])

U_goals_flux_err = np.asarray([
    ricci_Fx_hard_match_mjy_U*1000*0.2,
    ricci_Fx_soft_match_mjy_U*1000*0.2,
    # U_phot_data['HXerr'][iy]*1E6,
    # U_phot_data['SXerr'][iy]*1E6,
    goals_nan_array,
    U_phot_data['FUVerr'][iy]*1E6,
    U_phot_data['NUVerr'][iy]*1E6,
    U_phot_data['Uerr'][iy]*1E6,
    U_phot_data['Berr'][iy]*1E6,
    U_phot_data['Verr'][iy]*1E6,
    U_phot_data['Rerr'][iy]*1E6,
    U_phot_data['Ierr'][iy]*1E6,
    U_phot_data['Jerr'][iy]*1E6,
    U_phot_data['Herr'][iy]*1E6,
    U_phot_data['Kserr'][iy]*1E6,
    U_phot_data['IRAC1err'][iy]*1E6,
    U_phot_data['IRAC2err'][iy]*1E6,
    U_phot_data['IRAC3err'][iy]*1E6,
    U_phot_data['IRAC4err'][iy]*1E6,
    U_phot_data['IRAS1err'][iy]*1E6,
    U_phot_data['MIPS1err'][iy]*1E6,
    U_phot_data['IRAS2err'][iy]*1E6,
    U_phot_data['IRAS3err'][iy]*1E6,
    U_phot_data['MIPS2err'][iy]*1E6,
    U_phot_data['PACS2err'][iy]*1E6,
    U_phot_data['MIPS3err'][iy]*1E6,
    U_phot_data['F250err'][iy]*1E6,
    U_phot_data['F350err'][iy]*1E6,
    U_phot_data['F500err'][iy]*1E6
    # U_phot_data['SCUBA2err'][iy]*1E6,
    # U_phot_data['VLA1err'][iy]*1E6,
    # U_phot_data['VLA2err'][iy]*1E6
])

U_goals_flux = U_goals_flux.T
U_goals_flux_err = U_goals_flux_err.T



ix, iy = match(ricci_ID, Ned_phot_ID)
ricci_ID_match_Ned = ricci_ID[ix]
ricci_z_match_Ned = ricci_z[ix]
ricci_Lx_hard_match_Ned = ricci_Lx_hard[ix]
ricci_Lx_full_match_Ned = ricci_Lx_full[ix]
ricci_Lx_hard_obs_match_Ned = ricci_Lx_hard_obs[ix]
ricci_Fhx_match_Ned = ricci_Fhx[ix]
ricci_Fsx_match_Ned = ricci_Fsx[ix]
ricci_Nh_match_Ned = ricci_Nh[ix]
ricci_LIR_match_Ned = ricci_LIR[ix]

Ned_phot_ID_match = Ned_phot_ID[iy]

ricci_Fx_hard_match_mjy_Ned = ricci_Fhx_match_Ned*4.136E8/(10-2)
ricci_Fx_soft_match_mjy_Ned = ricci_Fsx_match_Ned*4.136E8/(2-0.5)

print('Here')
print(ricci_Fx_hard_match_mjy_Ned*1000)
print(np.asarray(Ned_phot_data['Fx_h'][iy], dtype=float)*1E6)

goals_nan_array = np.zeros(np.shape(Ned_phot_data['Fx_h'][iy]))
goals_nan_array[goals_nan_array == 0] = np.nan

# print(Ned_phot_data['Fx_h'][iy])

Ned_goals_flux = np.asarray([
    ricci_Fx_hard_match_mjy_Ned*1000,
    ricci_Fx_soft_match_mjy_Ned*1000,
    # np.asarray(Ned_phot_data['Fx_h'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['Fx_s'][iy],dtype=float)*1E6,
    goals_nan_array,
    np.asarray(Ned_phot_data['FUV'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['NUV'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['U'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['B'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['V'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['R'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['I'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['J'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['H'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Ks'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC1'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC4'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS1'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS1'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['PACS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F250'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F350'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F500'][iy],dtype=float)*1E6
    # np.asarray(Ned_phot_data['SCUBA2'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA1'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA2'][iy],dtype=float)*1E6
])

Ned_goals_flux_err = np.asarray([
    ricci_Fx_hard_match_mjy_Ned*1000*0.2,
    ricci_Fx_soft_match_mjy_Ned*1000*0.2,
    # np.asarray(Ned_phot_data['Fx_h'][iy],dtype=float)*1E6*0.2,
    # np.asarray(Ned_phot_data['Fx_s'][iy],dtype=float)*1E6*0.2,
    goals_nan_array,
    np.asarray(Ned_phot_data['FUVerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['NUVerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Uerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Berr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Verr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Rerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Ierr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Jerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Herr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Kserr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC1err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC4err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS1err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS1err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['PACS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F250err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F350err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F500err'][iy],dtype=float)*1E6
    # np.asarray(Ned_phot_data['SCUBA2err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA1err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA2err'][iy],dtype=float)*1E6
])


Ned_goals_flux = Ned_goals_flux.T
Ned_goals_flux_err = Ned_goals_flux_err.T


#### Read in Galaxy Templates ####
temp = ascii.read('/Users/connor_auge/Desktop/templets/A10_templates.txt')
temp_wave = np.asarray(temp['Wave'])
temp_flux = np.asarray(temp['E'])*1E-16  # erg/s/cm^-2/Hz
temp_wave_cgs = temp_wave*1E-8
temp_freq = 3E10/temp_wave_cgs
temp_nuFnu = temp_flux*temp_freq

dl = 10
dl_cgs = dl*3.086E18
temp_lum = temp_nuFnu*4*np.pi*dl_cgs**2

scale_array = [1.87E44, 2.33E44, 3.93E44]
###################################


# goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                # 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500', 'SCUBA2', 'VLA1', 'VLA2'])
goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])


###############################################################################
############################## Start CIGALE File ##############################
# cigale_name = 'GOALS_med2.mag'
# inf = open(f'../xcigale/data_input/{cigale_name}', 'w')
# header = np.asarray(['# id', 'redshift'])
# cigale_filters = Filters('filter_list.dat').pull_filter(goals_filter_name, 'xcigale name')
# for i in range(len(cigale_filters)):
#     header = np.append(header, cigale_filters[i])
#     header = np.append(header, cigale_filters[i]+'_err')
# np.savetxt(inf, header, fmt='%s', delimiter='    ', newline=' ')
# print('header: ',np.shape(header))
# inf.close()
###############################################################################


'''CGOALS ULIRGS'''
norm_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = [], [], [], [], []
FFIR_ulirg, WFIR_ulirg = [], []
UVslope_ulirg, MIRslope1_ulirg, MIRslope2_ulirg = [], [], []
uv_lum_ulirg, opt_lum_ulirg, mir_lum_ulirg, fir_lum_ulirg = [], [], [], []
ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_frac_err = [], [], [], [], []
median_x_ulirg, median_y_ulirg = [], []
median_fir_x_ulirg, median_fir_y_ulirg = [], []
ulirg_Lx_out, ulirg_Lx_corr_out = [], []
Lbol_ulirg, Lbol_ulirg_sub = [], []
ulirg_field = []
ulirg_up_check = []
ulirg_Nh_out, ulirg_LIR_out = [], []
goals_irac_ch1, goals_irac_ch2, goals_irac_ch3, goals_irac_ch4 = [], [], [], []
int_x, int_y = [], []

ulirg_frac_error = []
ulirg_FIR_upper_lims = []

med_flux = []

IRx, IRy, IRagn = [], [], []

# '''
# fill_nan = np.zeros(len(ned_goals_filter_name)-len(cgoals_filter_name))
# fill_nan[fill_nan == 0] = np.nan
# for i in range(len(ricci_ID_match_U)):
#     if ricci_ID_match_U[i] == 'UGC 08058':
#         continue
#     elif len(U_goals_flux[i][U_goals_flux[i] > 0]) > 1:
#         print('U: ',ricci_ID_match_U[i])
#         source = AGN(ricci_ID_match_U[i],ricci_z_match_U[i],goals_filter_name,U_goals_flux[i],U_goals_flux_err[i])
#         source.MakeSED()
#         med_flux.append(U_goals_flux[i])
#         source.FIR_extrap(['FLUX_24','IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

#         ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
#         median_x_ulirg.append(ix)
#         median_y_ulirg.append(iy)

#         wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
#         WFIR_ulirg.append(wfir)
#         FFIR_ulirg.append(ffir)
#         F100_ulirg.append(f100/source.Find_value(1.0))

#         lbol = source.Find_Lbol()
#         lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
#         shape = source.SED_shape()

#         f1 = source.Find_value(1.0)
#         xval = source.Find_value(3E-4)
#         f6 = source.Find_value(6.0)
#         f025 = source.Find_value(0.25)
#         f10 = source.Find_value(10)
#         f2kev = source.Find_value(6.1992e-4)

#         Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
#         ulirg_id.append(Id)
#         ulirg_x.append(w)
#         ulirg_y.append(f)
#         ulirg_frac_error.append(frac_err)
#         ulirg_FIR_upper_lims.append(up_check)
#         ulirg_Lx_out.append(np.log10(ricci_Lx_hard_obs_match_U[i]))
#         ulirg_Lx_corr_out.append(ricci_Lx_full_match_U[i])
#         ulirg_Nh_out.append(ricci_Nh_match_U[i])
#         ulirg_LIR_out.append(ricci_LIR_match_U[i])
#         ulirg_z.append(ricci_z_match_U[i])

#         # plot = Plotter(Id, redshift, w, f, ricci_Lx_full_match_U[i],f1,up_check)
#         # plot.PlotSED(point_x=100,point_y=f100/f1)

#         norm_ulirg.append(f1)
#         F025_ulirg.append(f025)
#         F6_ulirg.append(f6)
#         F10_ulirg.append(f10)
#         F100_ulirg.append(f100)
#         Lbol_ulirg.append(lbol)
#         Lbol_ulirg_sub.append(lbol_sub)

#         UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
#         MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
#         MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

#         uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
#         opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
#         mir_lum_ulirg.append(source.find_Lum_range(3,30))
#         fir_lum_ulirg.append(source.find_Lum_range(30,500/(1+ricci_z_match_U[i])))

#         goals_irac_ch1.append(U_goals_flux[i][goals_filter_name == 'SPLASH_1_FLUX'][0])
#         goals_irac_ch2.append(U_goals_flux[i][goals_filter_name == 'SPLASH_2_FLUX'][0])
#         goals_irac_ch3.append(U_goals_flux[i][goals_filter_name == 'SPLASH_3_FLUX'][0])
#         goals_irac_ch4.append(U_goals_flux[i][goals_filter_name == 'SPLASH_4_FLUX'][0])
        
#         # source.write_cigale_file(cigale_name,goals_filter_name)
#         ulirg_field.append(5)

print(np.shape(ricci_ID_match_Ned))
for i in range(len(ricci_ID_match_Ned)):
    # if ricci_ID_match_Ned[i] == 'UGC 08058':
        # continue
    # if len(Ned_goals_flux[i][Ned_goals_flux[i] > 0]) > 1:
        print(i,ricci_ID_match_Ned[i], ricci_LIR_match_Ned[i])
        source = AGN(ricci_ID_match_Ned[i],ricci_z_match_Ned[i],goals_filter_name,Ned_goals_flux[i],Ned_goals_flux_err[i])
        source.MakeSED()
        med_flux.append(Ned_goals_flux[i])
        source.FIR_extrap(['FLUX_24','IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        median_x_ulirg.append(ix)
        median_y_ulirg.append(iy)

        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
        WFIR_ulirg.append(wfir)
        FFIR_ulirg.append(ffir)
        F100_ulirg.append(f100/source.Find_value(1.0))

        ir_x, ir_y, ir_agn = source.IR_colors('SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX')
        IRx.append(ir_x)
        IRy.append(ir_y)
        IRagn.append(ir_agn)

        f1 = source.Find_value(1.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)
        f2kev = source.Find_value(6.1992e-4)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
        shape = source.SED_shape()

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
        ulirg_id.append(Id)
        ulirg_x.append(w)
        ulirg_y.append(f)
        ulirg_frac_error.append(frac_err)
        ulirg_FIR_upper_lims.append(up_check)
        ulirg_Lx_out.append(ricci_Lx_hard_obs_match_Ned[i])
        ulirg_Lx_corr_out.append(ricci_Lx_full_match_Ned[i])
        ulirg_Nh_out.append(ricci_Nh_match_Ned[i])
        ulirg_LIR_out.append(ricci_LIR_match_Ned[i])
        ulirg_z.append(ricci_z_match_Ned[i])
        
        # plot = Plotter(Id, redshift, w, f, ricci_Lx_full_match_Ned[i],f1,up_check)
        # plot.PlotSED(point_x=100,point_y=f100/f1)

        norm_ulirg.append(f1)
        F025_ulirg.append(f025)
        F6_ulirg.append(f6)
        F10_ulirg.append(f10)
        F100_ulirg.append(f100)
        Lbol_ulirg.append(lbol)
        Lbol_ulirg_sub.append(lbol_sub)

        UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
        MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
        MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

        uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
        opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
        mir_lum_ulirg.append(source.find_Lum_range(3,30))
        fir_lum_ulirg.append(source.find_Lum_range(30,500/(1+ricci_z_match_Ned[i])))

        goals_irac_ch1.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_1_FLUX'][0])
        goals_irac_ch2.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_2_FLUX'][0])
        goals_irac_ch3.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_3_FLUX'][0])
        goals_irac_ch4.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_4_FLUX'][0])
        
        # source.write_cigale_file(cigale_name, goals_filter_name, use_int_fx=False)
        ulirg_field.append(5)


# '''
print('Done with ULIRGS')

ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_Lx_out, ulirg_Lx_corr_out, median_x_ulirg, median_y_ulirg = np.asarray(ulirg_id), np.asarray(ulirg_z), np.asarray(ulirg_x), np.asarray(ulirg_y), np.asarray(ulirg_Lx_out), np.asarray(ulirg_Lx_corr_out), np.asarray(median_x_ulirg), np.asarray(median_y_ulirg)
norm_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = np.asarray(norm_ulirg), np.asarray(F025_ulirg), np.asarray(F6_ulirg), np.asarray(F10_ulirg), np.asarray(F100_ulirg)
UVslope_ulirg, MIRslope1_ulirg, MIRslope2_ulirg = np.asarray(UVslope_ulirg), np.asarray(MIRslope1_ulirg), np.asarray(MIRslope2_ulirg)
FFIR_ulirg, WFIR_ulirg = np.asarray(FFIR_ulirg), np.asarray(WFIR_ulirg)
Lbol_ulirg, Lbol_ulirg_sub = np.asarray(Lbol_ulirg), np.asarray(Lbol_ulirg_sub)
median_fir_x_ulirg, median_fir_y_ulirg = np.asarray(median_fir_x_ulirg), np.asarray(median_fir_y_ulirg)
uv_lum_ulirg, opt_lum_ulirg, mir_lum_ulirg, fir_lum_ulirg = np.asarray(uv_lum_ulirg), np.asarray(opt_lum_ulirg), np.asarray(mir_lum_ulirg), np.asarray(fir_lum_ulirg)
ulirg_field = np.asarray(ulirg_field)
ulirg_up_check = np.asarray(ulirg_up_check)
ulirg_Nh_out = np.asarray(ulirg_Nh_out)
ulirg_LIR_out = np.asarray(ulirg_LIR_out)
goals_irac_ch1, goals_irac_ch2, goals_irach_ch3, goals_irach_ch4 = np.asarray(goals_irac_ch1), np.asarray(goals_irac_ch2), np.asarray(goals_irac_ch3), np.asarray(goals_irac_ch4)
IRx, IRy, IRagn = np.asarray(IRx), np.asarray(IRy), np.asarray(IRagn)

ulirg_plot = Plotter_Letter(ulirg_id,ulirg_z,ulirg_x,ulirg_y,ulirg_frac_err)
plot = Plotter(ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_Lx_corr_out, norm_ulirg, ulirg_FIR_upper_lims)

for i in range(len(ulirg_id)):
    print(i,ulirg_id[i], ulirg_LIR_out[i],IRagn[i])

med_flux = np.asarray(med_flux)
med_flux_combine = np.nanmedian(med_flux,axis=0)

# source_combine = AGN('Med_GOALS_obs', np.nanmedian(ricci_z_match_Ned), goals_filter_name,med_flux_combine,med_flux_combine*0.3)
# source_combine.write_cigale_file(cigale_name, goals_filter_name)

# ulirg_plot.multi_SED('ULIRG_n',ulirg_x,ulirg_y,ulirg_Lx_out,median_x_ulirg,median_y_ulirg,suptitle='SEDs of ULIRGs',norm=norm_ulirg,mark=ulirg_field,spec_z=ulirg_z,wfir=None,ffir=None,up_check=ulirg_up_check,med_x_fir=median_fir_x_ulirg,med_y_fir=median_fir_y_ulirg)
# plot.multi_SED('GOALS_figs/SEDs_obsLx',median_x=median_x_ulirg,median_y=median_y_ulirg,wfir=WFIR_ulirg,ffir=FFIR_ulirg,Median_line=False,FIR_upper='data only')
# plot.L_scatter_comp('GOALS_figs/Lx_comp',ulirg_Lx_out,ulirg_Lx_corr_out,color_array=ulirg_LIR_out,xlabel=r'Observed log L$_{\rm X}$ [erg/s]', ylabel=r'Intrinsic log L$_{\rm X}$ [erg/s]',colorbar_label=r'log L$_{\rm IR}$ [L$_\odot$]')
# plot.IR_colors('GOALS_figs/IR colors2',IRx,IRy,ulirg_LIR_out,colorbar=True,colorbar_label=r'log L$_{\rm IR}$ [L$_\odot$]',Lacy=True,agn=IRagn)
# print(Lbol_ulirg_sub)
# print(ulirg_Lx_corr_out)
# print(ulirg_LIR_out) 

# for i in range(len(ulirg_Lx_out)):
#     print(ulirg_Lx_out[i],ulirg_Lx_corr_out[i],ulirg_Lx_out[i]/ulirg_Lx_corr_out[i])

# print(ulirg_Lx_out/ulirg_Lx_corr_out)
# print(np.nanmedian(ulirg_Lx_out/ulirg_Lx_corr_out))