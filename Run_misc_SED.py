import time 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from SED_v8 import AGN
from SED_plots_v2 import Plotter
from filters import Filters
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr

path = '/Users/connor_auge/Research/Disertation/catalogs/'

ti = time.perf_counter()


# with fits.open(path+'cand_final_forSED.fits') as hdul:
#     phot = hdul[1].data
#     cols = hdul[1].columns

with fits.open(path+'cand_final_forSED_int_fluxes.fits') as hdul:
    phot = hdul[1].data
    cols = hdul[1].columns


phot_id = phot['srcid']
phot_z = phot['spec_z']
Fx_full = phot['FULL_FLUX']
Fx_hard = phot['HARD_FLUX']
Fx_soft = phot['SOFT_FLUX']
Fx_full_int = phot['flux_fi']
Fx_hard_int = phot['flux_hi']
Fx_soft_int = phot['flux_si']
Fx_full_err = phot['FULL_FLUX_ERROR']
Fx_hard_err = phot['HARD_FLUX_ERROR']
Fx_soft_err = phot['SOFT_FLUX_ERROR']
ra = phot['cp_ra']
dec = phot['cp_dec']

up_area = (np.logical_and(ra >= 13, ra <= 37))

s82x_Fx_full_mjy = Fx_full*4.136E8/(10-0.5)
s82x_Fx_hard_mjy = Fx_hard*4.136E8/(10-2)
s82x_Fx_soft_mjy = Fx_soft*4.136E8/(2-0.5)
s82x_Fx_full_int_mjy = Fx_full_int*4.136E8/(10-0.5)
s82x_Fx_hard_int_mjy = Fx_hard_int*4.136E8/(10-2)
s82x_Fx_soft_int_mjy = Fx_soft_int*4.136E8/(2-0.5)
s82x_Fxerr_full_mjy = Fx_full_err*4.136E8/(10-0.5)
s82x_Fxerr_hard_mjy = Fx_hard_err*4.136E8/(10-2)
s82x_Fxerr_soft_mjy = Fx_soft_err*4.136E8/(2-0.5)

# print('WISE')
# print(phot['W1'][phot_id == 1476])
# print(phot['W2'][phot_id == 1476])
# print(phot['W3'][phot_id == 1476])
# print(phot['W4'][phot_id == 1476])

# s82x_flux_array = np.array([
#     s82x_Fx_hard_mjy*1000, s82x_Fx_soft_mjy*1000,
#     mag_to_flux(phot['mag_FUV'], 'FUV')*1E6,
#     mag_to_flux(phot['mag_NUV'], 'NUV')*1E6,
#     mag_to_flux(phot['u'], 'sloan_u')*1E6,
#     mag_to_flux(phot['g'], 'sloan_g')*1E6,
#     mag_to_flux(phot['r'], 'sloan_r')*1E6,
#     mag_to_flux(phot['i'], 'sloan_i')*1E6,
#     mag_to_flux(phot['z'], 'sloan_z')*1E6,
#     mag_to_flux(phot['JVHS'], 'JVHS')*1E6,
#     mag_to_flux(phot['HVHS'], 'HVHS')*1E6,
#     mag_to_flux(phot['KVHS'], 'KVHS')*1E6,
#     mag_to_flux(phot['W1'], 'W1')*1E6,
#     mag_to_flux(phot['W2'], 'W2')*1E6,
#     mag_to_flux(phot['W3'], 'W3')*1E6,
#     mag_to_flux(phot['W4'], 'W4')*1E6,
#     phot['F250']*1000,
#     phot['F350']*1000,
#     phot['F500']*1000
# ])

# s82x_flux_err_array = np.array([
#     s82x_Fxerr_hard_mjy*1000, s82x_Fxerr_soft_mjy*1000,
#     mag_to_flux(phot['mag_fuv'],'FUV')*1E6*0.2,
#     magerr_to_fluxerr(phot['mag_NUV'],
#                       phot['magerr_NUV'], 'NUV')*1E6,
#     magerr_to_fluxerr(phot['u'],
#                       phot['u_err'], 'sloan_u')*1E6,
#     magerr_to_fluxerr(phot['g'],
#                       phot['g_err'], 'sloan_g')*1E6,
#     magerr_to_fluxerr(phot['r'],
#                       phot['r_err'], 'sloan_r')*1E6,
#     magerr_to_fluxerr(phot['i'],
#                       phot['i_err'], 'sloan_i')*1E6,
#     magerr_to_fluxerr(phot['z'],
#                       phot['z_err'], 'sloan_z')*1E6,
#     magerr_to_fluxerr(phot['JVHS'],
#                       phot['JVHS_err'], 'JVHS')*1E6,
#     magerr_to_fluxerr(phot['HVHS'],
#                       phot['HVHS_err'], 'HVHS')*1E6,
#     magerr_to_fluxerr(phot['KVHS'],
#                       phot['KVHS_err'], 'KVHS')*1E6,
#     magerr_to_fluxerr(phot['W1'],
#                       phot['W1_err'], 'W1')*1E6,
#     magerr_to_fluxerr(phot['W2'],
#                       phot['W2_err'], 'W2')*1E6,
#     magerr_to_fluxerr(phot['W3'],
#                       phot['W3_err'], 'W3')*1E6,
#     magerr_to_fluxerr(phot['W4'],
#                       phot['W4_err'], 'W4')*1E6,
#     phot['F250_err']*1000,
#     phot['F350_err']*1000,
#     phot['F500_err']*1000
# ])

s82x_flux_array = np.array([
    # s82x_Fx_full_mjy*1000,
    s82x_Fx_full_int_mjy*1000,
    mag_to_flux(phot['mag_FUV'], 'FUV')*1E6,
    mag_to_flux(phot['mag_NUV'], 'NUV')*1E6,
    mag_to_flux(phot['u'], 'sloan_u')*1E6,
    mag_to_flux(phot['g'], 'sloan_g')*1E6,
    mag_to_flux(phot['r'], 'sloan_r')*1E6,
    mag_to_flux(phot['i'], 'sloan_i')*1E6,
    mag_to_flux(phot['z'], 'sloan_z')*1E6,
    mag_to_flux(phot['JVHS'], 'JVHS')*1E6,
    mag_to_flux(phot['HVHS'], 'HVHS')*1E6,
    mag_to_flux(phot['KVHS'], 'KVHS')*1E6,
    mag_to_flux(phot['W1'], 'W1')*1E6,
    mag_to_flux(phot['W2'], 'W2')*1E6,
    mag_to_flux(phot['W3'], 'W3')*1E6,
    mag_to_flux(phot['W4'], 'W4')*1E6,
    phot['F250']*1000,
    phot['F350']*1000,
    phot['F500']*1000
])

s82x_flux_err_array = np.array([
    s82x_Fxerr_full_mjy*1000,
    mag_to_flux(phot['mag_fuv'], 'FUV')*1E6*0.2,
    magerr_to_fluxerr(phot['mag_NUV'],
                      phot['magerr_NUV'], 'NUV')*1E6,
    magerr_to_fluxerr(phot['u'],
                      phot['u_err'], 'sloan_u')*1E6,
    magerr_to_fluxerr(phot['g'],
                      phot['g_err'], 'sloan_g')*1E6,
    magerr_to_fluxerr(phot['r'],
                      phot['r_err'], 'sloan_r')*1E6,
    magerr_to_fluxerr(phot['i'],
                      phot['i_err'], 'sloan_i')*1E6,
    magerr_to_fluxerr(phot['z'],
                      phot['z_err'], 'sloan_z')*1E6,
    magerr_to_fluxerr(phot['JVHS'],
                      phot['JVHS_err'], 'JVHS')*1E6,
    magerr_to_fluxerr(phot['HVHS'],
                      phot['HVHS_err'], 'HVHS')*1E6,
    magerr_to_fluxerr(phot['KVHS'],
                      phot['KVHS_err'], 'KVHS')*1E6,
    magerr_to_fluxerr(phot['W1'],
                      phot['W1_err'], 'W1')*1E6,
    magerr_to_fluxerr(phot['W2'],
                      phot['W2_err'], 'W2')*1E6,
    magerr_to_fluxerr(phot['W3'],
                      phot['W3_err'], 'W3')*1E6,
    magerr_to_fluxerr(phot['W4'],
                      phot['W4_err'], 'W4')*1E6,
    phot['F250_err']*1000,
    phot['F350_err']*1000,
    phot['F500_err']*1000
])

# Transpose arrays so each row is a new source and each column is a obs filter
s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T


# S82X_filters = np.asarray(['Fx_hard', 'Fx_soft', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
#                           'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

S82X_filters = np.asarray(['Fx_full', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
                          'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])


##################################################
##################################################
cigale_name = 's82_comp_thick/peca_sources_int.mag'
inf = open(f'../xcigale/data_input/{cigale_name}', 'w')
header = np.asarray(['# id', 'redshift'])
cigale_filters = Filters('filter_list.dat').pull_filter(
    S82X_filters, 'xcigale name')
for i in range(len(cigale_filters)):
    header = np.append(header, cigale_filters[i])
    header = np.append(header, cigale_filters[i]+'_err')
np.savetxt(inf, header, fmt='%s', delimiter='    ', newline=' ')
inf.close()
##################################################
##################################################


for i in range(len(phot_id)):
    source = AGN(phot_id[i],phot_z[i],S82X_filters,s82x_flux_array[i],s82x_flux_err_array[i])
    source.MakeSED()
    source.FIR_extrap(['W4', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0,discreet=True)

    f1 = source.Find_value(1.0)
    f025 = source.Find_value(0.25)

    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    plot = Plotter(Id, redshift, w, f, s82x_Fx_full_mjy[i], f1, up_check)
    plot.PlotSED(point_x=0.25,point_y=f025/f1)
    
    shape = source.SED_shape()

    print(phot_id[i],shape,ra[i],dec[i],up_area[i])

    # source.write_cigale_file(cigale_name, use_int_fx=False, use_upper=up_area[i])
