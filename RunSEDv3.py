import time
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from SED_v7 import AGN
from SED_plots import Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from match import match 
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr

ti = time.perf_counter()

path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

# set redshift and X-ray luminosity limits
z_min = 0.0
z_max = 1.2

Lx_min = 43
Lx_max = 50

###################################################################################
###################################################################################
############################ Read in COSMOS2020 files #############################

# Read in data
cosmos = fits.open(path+'cosmos2020/classic/COSMOS2020_CLASSIC_R1_v2.0_master.fits')
cosmos_data = cosmos[1].data
cosmos.close()

chandra_cosmos = fits.open(path+'chandra_COSMOS_legacy_opt_NIR_counterparts_20160113_4d.fits')
chandra_cosmos_data = chandra_cosmos[1].data
chandra_cosmos.close()

chandra_cosmos2 = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_20210225.fits')
chandra_cosmos2_data = chandra_cosmos2[1].data
chandra_cosmos2.close()

chandra_cosmos_ct = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_ComptonThick_Lanzuisi18.fits')
chandra_cosmos_ct_data = chandra_cosmos_ct[1].data
chandra_cosmos_ct.close()

deimos = ascii.read('/Users/connor_auge/Downloads/deimos_10k_March2018_new/deimos_redshifts.tbl')
deimos_id = np.asarray(deimos['ID'])
deimos_z = np.asarray(deimos['zspec'])
deimos_remarks = np.asarray(deimos['Remarks'])
deimos_ID = np.asarray([int(i[1:]) for i in deimos_id if 'L' in i])
deimos_z_spec = np.asarray([deimos_z[i] for i in range(len(deimos_z)) if 'L' in deimos_id[i]])

chandra_cosmos_phot_id = chandra_cosmos_data['id_k_uv']
cosmos_laigle_id = cosmos_data['ID_COSMOS2015']
cosmos_xid = cosmos_data['id_chandra']
chandra_cosmos_xid = chandra_cosmos_data['id_x']
chandra_cosmos2_xid = chandra_cosmos2_data['id_x']
chandra_cosmos_ct_xid = chandra_cosmos_ct_data['id_x']

chandra_cosmos_RA = chandra_cosmos_data['RA_x']
chandra_cosmos_DEC = chandra_cosmos_data['DEC_x']

cosmos_sz = cosmos_data['sz_zspec']
cosmos_ez = cosmos_data['ez_z_spec']
chandra_cosmos_z = chandra_cosmos_data['z_spec']
chandra_cosmos_z_phot = chandra_cosmos_data['z_best']
chandra_cosmos_Fx_hard = chandra_cosmos_data['flux_h']
chandra_cosmos_Fx_soft = chandra_cosmos_data['flux_s']
chandra_cosmos_Fx_full = chandra_cosmos_data['flux_f']
chandra_cosmos_Lx_hard = np.asarray([10**i for i in chandra_cosmos_data['Lx_210']])
chandra_cosmos_Lx_soft = np.asarray([10**i for i in chandra_cosmos_data['Lx_052']])
chandra_cosmos_Lx_full = np.asarray([10**i for i in chandra_cosmos_data['Lx_0510']])
chandra_cosmos_spec_type = chandra_cosmos_data['spec_type']
chandra_cosmos_nh = chandra_cosmos_data['Nh']
chandra_cosmos_nh_lo = chandra_cosmos_data['Nh_lo']
chandra_cosmos_nh_hi = chandra_cosmos_data['Nh_up']
chandra_cosmos_abs_corr_h = chandra_cosmos_data['abs_corr_210']
chandra_cosmos_abs_corr_up_h = chandra_cosmos_data['abs_corr_210_up']
chandra_cosmos_abs_corr_lo_h = chandra_cosmos_data['abs_corr_210_lo']
chandra_cosmos_abs_corr_s = chandra_cosmos_data['abs_corr_052']
chandra_cosmos_abs_corr_up_s = chandra_cosmos_data['abs_corr_052_up']
chandra_cosmos_abs_corr_lo_s = chandra_cosmos_data['abs_corr_052_lo']
chandra_cosmos_abs_corr_f = chandra_cosmos_data['abs_corr_0510']
chandra_cosmos_abs_corr_up_f = chandra_cosmos_data['abs_corr_0510_up']
chandra_cosmos_abs_corr_lo_f = chandra_cosmos_data['abs_corr_0510_lo']
chandra_cosmos2_Lx_hard = np.asarray([10**i for i in chandra_cosmos2_data['Lx_210']])
chandra_cosmos2_Lx_full = np.asarray([(10**i)*1.64 for i in chandra_cosmos2_data['Lx_210']])
chandra_cosmos2_z = chandra_cosmos2_data['z_best']
chandra_cosmos2_nh = chandra_cosmos2_data['nh']
chandra_cosmos2_nh_lo_err = chandra_cosmos2_data['nh_lo_err']
chandra_cosmos2_nh_up_err = chandra_cosmos2_data['nh_up_err']
chandra_cosmos2_Fx_hard = chandra_cosmos2_data['flux_210']
chandra_cosmos2_Fx_soft = chandra_cosmos2_data['flux_052']
chandra_cosmos2_Fx_full = chandra_cosmos2_data['flux_0510']

chandra_cosmos_ct_Lx_hard_obs = np.asarray([10**i for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_full_obs = np.asarray([(10**i)*1.64 for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_hard = np.asarray([10**i for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_Lx_full = np.asarray([(10**i)*1.64 for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_nh = chandra_cosmos_ct_data['nh2_22']

# Gather DEIMOS spec redshifts to redshift array
for i in range(len(chandra_cosmos_z)):
    ind = np.where(deimos_ID == chandra_cosmos_phot_id[i])[0]
    if len(ind) > 0:
        if deimos_z_spec[ind][0] > 0:
            chandra_cosmos_z[i] = deimos_z_spec[ind][0]
        else:
            continue
    # if chandra_cosmos_z[i] < 0:
    #     chandra_cosmos_z[i] = chandra_cosmos_z_phot[i]
    else:
        continue

# Gather Luminosity absorption corrections from the Chandra COSMOS cat1 and correct the luminositeis
abs_corr_use_h = []
abs_corr_use_s = []
abs_corr_use_f = []
for i in range(len(chandra_cosmos_abs_corr_f)):
    if chandra_cosmos_abs_corr_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_s[i])

    elif chandra_cosmos_abs_corr_up_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_up_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_up_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_up_s[i])

    elif chandra_cosmos_abs_corr_lo_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_lo_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_lo_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_lo_s[i])

    else:
        print('NO GOOD ABSORPTION CORRECTION DATA')
     
abs_corr_use_h = np.asarray(abs_corr_use_h)
abs_corr_use_s = np.asarray(abs_corr_use_s)
abs_corr_use_f = np.asarray(abs_corr_use_f)

chandra_cosmos_Lx_hard /= abs_corr_use_h
chandra_cosmos_Lx_soft /= abs_corr_use_s
chandra_cosmos_Lx_full /= abs_corr_use_f

chandra_cosmos_Nh = []
check = []
for i in range(len(chandra_cosmos_Lx_full)):
    ind = np.where(chandra_cosmos2_xid == chandra_cosmos_xid[i])[0]
    ind_ct = np.where(chandra_cosmos_ct_xid == chandra_cosmos_xid[i])[0]

    if len(ind_ct) > 0:
        chandra_cosmos_Nh.append(chandra_cosmos_ct_nh[ind_ct][0])
        chandra_cosmos_Lx_hard[i] = chandra_cosmos_ct_Lx_hard[ind_ct]
        chandra_cosmos_Lx_full[i] = chandra_cosmos_ct_Lx_full[ind_ct]
        check.append(3)

    elif len(ind) > 0:
        chandra_cosmos_Lx_hard[i] = chandra_cosmos2_Lx_hard[ind]
        chandra_cosmos_Lx_full[i] = chandra_cosmos2_Lx_full[ind]
        if chandra_cosmos2_nh_lo_err[ind][0] == -99.:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]+chandra_cosmos2_nh_up_err[ind][0])
            check.append(2.5)
        else:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0])
            check.append(2)
    else:
        if chandra_cosmos_nh[i] == -99.:
            if chandra_cosmos_nh_lo[i] != -99.:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
            else:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
        else:    
            chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
        check.append(1)
chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E22
check = np.asarray(check)

# print(chandra_cosmos_Nh)
# for i in range(len(chandra_cosmos_Lx_full)):
#     if chandra_cosmos_nh[i] == -99.:
#         if chandra_cosmos_nh_lo[i] != -99.:
#             chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
#         else:
#             chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
#     else:    
#         chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
# chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E2


print('COSMOS All Lx cat: ',len(chandra_cosmos_Lx_full))


# Select COSMOS sources from the Chandra catalog based on conditions listed at the top of the file 
cosmos_condition = (chandra_cosmos_z > z_min) & (chandra_cosmos_z <= z_max) & (np.log10(chandra_cosmos_Lx_full) >= Lx_min) & (np.log10(chandra_cosmos_Lx_full) <= Lx_max) & (chandra_cosmos_phot_id != -99.)

chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_condition]
chandra_cosmos_xid = chandra_cosmos_xid[cosmos_condition]
chandra_cosmos_RA = chandra_cosmos_RA[cosmos_condition]
chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_condition]
chandra_cosmos_z = chandra_cosmos_z[cosmos_condition]
chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_condition]
chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_condition]
chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_condition]
chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_condition]
chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_condition]
chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_condition]
chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_condition]
chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_condition]
print('COSMOS Lx z: ', len(chandra_cosmos_phot_id))


# Select COSMOS sources from the Chandra catalog that have spectral types
cosmos_spec_type_cond = np.logical_or(chandra_cosmos_spec_type == 1, chandra_cosmos_spec_type == 2)

# chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_spec_type_cond]
# chandra_cosmos_xid = chandra_cosmos_xid[cosmos_spec_type_cond]
# chandra_cosmos_RA = chandra_cosmos_RA[cosmos_spec_type_cond]
# chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_spec_type_cond]
# chandra_cosmos_z = chandra_cosmos_z[cosmos_spec_type_cond]
# chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_spec_type_cond]
# chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_spec_type_cond]
# chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_spec_type_cond]
# chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_spec_type_cond]
# chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_spec_type_cond]
# chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_spec_type_cond]
# chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_spec_type_cond]
# chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_spec_type_cond]
# print('COSMOS Lx z spec type: ',len(chandra_cosmos_phot_id))

############### Subset match ###############
'''
subset_id = np.asarray([262286,673312,818825,859825,950643,215847,385450,385450,935830])

cosmos_subset_ix, cosmos_supbset_iy = match(chandra_cosmos_phot_id,subset_id)

chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_subset_ix]
chandra_cosmos_xid = chandra_cosmos_xid[cosmos_subset_ix]
chandra_cosmos_RA = chandra_cosmos_RA[cosmos_subset_ix]
chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_subset_ix]
chandra_cosmos_z = chandra_cosmos_z[cosmos_subset_ix]
chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_subset_ix]
chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_subset_ix]
chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_subset_ix]
chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_subset_ix]
chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_subset_ix]
chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_subset_ix]
chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_subset_ix]
chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_subset_ix]
'''
############## ############## ##############

# Match data from COSMOS Chandra catalog to data from the COSMOS2020 photometry catalog 
# cosmos_ix, cosmos_iy = match(chandra_cosmos_xid, cosmos_xid) # Match Chandra and COSMOS data with X-ray id
cosmos_ix, cosmos_iy = match(chandra_cosmos_phot_id, cosmos_laigle_id)

cosmos_laigle_id_match = cosmos_laigle_id[cosmos_iy]
chandra_cosmos_phot_id_match = chandra_cosmos_phot_id[cosmos_ix]
chandra_cosmos_xid_match = chandra_cosmos_xid[cosmos_ix]
chandra_cosmos_RA_match = chandra_cosmos_RA[cosmos_ix]
chandra_cosmos_DEC_match = chandra_cosmos_DEC[cosmos_ix]
chandra_cosmos_z_match = chandra_cosmos_z[cosmos_ix]
chandra_cosmos_Fx_full_match = chandra_cosmos_Fx_full[cosmos_ix]
chandra_cosmos_Fx_hard_match = chandra_cosmos_Fx_hard[cosmos_ix]
chandra_cosmos_Fx_soft_match = chandra_cosmos_Fx_soft[cosmos_ix]
chandra_cosmos_Lx_full_match = chandra_cosmos_Lx_full[cosmos_ix]
chandra_cosmos_Lx_hard_match = chandra_cosmos_Lx_hard[cosmos_ix]
chandra_cosmos_Lx_soft_match = chandra_cosmos_Lx_soft[cosmos_ix]
chandra_cosmos_spec_type_match = chandra_cosmos_spec_type[cosmos_ix]
chandra_cosmos_Nh_match = chandra_cosmos_Nh[cosmos_ix]
print('COSMOS phot match: ', len(chandra_cosmos_phot_id_match))

chandra_cosmos_Fx_hard_match_mjy = chandra_cosmos_Fx_hard_match*4.136E8/(10-2)
chandra_cosmos_Fx_soft_match_mjy = chandra_cosmos_Fx_soft_match*4.136E8/(2-0.5)
chandra_cosmos_Fx_full_match_mjy = chandra_cosmos_Fx_full_match*4.136E8/(10-0.5)

cosmos_nan_array = np.zeros(np.shape(cosmos_laigle_id_match))

cosmos_flux_array = np.array([
    chandra_cosmos_Fx_hard_match_mjy*1000,chandra_cosmos_Fx_soft_match_mjy*1000,
    cosmos_nan_array,
    cosmos_data['GALEX_FUV_FLUX'][cosmos_iy],
    cosmos_data['GALEX_NUV_FLUX'][cosmos_iy],
    cosmos_data['CFHT_u_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_g_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_r_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_i_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_z_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_y_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_J_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_H_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_Ks_FLUX_APER2'][cosmos_iy],
    cosmos_data['SPLASH_CH1_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH2_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH3_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH4_FLUX'][cosmos_iy],
    cosmos_data['FIR_24_FLUX'][cosmos_iy],
    cosmos_data['FIR_100_FLUX'][cosmos_iy],
    cosmos_data['FIR_160_FLUX'][cosmos_iy],
    cosmos_data['FIR_250_FLUX'][cosmos_iy],
    cosmos_data['FIR_350_FLUX'][cosmos_iy],
    cosmos_data['FIR_500_FLUX'][cosmos_iy]
])

cosmos_flux_err_array = np.array([
    chandra_cosmos_Fx_hard_match_mjy*1000*0.2, chandra_cosmos_Fx_soft_match_mjy*1000*0.2,
    cosmos_nan_array,
    cosmos_data['GALEX_FUV_FLUXERR'][cosmos_iy],
    cosmos_data['GALEX_NUV_FLUXERR'][cosmos_iy],
    cosmos_data['CFHT_u_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_g_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_r_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_i_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_z_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_y_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_J_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_H_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_Ks_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['SPLASH_CH1_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH2_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH3_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH4_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_24_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_100_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_160_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_250_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_350_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_500_FLUXERR'][cosmos_iy]
])

cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T
###############################################################################


###############################################################################
###############################################################################
########################### Read in Stripe82X files ###########################

eboss = fits.open('/Users/connor_auge/Desktop/lamassa2019.fit')
eboss_data = eboss[1].data
eboss.close()

lamassa = fits.open(path+'S82X_catalog_with_photozs_unique_Xraysrcs_likely_cps_w_mbh.fits')
lamassa_cols = lamassa[1].columns
lamassa_data = lamassa[1].data
lamassa.close()

peca = ascii.read('/Users/connor_auge/Desktop/s82_spec_results/Auge_spec_results_safe.txt')
unwise = ascii.read('/Users/connor_auge/Desktop/unwise_matches.csv')

peca_ID = np.asarray(peca['ID'])
peca_Lx_full = np.asarray(peca['lumin_f'])
peca_Lx_hard = np.asarray(peca['lumin_h'])
peca_Lx_full_obs = np.asarray(peca['lumin_of'])
peca_Nh = np.asarray(peca['nh'])

unwise_ID = np.asarray(unwise['ID'])
# unwise_W1 = mag_to_flux(np.asarray(unwise['unW1']),'W1')*1E6
# unwise_W1_err = magerr_to_fluxerr(np.asarray(unwise['unW1']),np.asarray(unwise['unW1_err']),'W1')*1E6
# unwise_W2 = mag_to_flux(np.asarray(unwise['unW2']),'W2')*1E6
# unwise_W2_err = magerr_to_fluxerr(np.asarray(unwise['unW2']),np.asarray(unwise['unW2_err']),'W2')*1E6
# unwise_W3 = mag_to_flux(np.asarray(unwise['unW3']),'W3')*1E6
# unwise_W3_err = magerr_to_fluxerr(np.asarray(unwise['unW3']),np.asarray(unwise['unW3_err']),'W3')*1E6
# unwise_W4 = mag_to_flux(np.asarray(unwise['unW4']),'W4')*1E6
# unwise_W4_err = magerr_to_fluxerr(np.asarray(unwise['unW4']),np.asarray(unwise['unW4_err']),'W4')*1E6
unwise_W1 = np.asarray(unwise['unW1'])
unwise_W1_err = np.asarray(unwise['unW1_err'])
unwise_W2 = np.asarray(unwise['unW2'])
unwise_W2_err = np.asarray(unwise['unW2_err'])
unwise_W3 = np.asarray(unwise['unW3'])
unwise_W3_err = np.asarray(unwise['unW3_err'])
unwise_W4 = np.asarray(unwise['unW4'])
unwise_W4_err = np.asarray(unwise['unW4_err'])


lamassa_id = lamassa_data['msid']
lamassa_id2 = lamassa_data['rec_no']
lamassa_id3 = lamassa_data['OBSID']
lamassa_z = lamassa_data['SPEC_Z']
lamassa_z_phot = lamassa_data['PHOTO_Z']

lamassa_id_use = []
lamassa_z_use = []
for i in range(len(lamassa_id)):
    if lamassa_id[i] == 0:
        lamassa_id_use.append(lamassa_id2[i])
    else:
       lamassa_id_use.append(lamassa_id[i])
    # if lamassa_z[i] <=0:
    #     lamassa_z_use.append(lamassa_z_phot[i])
    # else:
    #     lamassa_z_use.append(lamassa_z[i])
lamassa_id_use = np.asarray(lamassa_id_use)
# lamassa_z = np.asarray(lamassa_z_use)

# ID1 = lamassa_data['MSID']
# ID2 = lamassa_data['REC_NO']
# ID3 = lamassa_data['OBSID']
# lamassa_z = lamassa_data['spec_z']
# lamassa_ra = lamassa_data['XRAY_RA']
# lamassa_id = []
# for i in range(len(ID1)):
#     if ID1[i] == 0:
#         lamassa_id.append(ID2[i])
#     else:
#         lamassa_id.append(ID1[i])
# lamassa_id_use = np.asarray(lamassa_id)


lamassa_ra = lamassa_data['XRAY_RA']
lamassa_dec = lamassa_data['XRAY_DEC']
lamassa_cat = lamassa_data['XRAY_SRC']

s82x_Lx_sp_full = np.asarray([10**i for i in lamassa_data['FULL_LUM']])
s82x_Lx_sp_hard = np.asarray([10**i for i in lamassa_data['HARD_LUM']])
s82x_Fx_hard = lamassa_data['HARD_FLUX']
s82x_Fx_soft = lamassa_data['SOFT_FLUX']
s82x_Fx_full = lamassa_data['FULL_FLUX']
# s82x_W1 = mag_to_flux(lamassa_data['W1'],'W1')*1E6
# s82x_W2 = mag_to_flux(lamassa_data['W2'],'W2')*1E6
# s82x_W3 = mag_to_flux(lamassa_data['W3'],'W3')*1E6
# s82x_W4 = mag_to_flux(lamassa_data['W4'],'W4')*1E6
# s82x_W1_err = magerr_to_fluxerr(lamassa_data['W1'],lamassa_data['W1_err'],'W1')*1E6
# s82x_W2_err = magerr_to_fluxerr(lamassa_data['W2'],lamassa_data['W2_err'],'W2')*1E6
# s82x_W3_err = magerr_to_fluxerr(lamassa_data['W3'],lamassa_data['W3_err'],'W3')*1E6
# s82x_W4_err = magerr_to_fluxerr(lamassa_data['W4'],lamassa_data['W4_err'],'W4')*1E6
# s82x_ch1_spies = mag_to_flux(lamassa_data['CH1_SPIES'],'Ch1Spies')*1E6
# s82x_ch2_spies = mag_to_flux(lamassa_data['CH2_SPIES'],'Ch2Spies')*1E6
# s82x_ch1_shela = mag_to_flux(lamassa_data['CH1_SHELA'],'Ch1Spies')*1E6
# s82x_ch2_shela = mag_to_flux(lamassa_data['CH2_SHELA'],'Ch2Spies')*1E6
# s82x_ch1_spies_err = magerr_to_fluxerr(lamassa_data['CH1_SPIES'],lamassa_data['CH1_SPIES_ERR'], 'Ch1Spies')*1E6
# s82x_ch2_spies_err = magerr_to_fluxerr(lamassa_data['CH2_SPIES'],lamassa_data['CH2_SPIES_ERR'], 'Ch2Spies')*1E6
# s82x_ch1_shela_err = magerr_to_fluxerr(lamassa_data['CH1_SHELA'],lamassa_data['CH1_SHELA_ERR'],'Ch1Spies')*1E6
# s82x_ch2_shela_err = magerr_to_fluxerr(lamassa_data['CH2_SHELA'],lamassa_data['CH2_SHELA_ERR'],'Ch2Spies')*1E6
s82x_spec_class = lamassa_data['SPEC_CLASS']

s82x_W1 = lamassa_data['W1']
s82x_W2 = lamassa_data['W2']
s82x_W3 = lamassa_data['W3']
s82x_W4 = lamassa_data['W4']
s82x_W1_err = lamassa_data['W1_err']
s82x_W2_err = lamassa_data['W2_err']
s82x_W3_err = lamassa_data['W3_err']
s82x_W4_err = lamassa_data['W4_err']

s82x_spec_class[s82x_spec_class == 'QSO'] = '1'
s82x_spec_class[s82x_spec_class == 'QSO_BAL'] = '1'
s82x_spec_class[s82x_spec_class == 'QSO(BA'] = '1'
s82x_spec_class[s82x_spec_class == 'GALAXY'] = '2'
s82x_spec_class[s82x_spec_class == 'AGN'] = '1'
s82x_spec_class[s82x_spec_class == 'N/A'] = '3'
s82x_spec_class[s82x_spec_class == 'STAR'] = '3'
s82x_spec_class[s82x_spec_class == 'NELG'] = '3'
s82x_spec_class[s82x_spec_class == '    '] = '3'
s82x_spec_class[s82x_spec_class == ''] = '3'
s82x_spec_class = np.asarray(s82x_spec_class)


for i in range(len(lamassa_id_use)):
    ind = np.where(unwise_ID == lamassa_id_use[i])[0]
    if len(ind) > 0:
        if np.isnan(unwise_W3[ind]):
            continue
        elif unwise_W3[ind] <= 0.0:
            continue
        elif np.isnan(unwise_W3[ind]):
            continue
        elif (magerr_to_fluxerr(unwise_W3[ind], unwise_W3_err[ind], 'W3', AB=True)/mag_to_flux(unwise_W3[ind], 'W3')) > 0.4:
            continue
        else:
            s82x_W3[i] = unwise_W3[ind][0]
            s82x_W3_err[i] = unwise_W3_err[ind][0]

        if np.isnan(unwise_W4[ind]):
            continue
        elif unwise_W4[ind] <= 0.0:
            continue
        elif np.isnan(unwise_W4[ind]):
            continue
        elif (magerr_to_fluxerr(unwise_W4[ind], unwise_W4_err[ind], 'W4', AB=True)/mag_to_flux(unwise_W4[ind], 'W4')) > 0.4:
            continue
        else:
            s82x_W4[i] = unwise_W4[ind][0]
            s82x_W4_err[i] = unwise_W4_err[ind][0]

# s82x_W1 = mag_to_flux(s82x_W1,'W1')*1E6
# s82x_W2 = mag_to_flux(s82x_W2,'W2')*1E6
# s82x_W3 = mag_to_flux(s82x_W3,'W3')*1E6
# s82x_W4 = mag_to_flux(s82x_W4,'W4')*1E6

# s82x_W1_err = magerr_to_fluxerr(s82x_W1,s82x_W1_err,'W1')*1E6
# s82x_W2_err = magerr_to_fluxerr(s82x_W2,s82x_W2_err,'W2')*1E6
# s82x_W3_err = magerr_to_fluxerr(s82x_W3,s82x_W3_err,'W3')*1E6
# s82x_W4_err = magerr_to_fluxerr(s82x_W4,s82x_W4_err,'W4')*1E6

s82x_Nh = []
for i in range(len(lamassa_id_use)):
    ind = np.where(peca_ID == lamassa_id_use[i])[0]
    if len(ind) > 0:
        s82x_Lx_sp_full[i] = peca_Lx_full[ind]
        # s82x_Lx_use[i] = peca_Lx_full[ind]
        s82x_Nh.append(peca_Nh[ind][0])

    else:
        s82x_Nh.append(0.0)

s82x_Nh = np.asarray(s82x_Nh)

# s82x_irac1, s82x_irac2 = [], []
# s82x_irac1_err, s82x_irac2_err = [], []
# for i in range(len(lamassa_id_use)):
#     if s82x_ch1_spies[i] > 0:
#         s82x_irac1.append(s82x_ch1_spies[i])
#         s82x_irac1_err.append(s82x_ch1_spies_err[i])
#     else:
#         s82x_irac1.append(s82x_ch1_shela[i])
#         s82x_irac1_err.append(s82x_ch1_shela_err[i])
#     if s82x_ch2_spies[i] > 0:
#         s82x_irac2.append(s82x_ch2_spies[i])
#         s82x_irac2_err.append(s82x_ch2_spies_err[i])
#     else:
#         s82x_irac2.append(s82x_ch2_shela[i])
#         s82x_irac2_err.append(s82x_ch2_shela_err[i])

# s82x_irac1, s82x_irac2 = np.asarray(s82x_irac1), np.asarray(s82x_irac2)
# s82x_irac1_err, s82x_irac2_err = np.asarray(s82x_irac1_err), np.asarray(s82x_irac2_err)




# mir1, mir2 = [], []
# mir1_err, mir2_err = [], []
# for i in range(len(lamassa_id_use)):
#     # if s82x_irac1[i] > 0:
#     #     mir1.append(s82x_irac1[i])
#     #     mir1_err.append(s82x_irac1_err[i])
#     # else: 
#         mir1.append(s82x_W1[i])
#         mir1_err.append(s82x_W1_err[i])

#     # if s82x_irac2[i] > 0:
#     #     mir2.append(s82x_irac2[i])
#     #     mir2_err.append(s82x_irac2_err[i])
#     # else:
#         mir2.append(s82x_W2[i])
#         mir2_err.append(s82x_W2_err[i])

# mir1, mir2 = np.asarray(mir1), np.asarray(mir2)
# mir1_err, mir2_err = np.asarray(mir1_err), np.asarray(mir2_err)

# for i in range(len(s82x_irac2)):
#     print(s82x_irac2[i], s82x_W2[i], mir2[i])


print('S82X All: ', len(lamassa_id_use))


# Select S82X sources from the Ananna2017 catalog based on conditions listed at the top of the file
s82x_condition = (lamassa_z > z_min) & (lamassa_z < z_max) & (np.logical_and(lamassa_ra >= 13, lamassa_ra <= 37)) & (
    np.logical_and(np.log10(s82x_Lx_sp_full) <= Lx_max, np.log10(s82x_Lx_sp_full) >= Lx_min))


lamassa_id_use = lamassa_id_use[s82x_condition]
lamassa_cat = lamassa_cat[s82x_condition]
s82x_z_sp = lamassa_z[s82x_condition]
lamassa_ra = lamassa_ra[s82x_condition]
lamassa_dec = lamassa_dec[s82x_condition]
s82x_Lx_sp_full = s82x_Lx_sp_full[s82x_condition]
s82x_Lx_sp_hard = s82x_Lx_sp_hard[s82x_condition]
s82x_Fx_hard = s82x_Fx_hard[s82x_condition]
s82x_Fx_soft = s82x_Fx_soft[s82x_condition]
s82x_Fx_full = s82x_Fx_full[s82x_condition]
s82x_Nh = s82x_Nh[s82x_condition]
s82x_spec_class = s82x_spec_class[s82x_condition]

print('S82X Lx z coords: ', len(lamassa_id_use))


print('S82X match: ', len(lamassa_id_use))


s82x_Fx_hard_match_mjy = s82x_Fx_hard*4.136E8/(10-2)
s82x_Fx_soft_match_mjy = s82x_Fx_soft*4.136E8/(2-0.5)
s82x_Fx_full_match_mjy = s82x_Fx_full*4.136E8/(10-0.5)
s82x_Fx_hard_err_match_mjy = s82x_Fx_hard_match_mjy*0.2
s82x_Fx_soft_err_match_mjy = s82x_Fx_soft_match_mjy*0.2
s82x_Fx_full_err_match_mjy = s82x_Fx_full_match_mjy*0.2


# Create nan array with length == to the number of sources to be input to the photometry array
s82x_nan_array = np.zeros(np.shape(lamassa_id_use))
s82x_nan_array[s82x_nan_array == 0] = np.nan

# Flux array for Stripe82X sources in Jy
s82x_flux_array = np.array([
	s82x_Fx_hard_match_mjy*1000, s82x_Fx_soft_match_mjy*1000,
	s82x_nan_array,
	mag_to_flux(lamassa_data['mag_FUV'][s82x_condition], 'FUV')*1E6,
	mag_to_flux(lamassa_data['mag_NUV'][s82x_condition], 'FUV')*1E6,
	mag_to_flux(lamassa_data['u'][s82x_condition], 'sloan_u')*1E6,
	mag_to_flux(lamassa_data['g'][s82x_condition], 'sloan_g')*1E6,
	mag_to_flux(lamassa_data['r'][s82x_condition], 'sloan_r')*1E6,
	mag_to_flux(lamassa_data['i'][s82x_condition], 'sloan_i')*1E6,
	mag_to_flux(lamassa_data['z'][s82x_condition], 'sloan_z')*1E6,
	mag_to_flux(lamassa_data['JVHS'][s82x_condition], 'JVHS')*1E6,
	mag_to_flux(lamassa_data['HVHS'][s82x_condition], 'HVHS')*1E6,
	mag_to_flux(lamassa_data['KVHS'][s82x_condition], 'HVHS')*1E6,
    # s82x_irac1[s82x_condition],
	# s82x_W1[s82x_condition],
	# s82x_W2[s82x_condition],
    # s82x_irac2[s82x_condition],
    # mir1[s82x_condition],
    # mir2[s82x_condition],
	# s82x_W3[s82x_condition],
	# s82x_W4[s82x_condition],
    mag_to_flux(s82x_W1[s82x_condition],'W1')*1E6,
    mag_to_flux(s82x_W2[s82x_condition],'W2')*1E6,
    mag_to_flux(s82x_W3[s82x_condition],'W3')*1E6,
    mag_to_flux(s82x_W4[s82x_condition],'W4')*1E6,    
	s82x_nan_array,
	lamassa_data['F250'][s82x_condition]*1000,
	lamassa_data['F350'][s82x_condition]*1000,
	lamassa_data['F500'][s82x_condition]*1000
])


s82x_flux_err_array = np.array([
	s82x_Fx_hard_err_match_mjy*1000, s82x_Fx_soft_err_match_mjy*1000,
	s82x_nan_array,
	magerr_to_fluxerr(lamassa_data['mag_FUV'][s82x_condition],lamassa_data['magerr_FUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['mag_NUV'][s82x_condition],lamassa_data['magerr_NUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['u'][s82x_condition],lamassa_data['u_err'][s82x_condition], 'sloan_u')*1E6,
	magerr_to_fluxerr(lamassa_data['g'][s82x_condition],lamassa_data['g_err'][s82x_condition], 'sloan_g')*1E6,
	magerr_to_fluxerr(lamassa_data['r'][s82x_condition],lamassa_data['r_err'][s82x_condition], 'sloan_r')*1E6,
	magerr_to_fluxerr(lamassa_data['i'][s82x_condition],lamassa_data['i_err'][s82x_condition], 'sloan_i')*1E6,
	magerr_to_fluxerr(lamassa_data['z'][s82x_condition],lamassa_data['z_err'][s82x_condition], 'sloan_z')*1E6,
	magerr_to_fluxerr(lamassa_data['JVHS'][s82x_condition],lamassa_data['JVHS_err'][s82x_condition], 'JVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['HVHS'][s82x_condition],lamassa_data['HVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['KVHS'][s82x_condition],lamassa_data['KVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
    # s82x_irac1_err[s82x_condition],
	# s82x_W1_err[s82x_condition],
	# s82x_W2_err[s82x_condition],
    # s82x_irac1_err[s82x_condition],
    # mir1_err[s82x_condition],
    # mir2_err[s82x_condition],
	# s82x_W3_err[s82x_condition],
	# s82x_W4_err[s82x_condition],
    magerr_to_fluxerr(s82x_W1[s82x_condition],s82x_W1_err[s82x_condition],'W1')*1E6,
    magerr_to_fluxerr(s82x_W2[s82x_condition],s82x_W2_err[s82x_condition],'W2')*1E6,
    magerr_to_fluxerr(s82x_W3[s82x_condition],s82x_W3_err[s82x_condition],'W3')*1E6,
    magerr_to_fluxerr(s82x_W4[s82x_condition],s82x_W4_err[s82x_condition],'W4')*1E6,
    s82x_nan_array,
	lamassa_data['F250_err'][s82x_condition]*1000,
	lamassa_data['F350_err'][s82x_condition]*1000,
	lamassa_data['F500_err'][s82x_condition]*1000
])

s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T



###############################################################################


###############################################################################
###############################################################################
############################ Read in GOODS-N files ############################

xue = fits.open(path+'XueCDFN.fit')
xue_data = xue[1].data
xue.close()

goodsN = ascii.read('/Users/connor_auge/Research/REU/2021/GOODS-N_phot2.txt')

xue_id = xue_data['seq']
xue_xRA = xue_data['RAJ2000']
xue_xDEC = xue_data['DEJ2000']
xue_Lx = xue_data['Lx']*1.180  # convert the 0.5-7keV Lx to 0.5-10kev
xue_Lx_hard = xue_data['Lx']*0.721 # convert the 0.5-7keV Lx to 2-10keV
xue_Fx_full = xue_data['FFlux']
xue_Fx_hard = xue_data['HFlux']
xue_Fx_soft = xue_data['SFlux']
xue_z = xue_data['zspec']
xue_offset = xue_data['C-XOff']
# xue_z = xue_data['zadopt']

# Select GOODS-N sources from the Xue catalog based on conditions listed at the top of the file
goodsN_condition = (np.log10(xue_Lx) >= Lx_min) & (np.log10(xue_Lx) <= Lx_max) & (xue_z > z_min) & (xue_z <= z_max) & (xue_z != 0.0)
print('GOODS-N All:', len(xue_id))

xue_xRA = xue_xRA[goodsN_condition]
xue_xDEC = xue_xDEC[goodsN_condition]
xue_Lx = xue_Lx[goodsN_condition]
xue_Lx_hard = xue_Lx_hard[goodsN_condition]
xue_Fx_full = xue_Fx_full[goodsN_condition]
xue_Fx_hard = xue_Fx_hard[goodsN_condition]
xue_Fx_soft = xue_Fx_soft[goodsN_condition]
xue_z = xue_z[goodsN_condition]
xue_id = xue_id[goodsN_condition]
xue_offset = xue_offset[goodsN_condition]

goodsN_phot_RA = np.asarray(goodsN['ALPHA_GR_DEC_ORDER'])
goodsN_phot_DEC = np.asarray(goodsN['DELTA_GR_DEC_ORDER'])
goodsN_phot_id = np.asarray(goodsN['object'])

print('GOODS-N Lx z:', len(xue_id))

# goodsN_coords_file = open('/Users/connor_auge/Research/Disertation/catalogs/goodsN_coords.txt','w')
# goodsN_coords_file.writelines('#ID,RA,Dec\n')
# for i in range(len(xue_id)):
#     goodsN_coords_file.writelines('%s,%s,%s\n' % (xue_id[i], xue_xRA[i], xue_xDEC[i]))
# goodsN_coords_file.close()

# Match the X-ray catalog and photometry catalog based on their corrdinates (5 arcsecond sep)
cat_match_dist = 1.5  # arcseconds
catalog = SkyCoord(ra=goodsN_phot_RA*u.degree, dec=goodsN_phot_DEC*u.degree)
c1 = SkyCoord(ra=xue_xRA, dec=xue_xDEC, unit=(u.hourangle, u.deg))
idx, d2d, d3d = c1.match_to_catalog_sky(catalog)
goodsN_idx_match = idx[d2d.arcsecond <= cat_match_dist]

goodsN_phot_id_match = goodsN_phot_id[goodsN_idx_match]
xue_xRA_match = xue_xRA[d2d.arcsecond < cat_match_dist]
xue_xDEC_match = xue_xDEC[d2d.arcsecond < cat_match_dist]
xue_z_match = xue_z[d2d.arcsecond < cat_match_dist]
xue_Lx_match = xue_Lx[d2d.arcsecond < cat_match_dist]
xue_Lx_hard_match = xue_Lx_hard[d2d.arcsecond < cat_match_dist]
xue_Fx_full_match = xue_Fx_full[d2d.arcsecond < cat_match_dist]
xue_Fx_hard_match = xue_Fx_hard[d2d.arcsecond < cat_match_dist]
xue_Fx_soft_match = xue_Fx_soft[d2d.arcsecond < cat_match_dist]
xue_id_match = xue_id[d2d.arcsecond < cat_match_dist]

print('GOODS-N match:', len(xue_id_match))

xue_Fx_hard_match_mjy = xue_Fx_hard_match*4.136E8/(10-2)
xue_Fx_soft_match_mjy = xue_Fx_soft_match*4.136E8/(2-0.5)
xue_Fx_full_match_mjy = xue_Fx_full_match*4.136E8/(10-0.5)
xue_Fx_hard_match_err_mjy = xue_Fx_hard_match_mjy*0.2
xue_Fx_soft_match_err_mjy = xue_Fx_soft_match_mjy*0.2
xue_Fx_full_match_err_mjy = xue_Fx_full_match_mjy*0.2

goodsN_Nh_match = np.zeros(np.shape(xue_Lx_match))

# Create nan array with length == to the number of sources to be input to the photometry array
goodsN_nan_array = np.zeros(np.shape(goodsN_phot_id_match))
goodsN_nan_array[goodsN_nan_array == 0] = np.nan

goodsN_flux_array = np.asarray([xue_Fx_hard_match_mjy*1000, xue_Fx_soft_match_mjy*1000,
                                goodsN_nan_array,
                                goodsN['GALEX_fuv'][goodsN_idx_match],
                                goodsN['GALEX_nuv'][goodsN_idx_match],
                                goodsN['TFIT_KPNO_U'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F435W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F606W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F775W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F814W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F105W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F125W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F140W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F160W'][goodsN_idx_match],
                                goodsN['TFIT_MOIRCS_K'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac36'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac45'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac58'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac80'][goodsN_idx_match],
                                goodsN['mips24_cryo'][goodsN_idx_match],
                                goodsN['mips70_cryo'][goodsN_idx_match],
                                goodsN['pacs100_merged'][goodsN_idx_match],
                                goodsN['pacs160_merged'][goodsN_idx_match],
                                goodsN['spire250_merged'][goodsN_idx_match],
                                goodsN['spire350_merged'][goodsN_idx_match],
                                goodsN['spire500_merged'][goodsN_idx_match]
                                ])

goodsN_flux_err_array = np.asarray([xue_Fx_hard_match_err_mjy*1000, xue_Fx_soft_match_err_mjy*1000,
                                    goodsN_nan_array,
                                    goodsN['err_GALEX_fuv'][goodsN_idx_match],
                                    goodsN['err_GALEX_nuv'][goodsN_idx_match],
                                    goodsN['err_TFIT_KPNO_U'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F435W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F606W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F775W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F814W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F105W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F125W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F140W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F160W'][goodsN_idx_match],
                                    goodsN['err_TFIT_MOIRCS_K'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac36'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac45'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac58'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac80'][goodsN_idx_match],
                                    goodsN['err_mips24_cryo'][goodsN_idx_match],
                                    goodsN['err_mips70_cryo'][goodsN_idx_match],
                                    goodsN['err_pacs100_merged'][goodsN_idx_match],
                                    goodsN['err_pacs160_merged'][goodsN_idx_match],
                                    goodsN['err_spire250_merged'][goodsN_idx_match],
                                    goodsN['err_spire350_merged'][goodsN_idx_match],
                                    goodsN['err_spire500_merged'][goodsN_idx_match]
                                    ])

goodsN_flux_array = goodsN_flux_array.T
goodsN_flux_err_array = goodsN_flux_err_array.T


###############################################################################

###############################################################################
###############################################################################
############################ Read in GOODS-S files ############################
luo = fits.open(path+'LuoCDFS.fit')
luo_data = luo[1].data
luo.close()

goodsS = ascii.read('/Users/connor_auge/Research/REU/2021/GOODS-S_phot2.txt')

luo_id = luo_data['seq']
# luo_xRA = luo_data['RACdeg']
# luo_xDEC = luo_data['DECdeg']
luo_xRA = luo_data['RAJ2000']
luo_xDEC = luo_data['DEJ2000']
luo_Lx = luo_data['Lxc']*1.180 # convert the 0.5-7kev Lx to 0.5-10kev
luo_Lx_hard = luo_data['Lxc']*0.721  # convert the 0.5-7kev Lx to 2-10kev
luo_Fx_full = luo_data['FFB']
luo_Fx_hard = luo_data['FHB']
luo_Fx_soft = luo_data['FSB']
luo_z = luo_data['zspec']
goodsS_Nh = luo_data['Nh']
# luo_z = luo_data['zf']

print('GOODS-S All: ', len(luo_id))

# Select GOODS-S sources from the Xue catalog based on conditions listed at the top of the file 
goodsS_condition = (np.log10(luo_Lx) >= Lx_min) & (np.log10(luo_Lx) <= Lx_max) &(luo_z > z_min) & (luo_z <= z_max) & (luo_z != 0.0)

luo_id = luo_id[goodsS_condition]
luo_xRA = luo_xRA[goodsS_condition]
luo_xDEC = luo_xDEC[goodsS_condition]
luo_Lx = luo_Lx[goodsS_condition]
luo_Lx_hard = luo_Lx_hard[goodsS_condition]
luo_Fx_full = luo_Fx_full[goodsS_condition]
luo_Fx_hard = luo_Fx_hard[goodsS_condition]
luo_Fx_soft = luo_Fx_soft[goodsS_condition]
luo_z = luo_z[goodsS_condition]
goodsS_Nh = goodsS_Nh[goodsS_condition]

print('GOODS-S Lx z: ', len(luo_id))

goodsS_phot_RA = np.asarray(goodsS['ALPHA_GR_DEC_ORDER'])
goodsS_phot_DEC = np.asarray(goodsS['DELTA_GR_DEC_ORDER'])
goodsS_phot_id = np.asarray(goodsS['object'])

# goodsS_coords_file = open('/Users/connor_auge/Research/Disertation/catalogs/goodsS_coords.txt','w')
# goodsS_coords_file.writelines('#ID,RA,Dec\n')
# for i in range(len(luo_id)):
#     goodsS_coords_file.writelines('%s,%s,%s\n' % (luo_id[i], luo_xRA[i], luo_xDEC[i]))
# goodsS_coords_file.close()

# Match the X-ray catalog and photometry catalog based on their corrdinates (5 arcsecond sep)
cat_match_dist = 5 #arcseconds
catalog = SkyCoord(ra=goodsS_phot_RA*u.degree,dec=goodsS_phot_DEC*u.degree)
c1 = SkyCoord(ra=luo_xRA*u.degree, dec=luo_xDEC*u.degree)
idx, d2d, d3d = c1.match_to_catalog_sky(catalog)
goodsS_idx_match = idx[d2d.arcsecond < cat_match_dist]

goodsS_phot_id_match = goodsS_phot_id[goodsS_idx_match]
luo_id_match = luo_id[d2d.arcsecond < cat_match_dist]
luo_xRA_match = luo_xRA[d2d.arcsecond < cat_match_dist]
luo_xDEC_match = luo_xDEC[d2d.arcsecond < cat_match_dist]
luo_z_match = luo_z[d2d.arcsecond < cat_match_dist]
luo_Lx_match = luo_Lx[d2d.arcsecond < cat_match_dist]
luo_Lx_hard_match = luo_Lx_hard[d2d.arcsecond < cat_match_dist]
luo_Fx_full_match = luo_Fx_full[d2d.arcsecond < cat_match_dist]
luo_Fx_hard_match = luo_Fx_hard[d2d.arcsecond < cat_match_dist]
luo_Fx_soft_match = luo_Fx_soft[d2d.arcsecond < cat_match_dist]
goodsS_NH_match = goodsS_Nh[d2d.arcsecond < cat_match_dist]

print('GOODS-S match: ', len(luo_id_match))

luo_Fx_hard_match_mjy = luo_Fx_hard_match*4.136E8/(10-2)
luo_Fx_soft_match_mjy = luo_Fx_soft_match*4.136E8/(2-0.5)
luo_Fx_full_match_mjy = luo_Fx_full_match*4.136E8/(10-0.5)
luo_Fx_hard_match_err_mjy = luo_Fx_hard_match_mjy*0.2
luo_Fx_soft_match_err_mjy = luo_Fx_soft_match_mjy*0.2
luo_Fx_full_match_err_mjy = luo_Fx_full_match_mjy*0.2

goodsS_nan_array = np.zeros(np.shape(goodsS_phot_id_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsS_nan_array[goodsS_nan_array == 0] = np.nan

goodsS_flux_array = np.asarray([luo_Fx_hard_match_mjy*1000,luo_Fx_soft_match_mjy*1000,
	goodsS_nan_array,
	goodsS['TFIT_130701_VIMOS_U'][goodsS_idx_match],
	goodsS['TFIT_130701_ACS_b'][goodsS_idx_match], #435
	goodsS['TFIT_130701_ACS_v'][goodsS_idx_match], #606
	goodsS['TFIT_130701_ACS_i'][goodsS_idx_match], #775
	goodsS['TFIT_130701_ACS_f814w'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F098M'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F105W'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F125W'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F160W'][goodsS_idx_match],
	goodsS['TFIT_130701_ISAAC_Ks'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC36'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC45'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC58'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC80'][goodsS_idx_match],
	goodsS['mips24_cryo'][goodsS_idx_match],
	goodsS['mips70_cryo'][goodsS_idx_match],
	goodsS['pacs100_merged'][goodsS_idx_match],
	goodsS['pacs160_merged'][goodsS_idx_match],
	goodsS['spire250_merged'][goodsS_idx_match],
	goodsS['spire350_merged'][goodsS_idx_match],
	goodsS['spire500_merged'][goodsS_idx_match]
])

goodsS_flux_err_array = np.asarray([luo_Fx_hard_match_err_mjy*1000,luo_Fx_soft_match_err_mjy*1000,
	goodsS_nan_array,
	goodsS['err_TFIT_130701_VIMOS_U'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_b'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_v'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_i'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_f814w'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F098M'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F105W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F125W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F160W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ISAAC_Ks'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC36'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC45'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC58'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC80'][goodsS_idx_match],
	goodsS['err_mips24_cryo'][goodsS_idx_match],
	goodsS['err_mips70_cryo'][goodsS_idx_match],
	goodsS['err_pacs100_merged'][goodsS_idx_match],
	goodsS['err_pacs160_merged'][goodsS_idx_match],
	goodsS['err_spire250_merged'][goodsS_idx_match],
	goodsS['err_spire350_merged'][goodsS_idx_match],
	goodsS['err_spire500_merged'][goodsS_idx_match]
])

goodsS_flux_array = goodsS_flux_array.T
goodsS_flux_err_array = goodsS_flux_err_array.T


###############################################################################
###############################################################################
############################ Read in GOODS-N files 2 ##########################

goodsN_auge = fits.open(path+'GOODsN_full_cat.fits')
goodsN_auge_data = goodsN_auge[1].data
goodsN_auge.close()

goodsN_auge_ID = goodsN_auge_data['id_xray']
goodsN_auge_Lx = goodsN_auge_data['Lx']*1.180
goodsN_auge_Lx_hard = goodsN_auge_data['Lx']*0.721
goodsN_auge_z = goodsN_auge_data['z_spec']

goodsN_auge_condition = (np.log10(goodsN_auge_Lx) >= Lx_min) & (np.log10(goodsN_auge_Lx) <= Lx_max) &(goodsN_auge_z > z_min) & (goodsN_auge_z <= z_max) & (goodsN_auge_z != 0.0)

goodsN_auge_ID_match = goodsN_auge_ID[goodsN_auge_condition]
goodsN_auge_Lx_match = goodsN_auge_Lx[goodsN_auge_condition]
goodsN_auge_Lx_hard_match = goodsN_auge_Lx_hard[goodsN_auge_condition]
goodsN_auge_z_match = goodsN_auge_z[goodsN_auge_condition]

print('GOODS-N 2 match: ',len(goodsN_auge_ID_match))

goodsN_auge_Fx_hard_match_mjy = goodsN_auge_data['Fx_hard'][goodsN_auge_condition]*4.136E8/(10-2)
goodsN_auge_Fx_soft_match_mjy = goodsN_auge_data['Fx_soft'][goodsN_auge_condition]*4.136E8/(2-0.5)

goodsN_nan_array = np.zeros(np.shape(goodsN_auge_ID_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsN_nan_array[goodsN_nan_array == 0] = np.nan
goodsN_flux_array_auge = np.asarray([goodsN_auge_Fx_hard_match_mjy*1000, goodsN_auge_Fx_soft_match_mjy*1000,
	goodsN_nan_array,
    goodsN_auge_data['FUV'][goodsN_auge_condition],
	goodsN_auge_data['NUV'][goodsN_auge_condition],
    goodsN_auge_data['U'][goodsN_auge_condition],
	goodsN_auge_data['B'][goodsN_auge_condition], 
	goodsN_auge_data['V'][goodsN_auge_condition],
    goodsN_auge_data['F606W'][goodsN_auge_condition],
	goodsN_auge_data['R'][goodsN_auge_condition], 
	goodsN_auge_data['I'][goodsN_auge_condition],
    goodsN_auge_data['F814W'][goodsN_auge_condition],
	goodsN_auge_data['z'][goodsN_auge_condition],
    goodsN_auge_data['F125W'][goodsN_auge_condition],
	goodsN_auge_data['J'][goodsN_auge_condition],
    goodsN_auge_data['F140W'][goodsN_auge_condition],
    goodsN_auge_data['F160W'][goodsN_auge_condition],
	goodsN_auge_data['H'][goodsN_auge_condition],
	goodsN_auge_data['K'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch1'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch2'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch3'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch4'][goodsN_auge_condition],
	goodsN_auge_data['f24'][goodsN_auge_condition],
	goodsN_auge_data['f100'][goodsN_auge_condition],
	goodsN_auge_data['f160'][goodsN_auge_condition],
	goodsN_auge_data['f250'][goodsN_auge_condition],
	goodsN_auge_data['f350'][goodsN_auge_condition],
	goodsN_auge_data['f500'][goodsN_auge_condition]
]) 

goodsN_flux_err_array_auge = np.asarray([goodsN_auge_Fx_hard_match_mjy*1000*0.2, goodsN_auge_Fx_soft_match_mjy*1000*0.2,
	goodsN_nan_array,
    goodsN_auge_data['FUVerr'][goodsN_auge_condition],
	goodsN_auge_data['NUVerr'][goodsN_auge_condition],
    goodsN_auge_data['Uerr'][goodsN_auge_condition],
	goodsN_auge_data['Berr'][goodsN_auge_condition], 
	goodsN_auge_data['Verr'][goodsN_auge_condition],
    goodsN_auge_data['F606Werr'][goodsN_auge_condition],
	goodsN_auge_data['Rerr'][goodsN_auge_condition], 
	goodsN_auge_data['Ierr'][goodsN_auge_condition],
    goodsN_auge_data['F814Werr'][goodsN_auge_condition],
	goodsN_auge_data['zerr'][goodsN_auge_condition],
    goodsN_auge_data['F125Werr'][goodsN_auge_condition],
	goodsN_auge_data['Jerr'][goodsN_auge_condition],
    goodsN_auge_data['F140Werr'][goodsN_auge_condition],
    goodsN_auge_data['F160Werr'][goodsN_auge_condition],
	goodsN_auge_data['Herr'][goodsN_auge_condition],
	goodsN_auge_data['Kerr'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch1err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch2err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch3err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch4err'][goodsN_auge_condition],
	goodsN_auge_data['f24err'][goodsN_auge_condition],
	goodsN_auge_data['f100err'][goodsN_auge_condition],
	goodsN_auge_data['f160err'][goodsN_auge_condition],
	goodsN_auge_data['f250err'][goodsN_auge_condition],
	goodsN_auge_data['f350err'][goodsN_auge_condition],
	goodsN_auge_data['f500err'][goodsN_auge_condition]
]) 


goodsN_flux_array_auge = goodsN_flux_array_auge.T
goodsN_flux_err_array_auge = goodsN_flux_err_array_auge.T

###############################################################################
###############################################################################
############################ Read in GOODS-N files 2 ##########################

goodsS_auge = fits.open(path+'GOODsS_full_cat.fits')
goodsS_auge_data = goodsS_auge[1].data
goodsS_auge.close()

goodsS_auge_ID = goodsS_auge_data['id_xray']
goodsS_auge_Lx = goodsS_auge_data['Lx']*1.180
goodsS_auge_Lx_hard = goodsS_auge_data['Lx']*0.721
goodsS_auge_z = goodsS_auge_data['z_spec']

goodsS_auge_condition = (np.log10(goodsS_auge_Lx) >= Lx_min) & (np.log10(goodsS_auge_Lx) <= Lx_max) &(goodsS_auge_z > z_min) & (goodsS_auge_z <= z_max) & (goodsS_auge_z != 0.0)

goodsS_auge_ID_match = goodsS_auge_ID[goodsS_auge_condition]
goodsS_auge_Lx_match = goodsS_auge_Lx[goodsS_auge_condition]
goodsS_auge_Lx_hard_match = goodsS_auge_Lx_hard[goodsS_auge_condition]
goodsS_auge_z_match = goodsS_auge_z[goodsS_auge_condition]

print('GOODS-S 2 match: ',len(goodsS_auge_ID_match))


goodsS_auge_Fx_hard_match_mjy = goodsS_auge_data['Fx_hard'][goodsS_auge_condition]*4.136E8/(10-2)
goodsS_auge_Fx_soft_match_mjy = goodsS_auge_data['Fx_soft'][goodsS_auge_condition]*4.136E8/(2-0.5)

goodsS_nan_array = np.zeros(np.shape(goodsS_auge_ID_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsS_nan_array[goodsS_nan_array == 0] = np.nan


goodsS_flux_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000, goodsS_auge_Fx_soft_match_mjy*1000,
    goodsS_nan_array,
    goodsS_auge_data['FUV'][goodsS_auge_condition],
    goodsS_auge_data['NUV'][goodsS_auge_condition],
    goodsS_auge_data['U'][goodsS_auge_condition],
    # goodsS_auge_data['IA427'][goodsS_auge_condition],
    goodsS_auge_data['F435W'][goodsS_auge_condition],
    # goodsS_auge_data['IA445'][goodsS_auge_condition],
    goodsS_auge_data['B'][goodsS_auge_condition],
    # goodsS_auge_data['IA464'][goodsS_auge_condition],
    # goodsS_auge_data['IA484'][goodsS_auge_condition],
    # goodsS_auge_data['IA505'][goodsS_auge_condition],
    # goodsS_auge_data['IA527'][goodsS_auge_condition],
    goodsS_auge_data['V'][goodsS_auge_condition],
    # goodsS_auge_data['IA550'][goodsS_auge_condition],
    # goodsS_auge_data['IA574'][goodsS_auge_condition],
    goodsS_auge_data['F606W'][goodsS_auge_condition],
    # goodsS_auge_data['IA598'][goodsS_auge_condition],
    # goodsS_auge_data['IA624'][goodsS_auge_condition],
    goodsS_auge_data['R'][goodsS_auge_condition],
    # goodsS_auge_data['IA651'][goodsS_auge_condition],
    # goodsS_auge_data['IA679'][goodsS_auge_condition],
    # goodsS_auge_data['IA709'][goodsS_auge_condition],
    # goodsS_auge_data['IA738'][goodsS_auge_condition],
    goodsS_auge_data['I'][goodsS_auge_condition],
    goodsS_auge_data['F775W'][goodsS_auge_condition],
    # goodsS_auge_data['IA767'][goodsS_auge_condition],
    # goodsS_auge_data['IA797'][goodsS_auge_condition],
    goodsS_auge_data['F814W'][goodsS_auge_condition],
    # goodsS_auge_data['IA827'][goodsS_auge_condition],
    # goodsS_auge_data['IA856'][goodsS_auge_condition],
    goodsS_auge_data['z'][goodsS_auge_condition],
    goodsS_auge_data['F850LP'][goodsS_auge_condition],
    goodsS_auge_data['F098M'][goodsS_auge_condition],
    goodsS_auge_data['F105W'][goodsS_auge_condition],
    goodsS_auge_data['F125W'][goodsS_auge_condition],
    goodsS_auge_data['J'][goodsS_auge_condition],
    goodsS_auge_data['F140W'][goodsS_auge_condition],
    goodsS_auge_data['F160W'][goodsS_auge_condition],
    goodsS_auge_data['H'][goodsS_auge_condition],
    goodsS_auge_data['K'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch1'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch2'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch3'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch4'][goodsS_auge_condition],
    goodsS_auge_data['F24'][goodsS_auge_condition],
    goodsS_auge_data['F70'][goodsS_auge_condition],
    goodsS_auge_data['F100'][goodsS_auge_condition],
    goodsS_auge_data['F160'][goodsS_auge_condition],
    goodsS_auge_data['F250'][goodsS_auge_condition],
    goodsS_auge_data['F350'][goodsS_auge_condition],
    goodsS_auge_data['F500'][goodsS_auge_condition],
    ])

goodsS_flux_err_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000*0.2, goodsS_auge_Fx_soft_match_mjy*1000*0.2,
    goodsS_nan_array,
    goodsS_auge_data['FUVerr'][goodsS_auge_condition],
    goodsS_auge_data['NUVerr'][goodsS_auge_condition],
    goodsS_auge_data['Uerr'][goodsS_auge_condition],
    # goodsS_auge_data['IA427err'][goodsS_auge_condition],
    goodsS_auge_data['F435Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA445err'][goodsS_auge_condition],
    goodsS_auge_data['Berr'][goodsS_auge_condition],
    # goodsS_auge_data['IA464err'][goodsS_auge_condition],
    # goodsS_auge_data['IA484err'][goodsS_auge_condition],
    # goodsS_auge_data['IA505err'][goodsS_auge_condition],
    # goodsS_auge_data['IA527err'][goodsS_auge_condition],
    goodsS_auge_data['Verr'][goodsS_auge_condition],
    # goodsS_auge_data['IA550err'][goodsS_auge_condition],
    # goodsS_auge_data['IA574err'][goodsS_auge_condition],
    goodsS_auge_data['F606Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA598err'][goodsS_auge_condition],
    # goodsS_auge_data['IA624err'][goodsS_auge_condition],
    goodsS_auge_data['Rerr'][goodsS_auge_condition],
    # goodsS_auge_data['IA651err'][goodsS_auge_condition],
    # goodsS_auge_data['IA679err'][goodsS_auge_condition],
    # goodsS_auge_data['IA709err'][goodsS_auge_condition],
    # goodsS_auge_data['IA738err'][goodsS_auge_condition],
    goodsS_auge_data['Ierr'][goodsS_auge_condition],
    goodsS_auge_data['F775Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA767err'][goodsS_auge_condition],
    # goodsS_auge_data['IA797err'][goodsS_auge_condition],
    goodsS_auge_data['F814Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA827err'][goodsS_auge_condition],
    # goodsS_auge_data['IA856err'][goodsS_auge_condition],
    goodsS_auge_data['zerr'][goodsS_auge_condition],
    goodsS_auge_data['F850LPerr'][goodsS_auge_condition],
    goodsS_auge_data['F098Merr'][goodsS_auge_condition],
    goodsS_auge_data['F105Werr'][goodsS_auge_condition],
    goodsS_auge_data['F125Werr'][goodsS_auge_condition],
    goodsS_auge_data['Jerr'][goodsS_auge_condition],
    goodsS_auge_data['F140Werr'][goodsS_auge_condition],
    goodsS_auge_data['F160Werr'][goodsS_auge_condition],
    goodsS_auge_data['Herr'][goodsS_auge_condition],
    goodsS_auge_data['Kerr'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch1err'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch2err'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch3err'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch4err'][goodsS_auge_condition],
    goodsS_auge_data['F24err'][goodsS_auge_condition],
    goodsS_auge_data['F70err'][goodsS_auge_condition],
    goodsS_auge_data['F100err'][goodsS_auge_condition],
    goodsS_auge_data['F160err'][goodsS_auge_condition],
    goodsS_auge_data['F250err'][goodsS_auge_condition],
    goodsS_auge_data['F350err'][goodsS_auge_condition],
    goodsS_auge_data['F500err'][goodsS_auge_condition]
    ])




# goodsS_flux_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000, goodsS_auge_Fx_soft_match_mjy*1000,
# 	goodsS_nan_array,
#     mag_to_flux(goodsS_auge_data['FUV'][goodsS_auge_condition],'FUV')*1E6,
# 	mag_to_flux(goodsS_auge_data['NUV'][goodsS_auge_condition],'NUV')*1E6,
# 	mag_to_flux(goodsS_auge_data['Umag'][goodsS_auge_condition],'U')*1E6,
#     mag_to_flux(goodsS_auge_data['F435W'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Bmag'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Vmag'][goodsS_auge_condition],'V')*1E6,
#     mag_to_flux(goodsS_auge_data['F606W'][goodsS_auge_condition], 'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Rmag'][goodsS_auge_condition],'R')*1E6,
# 	mag_to_flux(goodsS_auge_data['Imag'][goodsS_auge_condition], 'I')*1E6,
#     mag_to_flux(goodsS_auge_data['F775W'][goodsS_auge_condition], 'AB')*1E6,
#    	mag_to_flux(goodsS_auge_data['F814W'][goodsS_auge_condition], 'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['zmag'][goodsS_auge_condition],'sloan_z')*1E6,
#     mag_to_flux(goodsS_auge_data['F098M'][goodsS_auge_condition], 'AB')*1E6,
#     mag_to_flux(goodsS_auge_data['F125W'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Jmag'][goodsS_auge_condition],'JUK')*1E6,
# 	mag_to_flux(goodsS_auge_data['Hmag'][goodsS_auge_condition],'HUK')*1E6,
# 	mag_to_flux(goodsS_auge_data['Kmag'][goodsS_auge_condition], 'KUK')*1E6,
# 	goodsS_auge_data['IRAC1'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC2'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC3'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC4'][goodsS_auge_condition],
# 	goodsS_auge_data['F24'][goodsS_auge_condition],
#     goodsS_auge_data['F70'][goodsS_auge_condition],
# 	goodsS_auge_data['F100'][goodsS_auge_condition],
# 	goodsS_auge_data['F160'][goodsS_auge_condition],
# 	goodsS_auge_data['F250'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F350'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F500'][goodsS_auge_condition]*1000
# ]) 

# goodsS_flux_err_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000*0.2, goodsS_auge_Fx_soft_match_mjy*1000*0.2,
# 	goodsS_nan_array,
#     magerr_to_fluxerr(goodsS_auge_data['FUV'][goodsS_auge_condition],goodsS_auge_data['FUVerr'][goodsS_auge_condition],'U')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F435W'][goodsS_auge_condition],goodsS_auge_data['F435W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['NUV'][goodsS_auge_condition],goodsS_auge_data['NUVerr'][goodsS_auge_condition], 'B')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['Umag'][goodsS_auge_condition],goodsS_auge_data['Uerrmag'][goodsS_auge_condition],'U')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Bmag'][goodsS_auge_condition],goodsS_auge_data['Berrmag'][goodsS_auge_condition], 'B')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Vmag'][goodsS_auge_condition],goodsS_auge_data['Verrmag'][goodsS_auge_condition], 'V')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F606W'][goodsS_auge_condition],goodsS_auge_data['F606W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Rmag'][goodsS_auge_condition],goodsS_auge_data['Rerrmag'][goodsS_auge_condition], 'R')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Imag'][goodsS_auge_condition],goodsS_auge_data['Imagerr'][goodsS_auge_condition],'I')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F775W'][goodsS_auge_condition],goodsS_auge_data['F775W'][goodsS_auge_condition],'AB')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F814W'][goodsS_auge_condition],goodsS_auge_data['F814W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['zmag'][goodsS_auge_condition],goodsS_auge_data['zmagerr'][goodsS_auge_condition],'sloan_z')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Jmag'][goodsS_auge_condition],goodsS_auge_data['Jmagerr'][goodsS_auge_condition],'JVHS')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F098M'][goodsS_auge_condition],goodsS_auge_data['F098M'][goodsS_auge_condition],'AB')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F125W'][goodsS_auge_condition],goodsS_auge_data['F125W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Hmag'][goodsS_auge_condition],goodsS_auge_data['Hmagerr'][goodsS_auge_condition],'HVHS')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Kmag'][goodsS_auge_condition],goodsS_auge_data['Kmagerr'][goodsS_auge_condition],'KVHS')*1E6,
# 	goodsS_auge_data['IRAC1err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC2err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC3err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC4err'][goodsS_auge_condition],
# 	goodsS_auge_data['F24err'][goodsS_auge_condition],
#     goodsS_auge_data['F70err'][goodsS_auge_condition],
# 	goodsS_auge_data['F100err'][goodsS_auge_condition],
# 	goodsS_auge_data['F160err'][goodsS_auge_condition],
# 	goodsS_auge_data['F250err'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F350err'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F500err'][goodsS_auge_condition]*1000
# ])

print(np.shape(goodsS_flux_array_auge))

goodsS_flux_array_auge = goodsS_flux_array_auge.T
goodsS_flux_err_array_auge = goodsS_flux_err_array_auge.T

##### Figure 1 & 2 plots #####

plt.rcParams['font.size']=24
plt.rcParams['axes.linewidth']=3
plt.rcParams['xtick.major.size']=3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size']=3
plt.rcParams['ytick.major.width'] = 3

# plt.figure(figsize=(10,10))
# plt.plot(goodsS_auge_z_match,np.log10(goodsS_auge_Lx_match),'.',color='gray',rasterized=True)
# plt.plot(goodsN_auge_z_match,np.log10(goodsN_auge_Lx_match),'.',color='gray',rasterized=True)
# plt.plot(xue_z_match,np.log10(xue_Lx_match),'.',color='gray',rasterized=True)
# plt.plot(luo_z_match,np.log10(luo_Lx_match),'.',color='gray',rasterized=True,label='GOODS-N/S')
# plt.plot(chandra_cosmos_z_match,np.log10(chandra_cosmos_Lx_full_match),'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# # plt.plot(s82x_z_sp_match,np.log10(s82x_≥Lx_sp_full_match),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.plot(s82x_z_sp,np.log10(s82x_Lx_sp_full),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.xlabel('Spectroscopic Redshift')
# plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# plt.text(3.25, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)}')
# # plt.text(3.25, 40.55, f'n = {len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)}')
# plt.legend()
# plt.xlim(-0.075,5.5)
# plt.grid()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_spec.pdf')
# plt.show()

cdf_z = np.append(xue_z_match,luo_z_match)
cdf_lx = np.append(xue_Lx_match,luo_Lx_match)

cdf_z = np.append(cdf_z, goodsS_auge_z_match)
cdf_lx = np.append(cdf_lx, goodsS_auge_Lx_match)

cdf_z = np.append(cdf_z, goodsN_auge_z_match)
cdf_lx = np.append(cdf_lx, goodsN_auge_Lx_match)

print('Stripe82X: ',len(s82x_Lx_sp_full))
print('COSMOS: ',len(chandra_cosmos_z_match))
print('GOODS: ',len(cdf_z))

# fig = plt.figure(figsize=(20,10))
# ax1 = plt.subplot(121)
# ax1.hist(cdf_z,bins=np.arange(0,6,0.25),histtype='step',color='gray',lw=3,label='GOODS-N/S')
# ax1.hist(chandra_cosmos_z_match,bins=np.arange(0,6,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax1.hist(s82x_z_sp,bins=np.arange(0,6,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax1.set_xlabel('Spectroscopic Redshift')
# ax1.set_xlim(-0.075, 5.5)
# ax1.set_ylim(0,475)
# ax1.set_ylabel('N')
# ax1.grid()

# ax2 = plt.subplot(122)
# ax2.hist(np.log10(cdf_lx),bins=np.arange(39,46.5,0.25),histtype='step',color='gray',lw=3)
# ax2.hist(np.log10(chandra_cosmos_Lx_full_match),bins=np.arange(39,46.5,0.25),histtype='step',color='b',lw=3)
# ax2.hist(np.log10(s82x_Lx_sp_full),bins=np.arange(39,46.5,0.25),histtype='step',color='r',lw=3)
# ax2.set_xlabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# # ax2.set_ylabel('N')
# ax2.set_ylim(0,475)
# ax2.set_yticklabels([])
# ax2.grid()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_hist_spec_NEW.pdf')
# plt.show()


##############################

###############################################################################
COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'J_FLUX_APER2', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
S82X_filters = np.asarray(['Fx_hard','Fx_soft','nan','MAG_FUV','MAG_NUV','U','G','R','I','Z','JVHS','HVHS','KVHS','W1','W2','W3','W4','nan','FLUX_250_s82x','FLUX_350_s82x','FLUX_500_s82x'])
# S82X_filters = np.asarray(['Fx_hard','Fx_soft','nan','MAG_FUV','MAG_NUV','U','G','R','I','Z','JVHS','HVHS','KVHS','W1','W2','W3','W4','nan'])
GOODSN_filters  = np.asarray(['Fx_hard','Fx_soft','nan','FLUX_GALEX_FUV','FLUX_GALEX_NUV','u_FLUX_APER2','F435W','F606W','F775W','F814W','F105W','F125W','F140W','F160W','Ks_FLUX_APER2','SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX','FLUX_24','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_350','FLUX_500'])
GOODSS_filters = np.asarray(['Fx_hard','Fx_soft','nan','u_FLUX_APER2','F435W','F606W','F775W','F814W','F098M','F105W','F125W','F160W','Ks_FLUX_APER2','SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX','FLUX_24','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_350','FLUX_500'])

# GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'Z', 'JVHS', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV','U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F098M', 'F125W', 'JVHS', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'IB427_FLUX', 'F435W', 'IB445_FLUX', 'B_FLUX_APER2', 'IB464_FLUX_APER2', 'IA484_FLUX_APER2', 'IB505_FLUX_APER2', 'IA527_FLUX_APER2', 'V_FLUX_APER2', 'IB550_FLUX', 'IB574_FLUX_APER2', 'F606W', 'IB598_FLUX', 'IA624_FLUX_APER2', 'R', 'IB651_FLUX', 'IA679_FLUX_APER2', 'IB709_FLUX_APER2', 'IA738_FLUX_APER2', 'I', 
                                #   'F775W', 'IA767_FLUX_APER2', 'IA797_FLUX', 'F814W', 'IB827_FLUX_APER2', 'IA856_FLUX', 'Z','F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F814W', 'Z', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I',
                                  'F775W', 'F814W', 'Z', 'F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
print('HERE: ',len(GOODSS_auge_filters))
###############################################################################
tfl = time.perf_counter()
print(f'Done with file reading ({tfl - ti:0.4f} second)')

# Make empty lists to be filled with SED outputs
all_id, all_z, all_x, all_y, all_frac_err = [], [], [], [], []
F1, F025, F5, F10, F100 = [], [], [], [], []
Lx, Lbol = [], []
field = []
UVslope, MIRslope1, MIRslope2 = [], [], []
spec_type = []
Nh = []
FFIR, WFIR = [], []
median_x, median_y = [], []
median_fir_x, median_fir_y = [], []
upper_check, fir_frac = [], []
FFIR_2, WFIR_2, F100_2 = [], [], []
s82x_Lx = []
check_sed = []
uv_lum, opt_lum, mir_lum, fir_lum = [], [], [], []

###############################################################################
###############################################################################
############################### Run COSMOS SEDs ###############################
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(COSMOS_filters))
fill_nan[fill_nan == 0] = np.nan

cosmos_target_id = []
cosmos_target_ra = []
cosmos_target_dec = []
# '''
for i in range(len(chandra_cosmos_phot_id_match)):
    # if cosmos_flux_array[i][COSMOS_filters == 'FLUX_24'] <= 0:
    #     continue
    # elif cosmos_flux_err_array[i][COSMOS_filters == 'FLUX_24']/cosmos_flux_array[i][COSMOS_filters == 'FLUX_24'] > 0.5:
    #     continue
    # else:
    # elif chandra_cosmos_phot_id_match[i] == 286067: 
        source = AGN(chandra_cosmos_phot_id_match[i],chandra_cosmos_z_match[i],COSMOS_filters,cosmos_flux_array[i],cosmos_flux_err_array[i])
        source.MakeSED()
        check_sed.append(source.CheckSED(10, check_span=2.75))

        F1.append(source.Find_nuFnu(1.0))
        F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
        F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
        # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

        # ffir, wfir,f100 = source.median_FIR_filter(['FLUX_24','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
        ffir, wfir,f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)

        FFIR.append(ffir)
        WFIR.append(wfir)
        F100.append(f100/source.Find_nuFnu(1.0))

        # FFIR_2.append(ffir2)
        # WFIR_2.append(wfir2)
        # F100_2.append(f1002/source.Find_nuFnu(1.0))

        UVslope.append(source.Find_slope(0.15,1.0))
        MIRslope1.append(source.Find_slope(1.0,6.5))
        MIRslope2.append(source.Find_slope(6.5,10))

        uv_lum.append(source.find_Lum_range(0.1,0.35))
        opt_lum.append(source.find_Lum_range(0.35,3))
        mir_lum.append(source.find_Lum_range(3,30))
        fir_lum.append(source.find_Lum_range(30,500/(1+chandra_cosmos_z_match[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        w = np.append(w,fill_nan)
        f = np.append(f,fill_nan)
        frac_err = np.append(frac_err,fill_nan)
        all_id.append(Id)
        all_z.append(redshift)
        all_x.append(w)
        all_y.append(f)
        all_frac_err.append(frac_err)
        upper_check.append(up_check)

        # if chandra_cosmos_phot_id_match[i] == 262286:
        #     print(chandra_cosmos_xid_match[i])
        #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(chandra_cosmos_Lx_full_match[i]))
        #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))
        # elif chandra_cosmos_phot_id_match[i] == 818825:
        #     print(chandra_cosmos_xid_match[i])
        #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(chandra_cosmos_Lx_full_match[i]))
        #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

        med_x, med_y = source.median_SED(['U'], ['FLUX_24'])
        median_x.append(med_x)
        median_y.append(med_y)

        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
        median_fir_x.append(med_fir_x)
        median_fir_y.append(med_fir_y)

        Lx.append(chandra_cosmos_Lx_full_match[i])
        # Lx.append(chandra_cosmos_Lx_hard_match[i])
        Nh.append(chandra_cosmos_Nh_match[i])
        Lbol.append(source.Find_Lbol())
        fir_frac.append(source.FIR_frac())

        field.append(0)

        cosmos_target_id.append(chandra_cosmos_phot_id_match[i])
        cosmos_target_ra.append(chandra_cosmos_RA_match[i])
        cosmos_target_dec.append(chandra_cosmos_DEC_match[i])
        # print(chandra_cosmos_Lx_full_match[i])
# '''
tc = time.perf_counter()   
print(f'Done with COSMOS sources ({tc - tfl:0.4f} second)')

# cosmos_target_ra = np.asarray(cosmos_target_ra)
# cosmos_target_dec = np.asarray(cosmos_target_dec)
# cosmos_ra_out2 = []
# cosmos_dec_out2 = []
# c2 = SkyCoord(ra=cosmos_target_ra*u.degree, dec=cosmos_target_dec*u.degree)
# str_list = c2.to_string('hmsdms')
# ra_out2, dec_out2 = [], []
# for i in range(len(str_list)):
#     str_list[i] = str_list[i].replace("h","")
#     str_list[i] = str_list[i].replace("m", "")
#     str_list[i] = str_list[i].replace("d", "")
#     str_list[i] = str_list[i].replace("s", "")
#     out_str = str_list[i].split()

#     cosmos_ra_out2.append(out_str[0])
#     cosmos_dec_out2.append(out_str[1])

# outf = open('/Users/connor_auge/Desktop/all_cosmos_targets.csv','w')
# for i in range(len(cosmos_target_id)):
#     outf.writelines('%s,%s,%s,%s\n' % (cosmos_target_id[i],cosmos_ra_out2[i],cosmos_dec_out2[i],2000))

ra_out = []
dec_out = []
###############################################################################
###############################################################################
############################# Run Stripe82X SEDs ##############################
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(S82X_filters))
fill_nan[fill_nan == 0] = np.nan
# '''
for i in range(len(lamassa_id_use)):
    # elif s82x_flux_err_array[i][S82X_filters == 'W4']/s82x_flux_array[i][S82X_filters == 'W4'] >= 0.75:
        # continue
    # elif s82x_flux_err_array[i][S82X_filters == 'W3'] == 0:
    #     continue
    # elif s82x_flux_err_array[i][S82X_filters == 'W3']/s82x_flux_array[i][S82X_filters == 'W3'] >= 0.46:
        # continue

    # if s82x_flux_err_array[i][S82X_filters == 'W4'] == 0:
    #     continue
    # elif np.isnan(s82x_flux_err_array[i][S82X_filters == 'W4']/s82x_flux_array[i][S82X_filters == 'W4']):
    #     continue
    # else:

    # if s82x_flux_err_array[i][S82X_filters == 'W4'] == 0 or np.isnan(s82x_flux_err_array[i][S82X_filters == 'W4']/s82x_flux_array[i][S82X_filters == 'W4']):
        # print('1',s82x_flux_array[i])
        # print('1', len(s82x_flux_array[i][np.isnan(s82x_flux_array[i])]))

        # print('2',s82x_flux_err_array[i])
        try:
            source = AGN(lamassa_id_use[i], s82x_z_sp[i], S82X_filters, s82x_flux_array[i], s82x_flux_err_array[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10,check_span=2.75))
           

            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['W4', 'FLUX_250', 'FLUX_350_s82x', 'FLUX_500_s82x'],Find_value=100.0)
            ffir = np.append(np.array([np.nan, np.nan]), ffir)
            wfir = np.append(np.array([np.nan, np.nan]), wfir)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            UVslope.append(source.Find_slope(0.15, 1.0))
            MIRslope1.append(source.Find_slope(1.0, 6.5))
            MIRslope2.append(source.Find_slope(6.5, 10))

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+s82x_z_sp[i])))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            w = np.append(w,fill_nan)
            f = np.append(f,fill_nan)
            frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)

            # if check_sed[i] == 'GOOD':
            #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(s82x_Lx_sp_full[i]))
            #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

            med_x, med_y = source.median_SED(['U'], ['W4'])
            median_x.append(med_x)
            median_y.append(med_y)

            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(s82x_Lx_sp_full[i])
            # Lx.append(s82x_Lx_sp_hard[i])
            Nh.append(s82x_Nh[i])
            Lbol.append(source.Find_Lbol())
            fir_frac.append(source.FIR_frac())

            field.append(1)
            ra_out.append(lamassa_ra[i])
            dec_out.append(lamassa_dec[i])

        except ValueError:
            continue

# '''
# outf = open('/Users/connor_auge/Desktop/s82x_check_coords.csv','w')
# outf.writelines('ID,RA,DEC\n')
# for i in range(len(ra_out)):
#     outf.writelines('%s,%s,%s\n' % (all_id[i],ra_out[i],dec_out[i]))
# outf.close()

ts = time.perf_counter()
print(f'Done with S82X sources ({ts - tc:0.4f} second)')
###############################################################################
###############################################################################
############################### Run GOODS-N SEDs ##############################

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_filters))
fill_nan[fill_nan == 0] = np.nan

goodsN_id_candels = []
goodsN_id_auge = []

# '''
for i in range(len(goodsN_phot_id_match)):
    # if goodsN_flux_array[i][GOODSN_filters == 'FLUX_24'] <= 0:
    #     continue
    # elif goodsN_flux_err_array[i][GOODSN_filters == 'FLUX_24']/goodsN_flux_array[i][GOODSN_filters == 'FLUX_24'] > 0.5:
    #     continue
    # else:
        source = AGN(goodsN_phot_id_match[i],xue_z_match[i],GOODSN_filters,goodsN_flux_array[i],goodsN_flux_err_array[i])
        source.MakeSED()
        check_sed.append(source.CheckSED(10, check_span=2.75))


        F1.append(source.Find_nuFnu(1.0))
        F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
        F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
        # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

        ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
        FFIR.append(ffir)
        WFIR.append(wfir)
        F100.append(f100/source.Find_nuFnu(1.0))

        UVslope.append(source.Find_slope(0.15,1.0))
        MIRslope1.append(source.Find_slope(1.0,6.5))
        MIRslope2.append(source.Find_slope(6.5,10))

        uv_lum.append(source.find_Lum_range(0.1,0.35))
        opt_lum.append(source.find_Lum_range(0.35,3))
        mir_lum.append(source.find_Lum_range(3,30))
        fir_lum.append(source.find_Lum_range(30,500/(1+xue_z_match[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        w = np.append(w,fill_nan)
        f = np.append(f,fill_nan)
        frac_err = np.append(frac_err, fill_nan)
        all_id.append(Id)
        all_z.append(redshift)
        all_x.append(w)
        all_y.append(f)
        all_frac_err.append(frac_err)
        upper_check.append(up_check)

        med_x, med_y = source.median_SED(['U'], ['MIPS2'])
        median_x.append(med_x)
        median_y.append(med_y)
        
        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
        median_fir_x.append(med_fir_x)
        median_fir_y.append(med_fir_y)

        Lx.append(xue_Lx_match[i])
        # Lx.append(xue_Lx_hard_match[i])
        Nh.append(goodsN_Nh_match[i])
        Lbol.append(source.Find_Lbol())
        fir_frac.append(source.FIR_frac())
        
        if source.CheckSED(10, check_span=2.5) == 'GOOD':
            goodsN_id_candels.append(goodsN_phot_id_match[i])

        field.append(2)

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_auge_filters))
fill_nan[fill_nan == 0] = np.nan

for i in range(len(goodsN_auge_ID_match)):
    if goodsN_auge_ID_match[i] in xue_id_match:
        print('repeat')
    # if goodsN_auge_ID_match[i] == 348:
        # continue
    else:
        try:
            source = AGN(goodsN_auge_ID_match[i],goodsN_auge_z_match[i],GOODSN_auge_filters,goodsN_flux_array_auge[i],goodsN_flux_err_array_auge[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10, check_span=2.75))
            # print(check_sed[i])


            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            UVslope.append(source.Find_slope(0.15,1.0))
            MIRslope1.append(source.Find_slope(1.0,6.5))
            MIRslope2.append(source.Find_slope(6.5,10))

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+goodsN_auge_z_match[i])))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            w = np.append(w,fill_nan)
            f = np.append(f,fill_nan)
            frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)
            
            # if source.CheckSED(10, check_span=2.5) == 'GOOD':
            #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(goodsN_auge_Lx_match[i]))
            #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

            med_x, med_y = source.median_SED(['U'], ['FLUX_24'])
            median_x.append(med_x)
            median_y.append(med_y)

            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(goodsN_auge_Lx_match[i])
            # Lx.append(goodsN_auge_Lx_hard_match[i])    
            Nh.append(0.0)
            Lbol.append(source.Find_Lbol())
            fir_frac.append(source.FIR_frac())

            if source.CheckSED(10, check_span=2.5) == 'GOOD':
                goodsN_id_auge.append(goodsN_auge_ID_match[i])

            field.append(2)
        except ValueError:
            continue



# '''
tgn = time.perf_counter()   
print(f'Done with GOODS-N sources ({tgn - ts:0.4f} second)')

###############################################################################
###############################################################################
############################### Run GOODS-S SEDs ###############################

# fill_nan = np.zeros(len(GOODSN_filters)-len(GOODSS_filters))
# fill_nan[fill_nan == 0] = np.nan

goodsS_id_candels = []
goodsS_id_auge = []

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSS_filters))
fill_nan[fill_nan == 0] = np.nan

# '''
for i in range(len(goodsS_phot_id_match)):
    # if goodsS_flux_array[i][GOODSS_filters == 'FLUX_24'] <= 0:
    #     continue
    # elif goodsS_flux_err_array[i][COSMOS_filters == 'FLUX_24']/goodsS_flux_array[i][COSMOS_filters == 'FLUX_24'] > 0.5:
    #     continue
    # else:
        source = AGN(goodsS_phot_id_match[i],luo_z_match[i],GOODSS_filters,goodsS_flux_array[i],goodsS_flux_err_array[i])
        source.MakeSED()
        check_sed.append(source.CheckSED(10, check_span=2.75))

        F1.append(source.Find_nuFnu(1.0))
        F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
        F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
        # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

        ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
        FFIR.append(ffir)
        WFIR.append(wfir)
        F100.append(f100/source.Find_nuFnu(1.0))

        UVslope.append(source.Find_slope(0.15,1.0))
        MIRslope1.append(source.Find_slope(1.0,6.5))
        MIRslope2.append(source.Find_slope(6.5,10))

        uv_lum.append(source.find_Lum_range(0.1,0.35))
        opt_lum.append(source.find_Lum_range(0.35,3))
        mir_lum.append(source.find_Lum_range(3,30))
        fir_lum.append(source.find_Lum_range(30,500/(1+luo_z_match[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        w = np.append(w, fill_nan)
        f = np.append(f, fill_nan)
        all_id.append(Id)
        all_z.append(redshift)
        all_x.append(w)
        all_y.append(f)
        all_frac_err.append(frac_err)
        upper_check.append(up_check)

        med_x, med_y = source.median_SED(['U'], ['MIPS2'])
        median_x.append(med_x)
        median_y.append(med_y)
        
        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
        median_fir_x.append(med_fir_x)
        median_fir_y.append(med_fir_y)

        Lx.append(luo_Lx_match[i])
        # Lx.append(luo_Lx_hard_match[i])
        Nh.append(goodsS_NH_match[i])
        Lbol.append(source.Find_Lbol())
        fir_frac.append(source.FIR_frac())

        if source.CheckSED(10, check_span=2.5) == 'GOOD':
            goodsS_id_candels.append(goodsS_phot_id_match[i])

        field.append(3)


for i in range(len(goodsS_auge_ID_match)):
    if goodsS_auge_ID_match[i] in luo_id_match:
    #     # continue
        print('repeat')
    else:
        try:
            source = AGN(goodsS_auge_ID_match[i],goodsS_auge_z_match[i],GOODSS_auge_filters,goodsS_flux_array_auge[i],goodsS_flux_err_array_auge[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10, check_span=2.75))


            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            UVslope.append(source.Find_slope(0.15,1.0))
            MIRslope1.append(source.Find_slope(1.0,6.5))
            MIRslope2.append(source.Find_slope(6.5,10))

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+goodsS_auge_z_match[i])))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            # w = np.append(w,fill_nan)
            # f = np.append(f,fill_nan)
            # frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)

            # if source.CheckSED(10, check_span=2.5) == 'GOOD':
            #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(goodsS_auge_Lx_match[i]))
            #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))


            med_x, med_y = source.median_SED(['U'], ['MIPS2'])
            median_x.append(med_x)
            median_y.append(med_y)
            
            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(goodsS_auge_Lx_match[i])
            # Lx.append(goodsS_auge_Lx_hard_match[i])
            Nh.append(0.0)
            Lbol.append(source.Find_Lbol())
            fir_frac.append(source.FIR_frac())

            if source.CheckSED(10, check_span=2.5) == 'GOOD':
                goodsS_id_auge.append(goodsS_auge_ID_match[i])

            field.append(3)
        except ValueError:
            continue
# '''

tgs = time.perf_counter()   
print(f'Done with GOODS-S sources ({tgs - tgn:0.4f} second)')

plt.rcParams['font.size']=24
plt.rcParams['axes.linewidth']=3
plt.rcParams['xtick.major.size']=3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size']=3
plt.rcParams['ytick.major.width'] = 3




###############################################################################
###############################################################################
############################### Begin Plotting ################################
# plt.rcParams['font.size'] = 14
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['xtick.major.size'] = 2
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.size'] = 2
# plt.rcParams['ytick.major.width'] = 2

plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.major.width'] = 3

check_sed = np.asarray(check_sed)
GOOD_SEDs = np.where(check_sed == 'GOOD')
# GOOD_SEDs = np.where(check_sed == 'BAD')
print('ALL: ', len(check_sed))
print('BAD SEDs: ', len(check_sed[check_sed == 'BAD']))
print('GOOD SEDs: ', len(check_sed[check_sed == 'GOOD']))
# print('Except: ',len(bad_id))
# all_id, all_z, all_x, all_y, Lx, median_x, median_y = np.asarray(all_id), np.asarray(all_z), np.asarray(all_x), np.asarray(all_y), np.log10(np.asarray(Lx)), np.asarray(median_x), np.asarray(median_y)
# F1, F025, F5, F10, F100 = np.asarray(F1), np.asarray(F025), np.asarray(F5), np.asarray(F10), np.asarray(F100)
# field = np.asarray(field)
# UVslope, MIRslope1, MIRslope2 = np.asarray(UVslope), np.asarray(MIRslope1), np.asarray(MIRslope2)
# FFIR, WFIR = np.asarray(FFIR), np.asarray(WFIR)
# spec_type = np.asarray(spec_type)
# upper_check, fir_frac = np.asarray(upper_check), np.asarray(fir_frac)
# Lbol = np.asarray(Lbol)
# Nh = np.asarray(Nh)
# F100_2 = np.asarray(F100_2)
# FFIR_2, WFIR_2 = np.asarray(FFIR_2), np.asarray(WFIR_2)

all_id, all_z, all_x, all_y, Lx, median_x, median_y = np.asarray(all_id)[GOOD_SEDs], np.asarray(all_z)[GOOD_SEDs], np.asarray(all_x)[GOOD_SEDs], np.asarray(all_y)[GOOD_SEDs], np.log10(np.asarray(Lx))[GOOD_SEDs], np.asarray(median_x)[GOOD_SEDs], np.asarray(median_y)[GOOD_SEDs]
F1, F025, F5, F10, F100 = np.asarray(F1)[GOOD_SEDs], np.asarray(F025)[GOOD_SEDs], np.asarray(F5)[GOOD_SEDs], np.asarray(F10)[GOOD_SEDs], np.asarray(F100)[GOOD_SEDs]
field = np.asarray(field)[GOOD_SEDs]
UVslope, MIRslope1, MIRslope2 = np.asarray(UVslope)[GOOD_SEDs], np.asarray(MIRslope1)[GOOD_SEDs], np.asarray(MIRslope2)[GOOD_SEDs]
FFIR, WFIR = np.asarray(FFIR)[GOOD_SEDs], np.asarray(WFIR)[GOOD_SEDs]
# spec_type = np.asarray(spec_type)[GOOD_SEDs]
upper_check, fir_frac = np.asarray(upper_check)[GOOD_SEDs], np.asarray(fir_frac)[GOOD_SEDs]
Lbol = np.asarray(Lbol)[GOOD_SEDs]
Nh = np.asarray(Nh)[GOOD_SEDs]
# F100_2 = np.asarray(F100_2)[GOOD_SEDs]
# FFIR_2, WFIR_2 = np.asarray(FFIR_2)[GOOD_SEDs], np.asarray(WFIR_2)[GOOD_SEDs]
median_fir_x, median_fir_y = np.asarray(median_fir_x)[GOOD_SEDs], np.asarray(median_fir_y)[GOOD_SEDs]
uv_lum, opt_lum, mir_lum, fir_lum = np.asarray(uv_lum)[GOOD_SEDs], np.asarray(opt_lum)[GOOD_SEDs], np.asarray(mir_lum)[GOOD_SEDs], np.asarray(fir_lum)[GOOD_SEDs]


print('GOODS-N Candels: ', goodsN_id_candels)
print('GOODS-N Auge: ', goodsN_id_auge)
print('GOODS-S Candels: ', goodsS_id_candels)
print('GOODS-S Auge: ', goodsS_id_auge)



# plt.figure(figsize=(10,10))
# # plt.plot(all_z[field == 2],Lx[field == 2],'.',color='gray',rasterized=True)
# # plt.plot(all_z[field == 3],Lx[field == 3],'.',color='gray',rasterized=True,label='GOODS-N/S')
# # plt.plot(all_z[field == 0],Lx[field == 0],'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# plt.plot(all_z[field == 1],Lx[field == 1],'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.xlabel('Spectroscopic Redshift')
# plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# plt.text(0.025,45.5,f'n = {len(all_z)}')
# plt.legend()
# # plt.xlim(-0.075,1.4)
# # plt.ylim(42.9,46.5)
# plt.grid()
# plt.tight_layout()
# # plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_COSMOS_S82X_good.pdf')
# plt.show()

# fig = plt.figure(figsize=(20,10))
# ax1 = plt.subplot(121)
# ax1.hist(all_z[np.logical_or(field == 2, field == 3)],bins=np.arange(0,1.5,0.25),histtype='step',color='gray',lw=3,label='GOODS-N/S')
# ax1.hist(all_z[field == 0],bins=np.arange(0,1.5,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax1.hist(all_z[field == 1],bins=np.arange(0,1.5,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax1.set_xlabel('Spectroscopic Redshift')
# ax1.set_xlim(-0.075, 1.5)
# ax1.set_ylim(0,350)
# ax1.set_ylabel('N')
# ax1.grid()

# ax2 = plt.subplot(122)
# ax2.hist(Lx[np.logical_or(field == 2, field == 3)],bins=np.arange(39,46.5,0.25),histtype='step',color='gray',lw=3)
# ax2.hist(Lx[field == 0],bins=np.arange(39,47.5,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax2.hist(Lx[field == 1],bins=np.arange(39,47.5,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax2.set_xlabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# ax2.set_ylabel('N')
# ax2.set_ylim(0,150)
# ax2.set_xlim(42.8,46)
# ax2.grid()
# ax2.legend()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_hist_COSMOS_S82X_good.pdf')
# plt.show()




print('COSMOS Sample: ', len(Lx[field == 0]))
print('S82X Sample: ', len(Lx[field == 1]))
print('GOODS-N Sample: ', len(Lx[field == 2]))
print('GOODS-S Sample: ', len(Lx[field == 3]))

plot = Plotter_Letter(all_id,all_z,all_x,all_y,all_frac_err)
plot2 = Plotter_Letter2(all_id,all_z,all_x,all_y,all_frac_err)
sort = Lx.argsort()


# plt.plot(np.log10(F100[field == 0]*F1[field == 0]), np.log10(F100_2[field == 0]*F1[field == 0]), '.',color='b',label='COSMOS')
# plt.plot(np.log10(F100[field == 1]*F1[field == 1]), np.log10(F100_2[field == 1]*F1[field == 1]), '.',color='orange',label='Stripe82X')
# plt.plot(np.log10(F100[field == 2]*F1[field == 2]), np.log10(F100_2[field == 2]*F1[field == 2]), '.',color='gray',label='GOODS-N/S')
# plt.plot(np.log10(F100[field == 3]*F1[field == 3]), np.log10(F100_2[field == 3]*F1[field == 3]), '.',color='gray')
# plt.plot(np.arange(40,47,1),np.arange(40,47,1),color='k')
# plt.xlabel('New')
# plt.ylabel('Old')
# plt.legend()
# plt.show()

zlim_1 = 0.0
zlim_2 = 0.6
zlim_3 = 0.9
zlim_4 = 1.2

B1 = np.logical_and(all_z >= zlim_1,all_z <= zlim_2)
B2 = np.logical_and(all_z > zlim_2,all_z <= zlim_3)
B3 = np.logical_and(all_z > zlim_3,all_z <= zlim_4)

# plt.figure(figsize=(8,6))
# plt.hist(np.log10(F1[field==1]),bins=np.arange(40,50,0.25),histtype='step',color='r',label='Stripe82X',lw=4.0)
# plt.axvline(np.log10(np.nanmedian(F1[field==1])),ls='--',color='r',lw=3.5)
# plt.hist(np.log10(F1[field==0]),bins=np.arange(40,50,0.25),histtype='step',color='b',label='COSMOS',lw=3.5)
# plt.axvline(np.log10(np.nanmedian(F1[field==0])),ls='--',color='b',lw=3.5)
# plt.hist(np.log10(F1[np.logical_or(field==2,field==3)]),bins=np.arange(40,50,0.25),histtype='step',color='gray',label='GOODS-N/S',lw=3.0)
# plt.axvline(np.log10(np.nanmedian(F1[np.logical_or(field==2,field==3)])),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{1 \mu \mathrm{m}}$ [erg/s]')
# plt.xlim(42,47)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.hist(Lx[field==1],bins=np.arange(40,50,0.25),histtype='step',color='r',label='Stripe82X',lw=4.0)
# plt.axvline(np.nanmedian(Lx[field==1]),ls='--',color='r',lw=3.5)
# plt.hist(Lx[field==0],bins=np.arange(40,50,0.25),histtype='step',color='b',label='COSMOS',lw=3.5)
# plt.axvline(np.nanmedian(Lx[field==0]),ls='--',color='b',lw=3.5)
# plt.hist(Lx[np.logical_or(field==2,field==3)],bins=np.arange(40,50,0.25),histtype='step',color='gray',label='GOODS-N/S',lw=3.0)
# plt.axvline(np.nanmedian(Lx[np.logical_or(field==2,field==3)]),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{\mathrm{X}}$ [erg/s]')
# plt.xlim(42, 47)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,6))
# plt.hist(np.log10(F1[B3]),bins=np.arange(40,50,0.25),histtype='step',color='r',label='0.9 < z < 1.2',lw=4.0)
# plt.axvline(np.log10(np.nanmedian(F1[B3])),ls='--',color='r',lw=3.5)
# plt.hist(np.log10(F1[B2]),bins=np.arange(40,50,0.25),histtype='step',color='b',label='0.6 < z < 0.9',lw=3.5)
# plt.axvline(np.log10(np.nanmedian(F1[B2])),ls='--',color='b',lw=3.5)
# plt.hist(np.log10(F1[B1]),bins=np.arange(40,50,0.25),histtype='step',color='gray',label='0.0 < z < 0.6',lw=3.0)
# plt.axvline(np.log10(np.nanmedian(F1[B1])),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{1 \mu \mathrm{m}}$ [erg/s]')
# plt.xlim(42,47)
# plt.legend()
# plt.show()

print(uv_lum)
print(Lbol)
print(np.log10((uv_lum+mir_lum+fir_lum)/Lbol))

plt.figure(figsize=(10,9))
plt.hist(np.log10((uv_lum+mir_lum+fir_lum)/Lbol),bins=np.arange(-1,1,0.1))
plt.xlim(-0.7,0.25)
plt.xlabel('L(UV+MIR+FIR)/Lbol')
plt.grid()
plt.show()


# plot.plot_1panel('43',all_w,all_f,Lx,spec_type,np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),median_w,median_f,F1=norm,F2=marker,suptitle=str(z_min)+' < z < '+str(z_max),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,wfir=np.asarray(WFIR),ffir=np.asarray(FFIR))

################ Paper Plots ###################
# fig 3
plot.multi_SED('TEST_backup',all_x[sort],all_y[sort],Lx[sort],median_x[sort],median_y[sort],suptitle='SEDs of X-ray',norm=F1[sort],mark=field[sort],spec_z=all_z[sort],wfir=WFIR[sort],ffir=FFIR[sort],up_check=upper_check[sort],med_x_fir=median_fir_x[sort],med_y_fir=median_fir_y[sort])

# fig 4 & 5
# plot.multi_SED_zbins('All',all_x[sort], all_y[sort], Lx[sort], all_z[sort], median_x[sort], median_y[sort], F1[sort], field[sort], spec_z=all_z[sort],wfir=WFIR[sort],ffir=FFIR[sort],up_check=upper_check[sort],med_x_fir=median_fir_x[sort],med_y_fir=median_fir_y[sort])
plot.multi_SED_field('TEST_backup', all_x[sort], all_y[sort], Lx[sort], all_z[sort], median_x[sort], median_y[sort], F1[sort], field[sort], spec_z=all_z[sort], wfir=WFIR[sort], ffir=FFIR[sort])

# fig 6
# plot.plot_5panel_zbins('43',all_x,all_y,Lx,spec_type,np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),median_x,median_y,F1=F1,F2=field,suptitle=str(z_min)+' < z < '+str(z_max),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,wfir=np.asarray(WFIR),ffir=np.asarray(FFIR))

# fig 7 
# plot.plot_median_zbins(np.asarray(FFIR),all_x,all_y,Lx,spec_type,np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),median_x,median_y,F1=F1,F2=field,median_FIR_w=np.asarray(WFIR),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2)

# fig 8 & 16
# plot.L_box_3zbins('Lx_box', Lx, F1, all_z, UVslope, MIRslope1, MIRslope2,label='X') # Need to change ylim between these plots
# plot.L_box_3zbins('Lbol_box3_43', np.log10(Lbol), F1, all_z, UVslope, MIRslope1, MIRslope2,label='bol')
# plot.L_box_fields('Lone_box_field_43', np.log10(F1), F1, all_z, UVslope, MIRslope1, MIRslope2,field,label='one')

# fig 9
# plot.Nh_box_3zbins('Nh_box', np.log10(Nh[Nh > 0]), F1[Nh > 0], all_z[Nh > 0], UVslope[Nh > 0], MIRslope1[Nh > 0], MIRslope2[Nh > 0])


# fig 10, 11 & 12
# plot.Emission_Scatter_Comp('UV_MIR_ALL_NEW',np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),Nh,F1=F1,F2=field,spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,up_check=upper_check)

# fig 13 & 14
# plot.Lx_Scatter_Comp('Lx_MIR_ALL_NEW',Lx,np.log10(Lbol),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),Nh,F1=F1,F2=field,spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,up_check=upper_check)

# fig 15
# plot.L_Lx_scatter_3zbins('Lx_Lbol_photz', Lx, np.log10(Lbol), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.log10(F025), np.log10(F5), np.log10(F100), np.log10(F10), F1=F1, F2=field, spec_z=all_z, uv_slope=UVslope, mir_slope1=MIRslope1, mir_slope2=MIRslope2, up_check=upper_check, fir_frac=fir_frac)


### New Plots ### 

# fig 13 & 14
# plot2.Lx_Scatter_Comp('Lx_MIR_plot_43_med_Lx','Lx','MIR','None','Lx' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_UV_plot_43_med_Lx','Lx','UV','None','Lx',Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_FIR_plot_43_med_Lx','Lx','FIR','Y','Lx',Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_Lbol','Lx','Lbol','None','Lx' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)


# # fig 10, 11 & 12
# plot2.Emission_Scatter_Comp('UV_MIR_plot_43_med_bins','MIR','UV','None','Bins' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Emission_Scatter_Comp('UV_FIR_NEWplot_43_med_bins','FIR','UV','Both','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Emission_Scatter_Comp('MIR10_FIR_NEWplot_43_normY_med_bins','FIR','MIR','Both','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.Nh_frac_plots('Nh_MIR','MIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV','UV','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV_MIR','UV/MIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV_FIR','UV/FIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_MIR_FIR','MIR/FIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_Lbol','Lbol','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F5),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

###############################################################################
###############################################################################
###############################################################################

tf = time.perf_counter()
print(f'Total time: {tf - ti:0.4f} seconds')


