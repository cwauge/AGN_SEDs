from statistics import median_grouped
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
from PlotSED_Check import IntPlot
from astropy.table import Table


ti = time.perf_counter()  # Start timer

# Path for photometry catalogs
path = '/Users/connor_auge/Research/Disertation/catalogs/'


# COSMOS2020 catalog
cosmos = fits.open(path+'COSMOS2020_CLASSIC_R1_v2.2_p3.fits')
cosmos_data = cosmos[1].data
cosmos.close()

# COSMOS2015 catalog
cosmos2015 = fits.open(path+'COSMOS2015_Laigle+_v1.1.fits')
cosmos2015_data = cosmos2015[1].data
cosmos2015.close()

# COSMOS FIR catalog
cosmos_fir = fits.open(path+'cosmos_superdeblended.fit')
cosmos_fir_data = cosmos_fir[1].data
cosmos.close()

# Chandra 2016 catalog
chandra_cosmos = fits.open(path+'chandra_COSMOS_legacy_opt_NIR_counterparts_20160113_4d.fits')
chandra_cosmos_data = chandra_cosmos[1].data
chandra_cosmos.close()

# Chandra updated fits catalog
chandra_cosmos2 = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_20210225.fits')
chandra_cosmos2_data = chandra_cosmos2[1].data
chandra_cosmos2.close()

# Chandra compton thick catalog
chandra_cosmos_ct = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_ComptonThick_Lanzuisi18.fits')
chandra_cosmos_ct_data = chandra_cosmos_ct[1].data
chandra_cosmos_ct.close()


# kelly = ascii.read('/Users/connor_auge/Research/REU/2022/Thresa/Kelly_sample2.csv')
kelly = ascii.read('/Users/connor_auge/Research/REU/2022/Thresa/2023_cont/v2_AGN_tricolor_data_new.csv')
kelly_id = np.asarray(kelly['ID'])
kelly_z = np.asarray(kelly['z'])
kelly_Lx = np.asarray(kelly['Lx'])
kelly_group = np.asarray(kelly['Group'])

kelly2 = ascii.read('/Users/connor_auge/Research/REU/2022/Thresa/COSMOS_z_matches_new_test.csv')
kelly2_id = np.asarray(kelly2['ID'])
kelly2_z = np.asarray(kelly2['z'])

for i in range(len(kelly_id)):
    ind = np.where(kelly2_id == kelly_id[i])[0]
    if len(ind) > 0:
        if kelly2_z[ind][0] > 0:
            kelly_z[i] = kelly2_z[ind][0]
        else:
            continue
    else:
        continue


# DEIMOS 10k Spec z cat
deimos = ascii.read(
    '/Users/connor_auge/Downloads/deimos_10k_March2018_new/deimos_redshifts.tbl')
deimos_id = np.asarray(deimos['ID'])
deimos_z = np.asarray(deimos['zspec'])
deimos_remarks = np.asarray(deimos['Remarks'])
deimos_ID = np.asarray([int(i[1:]) for i in deimos_id if 'L' in i])
deimos_z_spec = np.asarray(
    [deimos_z[i] for i in range(len(deimos_z)) if 'L' in deimos_id[i]])

# Gather all IDs
chandra_cosmos_phot_id = chandra_cosmos_data['id_k_uv']
cosmos_laigle_id = cosmos_data['ID_COSMOS2015']
cosmos_fir_id = cosmos_fir_data['ID']
cosmos2015_ID = cosmos2015_data['NUMBER']
cosmos_xid = cosmos_data['id_chandra']
chandra_cosmos_xid = chandra_cosmos_data['id_x']
chandra_cosmos2_xid = chandra_cosmos2_data['id_x']
chandra_cosmos_ct_xid = chandra_cosmos_ct_data['id_x']

# X-ray coords
chandra_cosmos_RA = chandra_cosmos_data['RA_x']
chandra_cosmos_DEC = chandra_cosmos_data['DEC_x']

# Redshfits
# cosmos_sz = cosmos_data['sz_zspec']
# cosmos_ez = cosmos_data['ez_z_spec']
chandra_cosmos_z = chandra_cosmos_data['z_spec']
chandra_cosmos_z_phot = chandra_cosmos_data['z_best']

chandra_cosmos2_z = chandra_cosmos2_data['z_best']

# X-ray Flux
chandra_cosmos_Fx_hard = chandra_cosmos_data['flux_h']
chandra_cosmos_Fx_soft = chandra_cosmos_data['flux_s']
chandra_cosmos_Fx_full = chandra_cosmos_data['flux_f']

chandra_cosmos2_Fx_hard = chandra_cosmos2_data['flux_210']
chandra_cosmos2_Fx_soft = chandra_cosmos2_data['flux_052']
chandra_cosmos2_Fx_full = chandra_cosmos2_data['flux_0510']

# X-ray Luminosity (non-log)
chandra_cosmos_Lx_hard = np.asarray(
    [10**i for i in chandra_cosmos_data['Lx_210']])
chandra_cosmos_Lx_soft = np.asarray(
    [10**i for i in chandra_cosmos_data['Lx_052']])
chandra_cosmos_Lx_full = np.asarray(
    [10**i for i in chandra_cosmos_data['Lx_0510']])

chandra_cosmos2_Lx_hard = np.asarray(
    [10**i for i in chandra_cosmos2_data['Lx_210']])
# Correction from hard to full band
chandra_cosmos2_Lx_full = np.asarray(
    [(10**i)*1.64 for i in chandra_cosmos2_data['Lx_210']])

# Other Chandra Data
# Spec-type from hardness ratio
chandra_cosmos_spec_type = chandra_cosmos_data['spec_type']  # spec type

# Column Density
chandra_cosmos_nh = chandra_cosmos_data['Nh']
chandra_cosmos_nh_lo = chandra_cosmos_data['Nh_lo']
chandra_cosmos_nh_hi = chandra_cosmos_data['Nh_up']

chandra_cosmos2_nh = chandra_cosmos2_data['nh']
chandra_cosmos2_nh_lo_err = chandra_cosmos2_data['nh_lo_err']
chandra_cosmos2_nh_up_err = chandra_cosmos2_data['nh_up_err']


# Absorption Correction - full, hard, and soft bands with uppr and lower limts
chandra_cosmos_abs_corr_h = chandra_cosmos_data['abs_corr_210']
chandra_cosmos_abs_corr_up_h = chandra_cosmos_data['abs_corr_210_up']
chandra_cosmos_abs_corr_lo_h = chandra_cosmos_data['abs_corr_210_lo']
chandra_cosmos_abs_corr_s = chandra_cosmos_data['abs_corr_052']
chandra_cosmos_abs_corr_up_s = chandra_cosmos_data['abs_corr_052_up']
chandra_cosmos_abs_corr_lo_s = chandra_cosmos_data['abs_corr_052_lo']
chandra_cosmos_abs_corr_f = chandra_cosmos_data['abs_corr_0510']
chandra_cosmos_abs_corr_up_f = chandra_cosmos_data['abs_corr_0510_up']
chandra_cosmos_abs_corr_lo_f = chandra_cosmos_data['abs_corr_0510_lo']


# X-ray luminosities and Column densites from the Chandra Compoton thick catalog
chandra_cosmos_ct_Lx_hard_obs = np.asarray(
    [10**i for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_full_obs = np.asarray(
    [(10**i)*1.64 for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_hard = np.asarray(
    [10**i for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_Lx_full = np.asarray(
    [(10**i)*1.64 for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_nh = chandra_cosmos_ct_data['nh2_22']

# Gather DEIMOS spec redshifts to redshift array
for i in range(len(chandra_cosmos_z)):
    ind = np.where(deimos_ID == chandra_cosmos_phot_id[i])[0]
    if len(ind) > 0:
        if deimos_z_spec[ind][0] > 0:
            chandra_cosmos_z[i] = deimos_z_spec[ind][0]
        else:
            continue
    else:
        continue

# Gather Luminosity absorption corrections from the Chandra COSMOS cat1 and correct the luminositeis
abs_corr_use_h = []
abs_corr_use_s = []
abs_corr_use_f = []
check_abs = []
for i in range(len(chandra_cosmos_abs_corr_f)):
    if chandra_cosmos_abs_corr_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_s[i])
        check_abs.append(0)

    elif chandra_cosmos_abs_corr_up_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_up_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_up_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_up_s[i])
        check_abs.append(1)

    elif chandra_cosmos_abs_corr_lo_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_lo_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_lo_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_lo_s[i])
        check_abs.append(2)

    else:
        print('NO GOOD ABSORPTION CORRECTION DATA')

check_abs = np.asarray(check_abs)
# Turn final absorption lists into arrays
abs_corr_use_h = np.asarray(abs_corr_use_h)
abs_corr_use_s = np.asarray(abs_corr_use_s)
abs_corr_use_f = np.asarray(abs_corr_use_f)

# Correct the X-ray luminosity from the 2016 Chandra Catalog for absorption
chandra_cosmos_Lx_full_obs = chandra_cosmos_Lx_full.copy()
chandra_cosmos_Lx_hard_obs = chandra_cosmos_Lx_hard.copy()
chandra_cosmos_Lx_sofr_obs = chandra_cosmos_Lx_soft.copy()

chandra_cosmos_Lx_hard /= abs_corr_use_h
chandra_cosmos_Lx_soft /= abs_corr_use_s
chandra_cosmos_Lx_full /= abs_corr_use_f

# Gather the column density from the three chandra catalogs for the full and hard band Lx
chandra_cosmos_Nh = []
check = []
cosmos_Nh_check = []
for i in range(len(chandra_cosmos_Lx_full)):
    # Check if there is a match to updated Chandra catalog
    ind = np.where(chandra_cosmos2_xid == chandra_cosmos_xid[i])[0]
    # Check if there is a match to compton thick Chandra catalog
    ind_ct = np.where(chandra_cosmos_ct_xid == chandra_cosmos_xid[i])[0]

    if len(ind_ct) > 0:
        # if there is a match append Nh from compton thick catalog
        chandra_cosmos_Nh.append(chandra_cosmos_ct_nh[ind_ct][0])
        # replace Lx from original Chandra catalog with that from the CT cat
        chandra_cosmos_Lx_hard[i] = chandra_cosmos_ct_Lx_hard[ind_ct]
        chandra_cosmos_Lx_full[i] = chandra_cosmos_ct_Lx_full[ind_ct]
        abs_corr_use_f[i] = chandra_cosmos_Lx_full_obs[i]/chandra_cosmos_ct_Lx_full[ind_ct]
        abs_corr_use_h[i] = chandra_cosmos_Lx_hard_obs[i]/chandra_cosmos_ct_Lx_hard[ind_ct]
        check.append(3)  # count which catalog data is from
        cosmos_Nh_check.append(0)
        check_abs[i] = 0

    elif len(ind) > 0:
        # replace Lx from orginal Chandra catalog with that from updated cat
        chandra_cosmos_Lx_hard[i] = chandra_cosmos2_Lx_hard[ind]
        chandra_cosmos_Lx_full[i] = chandra_cosmos2_Lx_full[ind]
        abs_corr_use_f[i] = chandra_cosmos_Lx_full_obs[i]/chandra_cosmos2_Lx_full[ind]
        abs_corr_use_h[i] = chandra_cosmos_Lx_hard_obs[i]/chandra_cosmos2_Lx_hard[ind]
        check_abs[i] = 0
        if chandra_cosmos2_nh_lo_err[ind][0] == -99.:
            # if there is Nh upper limit in updated cat append to Nh list
            chandra_cosmos_Nh.append(
                chandra_cosmos2_nh[ind][0]+chandra_cosmos2_nh_up_err[ind][0])
            # chandra_cosmos_Nh.append(0.0)
            check.append(2.5)
            cosmos_Nh_check.append(1)
        else:
            # if there is a match append Nh from updated catalog
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0])
            check.append(2)
            cosmos_Nh_check.append(0)
    else:  # if no matches to updated or CT catalogs take Nh value from original catalog
        # If no good value take upper or lower limits
        if chandra_cosmos_nh[i] == -99.:
            if chandra_cosmos_nh_lo[i] != -99.:
                # chandra_cosmos_Nh.append(0.0)
                chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
                cosmos_Nh_check.append(2)
            else:
                # chandra_cosmos_Nh.append(0.0)
                chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
                cosmos_Nh_check.append(1)
        else:
            chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
            cosmos_Nh_check.append(0)
        check.append(1)
chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E22
check = np.asarray(check)
cosmos_Nh_check = np.asarray(cosmos_Nh_check)


print('COSMOS All Lx cat: ', len(chandra_cosmos_Lx_full))


ix, iy = match(kelly_id,cosmos_laigle_id)

cosmos_laigle_id_match = cosmos_laigle_id[iy]

kelly_id_match = kelly_id[ix]
kelly_z_match = kelly_z[ix]
kelly_group_match = kelly_group[ix]
kelly_Lx_match = kelly_Lx[ix]
Lx_match = []
Nh_match = []
abs_corr_match = []
abs_corr_check_match = []
for i in range(len(kelly_id_match)):
    ind = np.where(chandra_cosmos_phot_id == kelly_id_match[i])[0]
    if len(ind > 0):
        Lx_match.append(chandra_cosmos_Lx_full[ind][0])
        Nh_match.append(chandra_cosmos_Nh[ind][0])
        abs_corr_match.append(abs_corr_use_f[ind][0])
        abs_corr_check_match.append(check_abs[ind][0])
    else:
        Lx_match.append(-999.9)
        Nh_match.append(-999.9)
        abs_corr_match.append(-999.9)
        abs_corr_check_match.append(-999.9)

Lx_match = np.asarray(Lx_match)
Nh_match = np.asarray(Nh_match)
abs_corr_match = np.asarray(abs_corr_match)
abs_corr_check_match = np.asarray(abs_corr_check_match)

cosmos_ix, cosmos_iy = match(cosmos_laigle_id_match,cosmos2015_ID)



# Create flux and flux error arrays for the COSMOS data. Matched to chandra data. NaN array separating the X-ray from the FUV data.
cosmos_flux_array = np.array([
    cosmos_data['GALEX_FUV_FLUX'][iy],
    cosmos_data['GALEX_NUV_FLUX'][iy],
    cosmos_data['CFHT_u_FLUX_APER2'][iy],
    cosmos_data['HSC_g_FLUX_APER2'][iy],
    cosmos_data['HSC_r_FLUX_APER2'][iy],
    cosmos_data['HSC_i_FLUX_APER2'][iy],
    cosmos_data['HSC_z_FLUX_APER2'][iy],
    cosmos_data['HSC_y_FLUX_APER2'][iy],
    cosmos_data['UVISTA_J_FLUX_APER2'][iy],
    cosmos_data['UVISTA_H_FLUX_APER2'][iy],
    cosmos_data['UVISTA_Ks_FLUX_APER2'][iy],
    cosmos_data['SPLASH_CH1_FLUX'][iy],
    cosmos_data['SPLASH_CH2_FLUX'][iy],
    cosmos_data['SPLASH_CH3_FLUX'][iy],
    cosmos_data['SPLASH_CH4_FLUX'][iy],
    cosmos2015_data['Flux_24'][cosmos_iy],
    cosmos2015_data['Flux_100'][cosmos_iy]*1000,
    cosmos2015_data['Flux_160'][cosmos_iy]*1000,
    cosmos2015_data['Flux_250'][cosmos_iy]*1000,
    cosmos2015_data['Flux_350'][cosmos_iy]*1000,
    cosmos2015_data['Flux_500'][cosmos_iy]*1000
])

cosmos_flux_err_array = np.array([
    cosmos_data['GALEX_FUV_FLUXERR'][iy],
    cosmos_data['GALEX_NUV_FLUXERR'][iy],
    cosmos_data['CFHT_u_FLUXERR_APER2'][iy],
    cosmos_data['HSC_g_FLUXERR_APER2'][iy],
    cosmos_data['HSC_r_FLUXERR_APER2'][iy],
    cosmos_data['HSC_i_FLUXERR_APER2'][iy],
    cosmos_data['HSC_z_FLUXERR_APER2'][iy],
    cosmos_data['HSC_y_FLUXERR_APER2'][iy],
    cosmos_data['UVISTA_J_FLUXERR_APER2'][iy],
    cosmos_data['UVISTA_H_FLUXERR_APER2'][iy],
    cosmos_data['UVISTA_Ks_FLUXERR_APER2'][iy],
    cosmos_data['SPLASH_CH1_FLUXERR'][iy],
    cosmos_data['SPLASH_CH2_FLUXERR'][iy],
    cosmos_data['SPLASH_CH3_FLUXERR'][iy],
    cosmos_data['SPLASH_CH4_FLUXERR'][iy],
    cosmos2015_data['Fluxerr_24'][cosmos_iy],
    cosmos2015_data['Fluxerr_100'][cosmos_iy]*1000,
    cosmos2015_data['Fluxerr_160'][cosmos_iy]*1000,
    cosmos2015_data['Fluxerr_250'][cosmos_iy]*1000,
    cosmos2015_data['Fluxerr_350'][cosmos_iy]*1000,
    cosmos2015_data['Fluxerr_500'][cosmos_iy]*1000
])

# Transpose arrays so each row is a new source and each column is a obs filter
cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T


# Print time taken to read in all files
tfl = time.perf_counter()
print(f'Done with file reading ({tfl - ti:0.4f} second)')


#### Read in Galaxy Templates ####
temp = ascii.read('/Users/connor_auge/Research/templets/A10_templates.txt')
temp_wave = np.asarray(temp['Wave'])
temp_flux = np.asarray(temp['E'])*1E-16  # erg/s/cm^-2/Hz
temp_wave_cgs = temp_wave*1E-8
temp_freq = 3E10/temp_wave_cgs
temp_nuFnu = temp_flux*temp_freq

dl = 10
dl_cgs = dl*3.086E18
temp_lum = temp_nuFnu*4*np.pi*dl_cgs**2

scale_array = [1.12E44, 1.72E44, 2.27E44]
###################################

#### Read in M82 Templates ####
m82 = ascii.read('/Users/connor_auge/Research/templets/M82.csv')
m82_wave = np.asarray(m82['wave'])
m82_lum = np.asarray(m82['lum'])
###################################


# Slope lims
uv1, uv2 = 0.2, 1.0
mir11, mir12 = 1.0, 6.0
mir21, mir22 = 6.0, 10


COSMOS_filters = np.array(['FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'JVHS', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])



###################################################################################
###################################################################################
###################################################################################
########################## Run AGN Class over each source #########################

# Make empty lists to be filled with SED outputs
out_ID, out_z, out_x, out_y, out_frac_error = [], [], [], [], []
out_Lx, out_Lx_hard, out_Lx_soft = [], [], []
out_SED_shape = []
check_sed = []
norm = []
wfir_out, ffir_out = [], []
int_x, int_y = [], []
FIR_upper_lims = []
F025, F1, F6, F10, F100 = [], [], [], [], []
F2 = []
xval_out, F2kev = [], []
field = []
uv_slope, mir_slope1, mir_slope2 = [], [], []
Lbol_out, Lbol_sub_out = [], []
Nh = []
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = [], [], [], []
X_UV_lum_out = []
FIR_R_lum = []
Nh_check = []
abs_check = []
mix_x, mix_y = [], []
F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = [], [], [], [], []
F2_boot = []
xval_out_boot, F2kev_boot = [], []
spec_class = []
check_sed6 = []
absorption_corr = []

F24_out = []
F100_ratio = []
stack_bin = []

group_Lx = []

group = []

Lbol_sf_sub_out = []
###############################################################################
###############################################################################
############################### Run COSMOS SEDs ###############################
# '''
nustar_id = np.asarray(
    [215929, 317570, 338071, 348646, 397900, 420912, 450055, 452180, 486971, 487826, 494577, 508074, 529855, 567452, 572088, 589540, 609017, 635561, 662467, 859808, 919147])

cigale_count = 0
for i in range(len(kelly_id_match)):
    # for i in range(50):
    # if chandra_cosmos_phot_id_match[i] in nustar_id:
    source = AGN(kelly_id_match[i], kelly_z_match[i],
                 COSMOS_filters, cosmos_flux_array[i], cosmos_flux_err_array[i])
    source.MakeSED(data_replace_filt=['FLUX_24'])
    source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160',
                       'FLUX_250', 'FLUX_350', 'FLUX_500'])

    cosmos_flux_dict = source.MakeDict(
        COSMOS_filters, cosmos_flux_array[i])
    cosmos_flux_err_dict = source.MakeDict(
        COSMOS_filters, cosmos_flux_err_array[i])

    Lcheck = source.Lum_filter('FLUX_24')
    F24_out.append(Lcheck)
    stack_bin.append(7)

    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    int_x.append(ix)
    int_y.append(iy)

    # wfir, ffir, f100, f100_boot = source.Int_SED_FIR(Find_value=100.0,discreet=True,boot=True)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    wfir_out.append(wfir)
    ffir_out.append(ffir)

    L100_ratio = source.flux_ratio_lower()
    F100_ratio.append(L100_ratio)

    # f1, f1_boot = source.Find_value(1.0, boot=True)
    # f2, f2_boot = source.Find_value(2.0, boot=True)
    # xval, xval_boot = source.Find_value(3E-4, boot=True)
    # f6, f6_boot = source.Find_value(6.0, boot=True)
    # f025, f025_boot = source.Find_value(0.25, boot=True)
    # f10, f10_boot = source.Find_value(10, boot=True)
    # f2kev, f2kev_boot = source.Find_value(6.1992e-4, boot=True)

    # f015, f015_boot = source.Find_value(0.15, boot=True)
    # f65, f65_boot = source.Find_value(6.5, boot=True)

    f1 = source.Find_value(1.0)
    f2 = source.Find_value(2.0)
    xval = source.Find_value(3E-4)
    f6 = source.Find_value(6.0)
    f025 = source.Find_value(0.25)
    f10 = source.Find_value(10)
    f2kev = source.Find_value(6.1992e-4)

    f015 = source.Find_value(0.2)
    f65 = source.Find_value(6.5)

    lbol = source.Find_Lbol()
    lbol_sf_sub = source.Find_Lbol(
        sub=True, Lscale=scale_array, Lnorm=f1, temp_x=m82_wave, temp_y=m82_lum)
    # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
    # lbol_sf_sub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum)
    # lbol_sf_sub = source.Find_Lbol(xmax=15)
    # lbol_sf_sub = np.nan
    # lbol_sf_sub, tx, ty, xsub, ysub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum,sed=True)
    # print(np.log10(lbol),np.log10(lbol_sub),np.log10(lbol_sf_sub))
    # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum, xmax=50)
    lbol_sub = source.Find_Lbol(
        sub=True, Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)

    shape = source.SED_shape()

    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(
        norm_w=1)
    out_ID.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_frac_error.append(frac_err)
    out_Lx.append(Lx_match[i])
    out_Lx_hard.append(Lx_match[i])
    out_Lx_soft.append(Lx_match[i])
    out_z.append(kelly_z_match[i])
    FIR_upper_lims.append(up_check)

    hao_x, hao_y = source.mix_loc([1, 0.3], [3, 1])
    # hao_x, hao_y = source.mix_loc([0.3, 1.0], [1.0, 3.0])
    mix_x.append(hao_x)
    mix_y.append(hao_y)

    norm.append(f1)
    F025.append(f025)
    F1.append(f1)
    F2.append(f2)
    F6.append(f6)
    F10.append(f10)
    F100.append(f100)
    F2kev.append(f2kev)
    xval_out.append(xval)
    # F1_boot.append(f1_boot)
    # F025_boot.append(f025_boot)
    # F2_boot.append(f2_boot)
    # F6_boot.append(f6_boot)
    # F10_boot.append(f10_boot)
    # xval_out_boot.append(xval_boot)
    # F100_boot.append(f100_boot)
    # F2kev_boot.append(f2kev_boot)
    Lbol_out.append(lbol)
    Lbol_sub_out.append(lbol_sub)
    Lbol_sf_sub_out.append(lbol_sf_sub)
    Nh.append(Nh_match[i])
    Nh_check.append(cosmos_Nh_check[i])
    abs_check.append(abs_corr_check_match[i])
    absorption_corr.append(abs_corr_match[i])

    shape = source.SED_shape()
    # shape = source.SED_shape(uv1, uv2, mir11, mir12, mir21, mir22)

    uv_slope.append(source.Find_slope(uv1, uv2))
    mir_slope1.append(source.Find_slope(mir11, mir12))
    mir_slope2.append(source.Find_slope(mir21, mir22))
    out_SED_shape.append(shape)
    UV_lum_out.append(source.find_Lum_range(0.1, 0.35))
    OPT_lum_out.append(source.find_Lum_range(0.35, 3))
    MIR_lum_out.append(source.find_Lum_range(3, 30))
    FIR_lum_out.append(source.find_Lum_range(
        30, 500/(1+kelly_z_match[i])))
    X_UV_lum_out.append(source.find_Lum_range(1E-3, 0.1))
    FIR_R_lum.append(source.find_Lum_range(40, 400))

    plot = Plotter(Id, redshift, w, f,
                   f1, f1, up_check)

    check = source.check_SED(10, check_span=2.75)
    check6 = source.check_SED(6, check_span=2.75)
    check_sed.append(check)
    check_sed6.append(check6)
    field.append('COSMOS')
    spec_class.append(-999)

    group.append(kelly_group_match[i])
    group_Lx.append(kelly_Lx_match[i])

    # if kelly_group_match[i] == 'RED':
    #     if kelly_z_match[i] < 1.0:
    #         print(Id)
    #         plot.PlotSED(point_x=[0.2,1.0,6.,10,100],point_y=[f015/f1,f1/f1,f6/f1,f10/f1,f100/f1],fir_x = wfir, fir_y = ffir)#,temp_x=tx,temp_y=ty)


    # if check6 == 'GOOD':
    # # if Id == 205032:
    #     if shape == -99.:
    # # # # # # #     print(shape, source.Find_slope(uv1,uv2), source.Find_slope(mir11, mir12), source.Find_slope(mir21, mir22))
    # # # #     if shape == 3:
    #         print(Id,check6,shape,up_check,source.Find_slope(uv1, uv2),source.Find_slope(mir11, mir12),mir_slope2[i])
    #         print(np.log10(lbol),np.log10(lbol_sub),np.log10(lbol_sf_sub))
    #         plot.PlotSED(point_x=[0.2,1.0,6.,10,100],point_y=[f015/f1,f1/f1,f6/f1,f10/f1,f100/f1],fir_x = wfir, fir_y = ffir)#,temp_x=tx,temp_y=ty)

    # if check6 == 'GOOD':
    #     c = SkyCoord(ra = chandra_cosmos_RA_match[i]*u.degree, dec = chandra_cosmos_DEC_match[i]*u.degree)
    #     coords = c.to_string('hmsdms').split()

    #     cols, data, dtyp = source.output_properties('COSMOS',chandra_cosmos_xid_match[i],coords[0],coords[1],chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i])
    #     source.write_output_file('AGN_Properties_final.csv',data,cols,dtyp,'w')

    #     cols, data, dtyp = source.output_phot('COSMOS',filter_COSMOS_total,filter_COSMOS_match)
    #     source.write_output_file('AGN_photometry_cosmos_final.csv',data,cols,dtyp,'w',phot=True)

    # if Id == 847966:

    # if check == 'GOOD':
    #     int_plot = IntPlot(check_sed_fname,Id, redshift, w, f, chandra_cosmos_Lx_full_match[i],f1,up_check)
    #     int_plot.Plot([0.25,1.0,6.0,100],[f025/f1,f1/f1,f6/f1,f100/f1])

    # # if check == 'GOOD':
    #     # plot.Plot_FIR_SED(wfir, ffir/f1)
    # print(chandra_cosmos_phot_id_match[i])
    # print('shape: ', shape)
    # print('uv: ', source.Find_slope(0.15, 1.0))
    # print('mir1: ', source.Find_slope(1.0, 6.5))
    # print('mir2: ', source.Find_slope(6.5, 10))
    # plot.PlotSED(point_x=100,point_y=f100/f1)

    # elif Id == 662467:
    # if check == 'GOOD':
    # if Id == 687415:
    # plot.Plot_FIR_SED(wfir, ffir/f1)
    #     source.write_cigale_file(cigale_name,int_fx=cosmos_Fx_int_array[i],use_int_fx=True)
    # plot.PlotSED(point_x=100,point_y=f100/f1)

    # if Id == 536989 or Id == 576750:
    #     # if check == 'GOOD':
    #     # plot.Plot_FIR_SED(wfir, ffir/f1)
    #     plot.PlotSED(point_x=100, point_y=f100/f1)
    # source.Find_Lbol()

    #     if check6 == 'GOOD':
    #         if shape == 5:
    #             cigale_count += 1
    #             if (cigale_count > 60) & (cigale_count <= 90):
    #             # if cigale_count <= 30:
    #                 source.write_cigale_file2(cigale_name, COSMOS_CIGALE_filters, cosmos_flux_dict, cosmos_flux_err_dict, int_fx=cosmos_Fx_int_array[0][i])
    #                 # source.write_cigale_file(cigale_name,int_fx=cosmos_Fx_int_array[i],use_int_fx=True)
    #             else:
    #                 continue
    #         else:
    #             continue
    #     else:
    #         continue

    # # else:
    #     # continue


# '''
tc = time.perf_counter()
print(f'Done with COSMOS sources ({tc - tfl:0.4f} second)')


out_ID, out_z, out_x, out_y, out_frac_error = np.asarray(out_ID), np.asarray(out_z), np.asarray(out_x), np.asarray(out_y), np.asarray(out_frac_error)
out_Lx, out_Lx_hard = np.log10(np.asarray(out_Lx)), np.log10(np.asarray(out_Lx_hard))
wfir_out, ffir_out = np.asarray(wfir_out), np.asarray(ffir_out)
int_x, int_y = np.asarray(int_x), np.asarray(int_y)
norm = np.asarray(norm)
FIR_upper_lims = np.asarray(FIR_upper_lims)
F025, F1, F6, F10, F100 = np.asarray(F025), np.asarray(
    F1), np.asarray(F6), np.asarray(F10), np.asarray(F100)
F2 = np.asarray(F2)
xval_out = np.asarray(xval_out)
group = np.asarray(group)
group_Lx = np.asarray(group_Lx)

# F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(F025_boot)[GOOD_SED], np.asarray(F1_boot)[GOOD_SED], np.asarray(F6_boot)[GOOD_SED], np.asarray(F10_boot)[GOOD_SED], np.asarray(F100_boot)[GOOD_SED]
# F2_boot = np.asarray(F2_boot)[GOOD_SED]
# xval_out_boot = np.asarray(xval_out_boot)[GOOD_SED]

F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(
    F025), np.asarray(F1), np.asarray(F6), np.asarray(F10), np.asarray(F100)
F2_boot = np.asarray(F2)
xval_out_boot = np.asarray(xval_out)

field = np.asarray(field)
out_SED_shape = np.asarray(out_SED_shape)
uv_slope, mir_slope1, mir_slope2 = np.asarray(
    uv_slope), np.asarray(mir_slope1), np.asarray(mir_slope2)
Lbol_out, Lbol_sub_out = np.asarray(Lbol_out), np.asarray(Lbol_sub_out)
Nh = np.asarray(Nh)
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = np.asarray(UV_lum_out), np.asarray(
    OPT_lum_out), np.asarray(MIR_lum_out), np.asarray(FIR_lum_out)
X_UV_lum_out = np.asarray(X_UV_lum_out)
FIR_R_lum = np.asarray(FIR_R_lum)
Nh_check = np.asarray(Nh_check)
abs_check = np.asarray(abs_check)
spec_class = np.asarray(spec_class)

mix_x, mix_y = np.asarray(mix_x), np.asarray(mix_y)

check_sed = check_sed
check_sed6 = np.asarray(check_sed6)

F24_out = np.asarray(F24_out)
stack_bin = np.asarray(stack_bin)
F100_ratio = np.asarray(F100_ratio)

Lbol_sf_sub_out = np.asarray(Lbol_sf_sub_out)

print(out_y)
print(out_y.shape)
print('check dtype: ', out_y.dtype)


# values, base = np.histogram(FIR_lum_out[FIR_upper_lims == 1]/Lbol_out[FIR_upper_lims == 1], bins=40)
# #evaluate the cumulative
# cumulative = np.cumsum(values)
# # plot the cumulative function
# plt.plot(base[:-1], cumulative, c='blue')
# plt.show()


# # Sort all output data by the intrinsic X-ray luminosity
# sort = out_Lx.argsort()
# out_ID, out_z, out_x, out_y, out_frac_error = out_ID[sort], out_z[sort], out_x[sort], out_y[sort], out_frac_error[sort]
# out_Lx, out_Lx_hard = out_Lx[sort], out_Lx_hard[sort]
# wfir_out, ffir_out = wfir_out[sort], ffir_out[sort]
# int_x, int_y = int_x[sort], int_y[sort]
# norm = norm[sort]
# FIR_upper_lims = FIR_upper_lims[sort]
# F025, F1, F6, F10, F100 = F025[sort], F1[sort], F6[sort], F10[sort], F100[sort]
# F2 = F2[sort]
# xval_out = xval_out[sort]

# # F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = F025_boot[sort], F1_boot[sort], F6_boot[sort], F10_boot[sort], F100_boot[sort]
# # F2_boot = F2_boot[sort]
# # xval_out_boot = xval_out_boot[sort]

# field = field[sort]
# out_SED_shape = out_SED_shape[sort]
# uv_slope, mir_slope1, mir_slope2 = uv_slope[sort], mir_slope1[sort], mir_slope2[sort]
# Lbol_out, Lbol_sub_out = Lbol_out[sort], Lbol_sub_out[sort]
# Nh = Nh[sort]
# UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = UV_lum_out[sort], OPT_lum_out[sort], MIR_lum_out[sort], FIR_lum_out[sort]
# FIR_R_lum = FIR_R_lum[sort]
# Nh_check = Nh_check[sort]
# abs_check = abs_check[sort]
# spec_class = spec_class[sort]
# mix_x, mix_y = mix_x[sort], mix_y[sort]

# check_sed = check_sed[sort]

# F24_out = F24_out[sort]
# stack_bin = stack_bin[sort]
# F100_ratio = F100_ratio[sort]

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
bin_stack_out = []
for i in range(len(out_ID)):
    if out_ID[i] in stack_bin1_ID:
        bin_stack_out.append(1)
    elif out_ID[i] in stack_bin2_ID:
        bin_stack_out.append(2)
    elif out_ID[i] in stack_bin3_ID:
        bin_stack_out.append(3)
    elif out_ID[i] in stack_bin4_ID:
        bin_stack_out.append(4)
    elif out_ID[i] in stack_bin5_ID:
        bin_stack_out.append(5)
    elif out_ID[i] in stack_bin6_ID:
        bin_stack_out.append(6)
    elif out_ID[i] in stack_bin7_ID:
        bin_stack_out.append(7)
    elif out_ID[i] in stack_bin8_ID:
        bin_stack_out.append(8)
    else:
        bin_stack_out.append(9)

# print(bin_stack_out)
bin_stack_out = np.asarray(bin_stack_out)
# print(len(bin_stack_out))
print('Here: ')
print(out_x)
print('Check 1: ', type(out_ID), type(field), type(out_Lx), type(out_Lx[0]))
print('Check 2: ', type(out_x), print(type(out_x[0])), print(out_x.dtype))

print(len(Lbol_out), len(Lbol_sf_sub_out))
print(len(out_ID), len(field), len(out_SED_shape), len(out_z), len(out_x), len(out_y), len(out_frac_error),
      len(out_Lx), len(out_Lx_hard), len(wfir_out), len(ffir_out), len(
          int_x), len(int_y), len(norm), len(FIR_upper_lims),
      len(F025), len(F025_boot), len(F1), len(F1_boot), len(F6), len(F6_boot), len(F10), len(F10_boot), len(
          F100), len(F100_boot), len(F2), len(F2_boot), len(uv_slope), len(mir_slope1), len(mir_slope2),
      len(Lbol_out), len(Lbol_sub_out), len(Lbol_sf_sub_out), len(Nh), len(UV_lum_out), len(OPT_lum_out), len(MIR_lum_out), len(FIR_lum_out), len(FIR_R_lum), len(Nh_check), len(abs_check), len(mix_x), len(mix_y), len(spec_class), len(check_sed), len(F24_out), len(bin_stack_out), len(F100_ratio), len(check_sed6))
h = np.asarray(['ID', 'field', 'shape', 'z', 'x', 'y', 'frac_err', 'Lx', 'Lx_hard', 'wfir', 'ffir', 'int_x', 'int_y', 'norm', 'FIR_upper_lims',
                'F025', 'F025_boot', 'F1', 'F1_boot', 'F6', 'F6_boot', 'F10', 'F10_boot', 'F100', 'F100_boot', 'F2', 'F2_boot', 'uv_slope', 'mir_slope1', 'mir_slope2',
                'Lbol', 'Lbol_sub', 'Lbol_sf_sub', 'Nh', 'UV_lum', 'OPT_lum', 'MIR_lum', 'FIR_lum', 'FIR_R_lum', 'Nh_check', 'abs_check','abs_corr', 'mix_x', 'mix_y', 'spec_class', 'sed_check', 'F24_lum', 'stack_bin', 'F100_ratio', 'check6','group','kelly_Lx'])
t = Table([out_ID, field, out_SED_shape, out_z, out_x, out_y, out_frac_error,
           out_Lx, out_Lx_hard, wfir_out, ffir_out, int_x, int_y, norm, FIR_upper_lims,
           F025, F025_boot, F1, F1_boot, F6, F6_boot, F10, F10_boot, F100, F100_boot, F2, F2_boot, uv_slope, mir_slope1, mir_slope2,
           Lbol_out, Lbol_sub_out, Lbol_sf_sub_out, Nh, UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out, FIR_R_lum, Nh_check, abs_corr_check_match, abs_corr_match, mix_x, mix_y, spec_class, check_sed, F24_out, bin_stack_out, F100_ratio, check_sed6, group, group_Lx], names=(h))

t.write('/Users/connor_auge/Research/Disertation/catalogs/Kelly_SEDs_out3_test.fits',
        format='fits', overwrite=True)
