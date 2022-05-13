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
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr


ti = time.perf_counter()

# Path for photometry catalogs
path = '/Users/connor_auge/Research/Disertation/catalogs/'

# set redshift and X-ray luminosity limits
z_min = 0.0
z_max = 1.2

Lx_min = 43
Lx_max = 50


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

lamassa_id_use = []
for i in range(len(lamassa_id)):
    if lamassa_id[i] == 0:
        lamassa_id_use.append(lamassa_id2[i])
    else:
       lamassa_id_use.append(lamassa_id[i])
lamassa_id_use = np.asarray(lamassa_id_use)

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
s82x_Fx_hard = lamassa_data['HARD_FLUX']
s82x_Fx_soft = lamassa_data['SOFT_FLUX']
s82x_Fx_full = lamassa_data['FULL_FLUX']
s82x_W1 = lamassa_data['W1']
s82x_W2 = lamassa_data['W2']
s82x_W3 = lamassa_data['W3']
s82x_W4 = lamassa_data['W4']
s82x_W1_err = lamassa_data['W1_err']
s82x_W2_err = lamassa_data['W2_err']
s82x_W3_err = lamassa_data['W3_err']
s82x_W4_err = lamassa_data['W4_err']
s82x_spec_class = lamassa_data['SPEC_CLASS']

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


ix, iy = match(lamassa_id_use, unwise_ID)

plt.plot(s82x_W1[ix], unwise_W1[iy], '.')
plt.plot(np.arange(0, 30), np.arange(0, 30), color='k')
plt.xlim(22, 10)
plt.ylim(22, 10)
plt.show()

for i in range(len(lamassa_id_use)):
    ind = np.where(unwise_ID == lamassa_id_use[i])[0]
    if len(ind) > 0:
        if np.isnan(unwise_W3[ind]):
            continue
        elif unwise_W3[ind] <= 0.0:
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
        elif (magerr_to_fluxerr(unwise_W4[ind], unwise_W4_err[ind], 'W4', AB=True)/mag_to_flux(unwise_W4[ind], 'W4')) > 0.4:
            continue
        else:
            s82x_W4[i] = unwise_W4[ind][0]
            s82x_W4_err[i] = unwise_W4_err[ind][0]

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


print('S82X All: ', len(lamassa_id_use))


# Select S82X sources from the Ananna2017 catalog based on conditions listed at the top of the file
s82x_condition = (lamassa_z > z_min) & (lamassa_z < z_max) & (np.logical_and(lamassa_ra >= 13, lamassa_ra <= 37)) & (np.logical_and(np.log10(s82x_Lx_sp_full) <= Lx_max, np.log10(s82x_Lx_sp_full) >= Lx_min))


lamassa_id_use = lamassa_id_use[s82x_condition]
lamassa_cat = lamassa_cat[s82x_condition]
s82x_z_sp = lamassa_z[s82x_condition]
lamassa_ra = lamassa_ra[s82x_condition]
lamassa_dec = lamassa_dec[s82x_condition]
s82x_Lx_sp_full = s82x_Lx_sp_full[s82x_condition]
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
	mag_to_flux(s82x_W1[s82x_condition], 'W1')*1E6,
	mag_to_flux(s82x_W2[s82x_condition], 'W2')*1E6,
	mag_to_flux(s82x_W3[s82x_condition], 'W3')*1E6,
	mag_to_flux(s82x_W4[s82x_condition], 'W4')*1E6,
	s82x_nan_array,
	lamassa_data['F250'][s82x_condition]*1000,
	lamassa_data['F350'][s82x_condition]*1000,
	lamassa_data['F500'][s82x_condition]*1000
])


s82x_flux_err_array = np.array([
	s82x_Fx_hard_err_match_mjy*1000, s82x_Fx_soft_err_match_mjy*1000,
	s82x_nan_array,
	magerr_to_fluxerr(lamassa_data['mag_FUV'][s82x_condition],
	                  lamassa_data['magerr_FUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['mag_NUV'][s82x_condition],
	                  lamassa_data['magerr_NUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['u'][s82x_condition],
	                  lamassa_data['u_err'][s82x_condition], 'sloan_u')*1E6,
	magerr_to_fluxerr(lamassa_data['g'][s82x_condition],
	                  lamassa_data['g_err'][s82x_condition], 'sloan_g')*1E6,
	magerr_to_fluxerr(lamassa_data['r'][s82x_condition],
	                  lamassa_data['r_err'][s82x_condition], 'sloan_r')*1E6,
	magerr_to_fluxerr(lamassa_data['i'][s82x_condition],
	                  lamassa_data['i_err'][s82x_condition], 'sloan_i')*1E6,
	magerr_to_fluxerr(lamassa_data['z'][s82x_condition],
	                  lamassa_data['z_err'][s82x_condition], 'sloan_z')*1E6,
	magerr_to_fluxerr(lamassa_data['JVHS'][s82x_condition],
	                  lamassa_data['JVHS_err'][s82x_condition], 'JVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['HVHS'][s82x_condition],
	                  lamassa_data['HVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['KVHS'][s82x_condition],
	                  lamassa_data['KVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
	magerr_to_fluxerr(s82x_W1[s82x_condition],
	                  s82x_W1_err[s82x_condition], 'W1', AB=True)*1E6,
	magerr_to_fluxerr(s82x_W2[s82x_condition],
	                  s82x_W2_err[s82x_condition], 'W2', AB=True)*1E6,
	magerr_to_fluxerr(s82x_W3[s82x_condition],
	                  s82x_W3_err[s82x_condition], 'W3', AB=True)*1E6,
	magerr_to_fluxerr(s82x_W4[s82x_condition],
	                  s82x_W4_err[s82x_condition], 'W4', AB=True)*1E6,
	s82x_nan_array,
	lamassa_data['F250_err'][s82x_condition]*1000,
	lamassa_data['F350_err'][s82x_condition]*1000,
	lamassa_data['F500_err'][s82x_condition]*1000
])

s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T


S82X_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
                          'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'nan', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

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
upper_check, fir_frac = [], []
FFIR_2, WFIR_2, F100_2 = [], [], []
s82x_Lx = []
check_sed = []

###############################################################################
###############################################################################
############################# Run Stripe82X SEDs ##############################
# fill_nan = np.zeros(len(GOODSS_auge_filters)-len(S82X_filters))
# fill_nan[fill_nan == 0] = np.nan

# '''
bad_id = []
for i in range(len(lamassa_id_use)):
        # print(i)
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
            source = AGN(lamassa_id_use[i], s82x_z_sp[i],
                         S82X_filters, s82x_flux_array[i], s82x_flux_err_array[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10, check_span=2.5))

            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F5.append(source.Find_nuFnu(5.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(
                ['W4', 'FLUX_250', 'FLUX_350_s82x', 'FLUX_500_s82x'], Find_value=100.0)
            ffir = np.append(np.array([np.nan, np.nan]), ffir)
            wfir = np.append(np.array([np.nan, np.nan]), wfir)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            UVslope.append(source.Find_slope(0.15, 1.0))
            MIRslope1.append(source.Find_slope(1.0, 6.5))
            MIRslope2.append(source.Find_slope(6.5, 10))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            # w = np.append(w, fill_nan)
            # f = np.append(f, fill_nan)
            # frac_err = np.append(frac_err, fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)

            # plot = Plotter(Id,redshift,w,f,frac_err,np.log10(s82x_Lx_use_match[i]))
            # plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

            med_x, med_y = source.median_SED(['U'], ['W4'])
            median_x.append(med_x)
            median_y.append(med_y)

            Lx.append(s82x_Lx_sp_full[i])
            Nh.append(s82x_Nh[i])
            Lbol.append(source.Find_Lbol())
            fir_frac.append(source.FIR_frac())

            field.append(1)

        except ValueError:
            bad_id.append(lamassa_id_use[i])


ts = time.perf_counter()
print(f'Done with S82X sources ({ts - tfl:0.4f} second)')


check_sed = np.asarray(check_sed)
GOOD_SEDs = np.where(check_sed == 'GOOD')
print('ALL: ',len(check_sed))
print('BAD SEDs: ', len(check_sed[check_sed == 'BAD']))
print('GOOD SEDs: ', len(check_sed[check_sed == 'GOOD']))
print('Except: ', len(bad_id))
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
F100_2 = np.asarray(F100_2)[GOOD_SEDs]

plot = Plotter_Letter(all_id, all_z, all_x, all_y, all_frac_err)
sort = Lx.argsort()

plot.multi_SED('S82X_new3',all_x[sort],all_y[sort],Lx[sort],median_x[sort],median_y[sort],suptitle='SEDs of X-ray',norm=F1[sort],mark=field[sort],spec_z=all_z[sort],wfir=WFIR[sort],ffir=FFIR[sort])



### OLD S82X File read in ###

# eboss = fits.open('/Users/connor_auge/Desktop/lamassa2019.fit')
# eboss_data = eboss[1].data
# eboss.close()

# ananna = fits.open(path+'Ananna2017_S82X.fit')
# ananna_data = ananna[1].data
# ananna.close()

# ananna_phot = fits.open(path+'Ananna2017_S82X.fit')
# ananna_phot_data = ananna_phot[1].data
# ananna_phot.close()

# lamassa = fits.open(
#     path+'S82X_catalog_with_photozs_unique_Xraysrcs_likely_cps_w_mbh_new.fits')
# lamassa_data = lamassa[1].data
# lamassa.close()

# peca = ascii.read(
#     '/Users/connor_auge/Desktop/s82_spec_results/Auge_spec_results_safe.txt')
# unwise = ascii.read('/Users/connor_auge/Desktop/unwise_matches.csv')

# peca_ID = np.asarray(peca['ID'])
# peca_Lx_full = np.asarray(peca['lumin_f'])
# peca_Lx_hard = np.asarray(peca['lumin_h'])
# peca_Lx_full_obs = np.asarray(peca['lumin_of'])
# peca_Nh = np.asarray(peca['nh'])

# unwise_ID = np.asarray(unwise['ID'])
# unwise_W1 = np.asarray(unwise['unW1'])
# unwise_W1_err = np.asarray(unwise['unW1_err'])
# unwise_W2 = np.asarray(unwise['unW2'])
# unwise_W2_err = np.asarray(unwise['unW2_err'])
# unwise_W3 = np.asarray(unwise['unW3'])
# unwise_W3_err = np.asarray(unwise['unW3_err'])
# unwise_W4 = np.asarray(unwise['unW4'])
# unwise_W4_err = np.asarray(unwise['unW4_err'])

# lamassa_id = lamassa_data['rec_no']
# lamassa_id2 = lamassa_data['msid']
# lamassa_z = lamassa_data['SPEC_Z']
# ananna_phot_id = ananna_phot_data['ID']

# lamassa_id_use = []
# for i in range(len(lamassa_id)):
#     if lamassa_id[i] > 0:
#         lamassa_id_use.append(lamassa_id[i])
#     elif lamassa_id2[i] > 0:
#        lamassa_id_use.append(lamassa_id2[i])
# lamassa_id_use = np.asarray(lamassa_id_use)

# ananna_id = ananna_data['ID']
# ananna_ra = ananna_data['RAJ2000']
# ananna_dec = ananna_data['DEJ2000']
# ananna_cat = ananna_data['cat']
# s82x_z_sp = ananna_data['zsp']
# s82x_z_ph = ananna_data['zph']
# s82x_z_ph_up = ananna_data['e_zph_lc']
# s82x_z_ph_lo = ananna_data['e_zph']
# s82x_Lx_sp_full = np.asarray([10**i for i in ananna_data['lglxsp']])
# s82x_Lx_ph_full = np.asarray([10**i for i in ananna_data['lglxph']])
# s82x_Fx_hard = ananna_data['FHard']
# s82x_Fx_soft = ananna_data['FSoft']
# s82x_Fx_full = ananna_data['FFull']
# s82x_W1 = ananna_data['W1mag']
# s82x_W2 = ananna_data['W2mag']
# s82x_W3 = ananna_data['W3mag']
# s82x_W4 = ananna_data['W4mag']
# s82x_W1_err = ananna_data['e_W1mag']
# s82x_W2_err = ananna_data['e_W2mag']
# s82x_W3_err = ananna_data['e_W3mag']
# s82x_W4_err = ananna_data['e_W4mag']

# # s82x_W1 = lamassa_data['W1']
# # s82x_W2 = lamassa_data['W2']
# # s82x_W3 = lamassa_data['W3']
# # s82x_W4 = lamassa_data['W4']
# # s82x_W1_err = lamassa_data['W1_err']
# # s82x_W2_err = lamassa_data['W2_err']
# # s82x_W3_err = lamassa_data['W3_err']
# # s82x_W4_err = lamassa_data['W4_err']

# s82x_spec_class = ananna_data['spclass']

# s82x_spec_class[s82x_spec_class == 'QSO'] = '1'
# s82x_spec_class[s82x_spec_class == 'QSO_BAL'] = '1'
# s82x_spec_class[s82x_spec_class == 'QSO(BA'] = '1'
# s82x_spec_class[s82x_spec_class == 'GALAXY'] = '2'
# s82x_spec_class[s82x_spec_class == 'AGN'] = '1'
# s82x_spec_class[s82x_spec_class == 'N/A'] = '3'
# s82x_spec_class[s82x_spec_class == 'STAR'] = '3'
# s82x_spec_class[s82x_spec_class == 'NELG'] = '3'
# s82x_spec_class[s82x_spec_class == '    '] = '3'
# s82x_spec_class = np.asarray(s82x_spec_class, dtype=float)

# s82x_z_use = []
# s82x_Lx_use = []
# # for i in range(len(s82x_z_sp)):
# #     if s82x_z_sp[i] > 0:
# #         s82x_z_use.append(s82x_z_sp[i])
# #         s82x_Lx_use.append(s82x_Lx_sp_full[i])
# #     else:
# #         if s82x_z_ph_up[i] - s82x_z_ph_lo[i] < 0.5:
# #             s82x_z_use.append(s82x_z_ph[i])
# #             s82x_Lx_use.append(s82x_Lx_ph_full[i])
# #         else:
# #             s82x_z_use.append(s82x_z_sp[i])
# #             s82x_Lx_use.append(s82x_Lx_sp_full[i])

# s82x_z_use = s82x_z_sp
# s82x_Lx_use = s82x_Lx_sp_full

# s82x_z_use = np.asarray(s82x_z_use)
# s82x_Lx_use = np.asarray(s82x_Lx_use)

# ix, iy = match(ananna_id, unwise_ID)

# plt.plot(s82x_W1[ix], unwise_W1[iy], '.')
# plt.plot(np.arange(0, 30), np.arange(0, 30), color='k')
# plt.xlim(22, 10)
# plt.ylim(22, 10)
# plt.show()


# for i in range(len(ananna_id)):
#     ind = np.where(unwise_ID == ananna_id[i])[0]
#     if len(ind) > 0:
#         # print('W3: ', s82x_W3[i], unwise_W3[ind][0])
#         # print('W4: ', s82x_W4[i], unwise_W4[ind][0])
#         if np.isnan(unwise_W3[ind]):
#             continue
#         elif unwise_W3[ind] <= 0.0:
#             continue
#         elif (magerr_to_fluxerr(unwise_W3[ind], unwise_W3_err[ind], 'W3', AB=True)/mag_to_flux(unwise_W3[ind], 'W3')) > 0.4:
#             continue
#         else:
#             s82x_W3[i] = unwise_W3[ind][0]
#             s82x_W3_err[i] = unwise_W3_err[ind][0]

#         if np.isnan(unwise_W4[ind]):
#             continue
#         elif unwise_W4[ind] <= 0.0:
#             continue
#         elif (magerr_to_fluxerr(unwise_W4[ind], unwise_W4_err[ind], 'W4', AB=True)/mag_to_flux(unwise_W4[ind], 'W4')) > 0.4:
#             continue
#         else:
#             s82x_W4[i] = unwise_W4[ind][0]
#             s82x_W4_err[i] = unwise_W4_err[ind][0]

# eboss_plate = eboss_data['plate']
# eboss_mjd = eboss_data['mjd']
# eboss_fiber = eboss_data['fiber']
# eboss_Lx = eboss_data['Lx']
# eboss_z = eboss_data['z']
# eboss_oiii_hb = eboss_data['logOIII_Hb']
# eboss_nii_ha = eboss_data['logNII_Ha']
# eboss_ra = eboss_data['RAJ2000']
# eboss_dec = eboss_data['DEJ2000']


# catalog = SkyCoord(ra=ananna_ra*u.degree, dec=ananna_dec*u.degree)
# c = SkyCoord(ra=eboss_ra*u.degree, dec=eboss_dec*u.degree)
# idx, d2d, d3d = c.match_to_catalog_sky(catalog)

# eboss_plate_match = eboss_plate[d2d.arcsecond < 5]
# eboss_mjd_match = eboss_mjd[d2d.arcsecond < 5]
# eboss_fiber_match = eboss_fiber[d2d.arcsecond < 5]
# eboss_Lx_match = eboss_Lx[d2d.arcsecond < 5]
# eboss_z_match = eboss_z[d2d.arcsecond < 5]
# eboss_oiii_hb_match = eboss_oiii_hb[d2d.arcsecond < 5]
# eboss_nii_ha_match = eboss_nii_ha[d2d.arcsecond < 5]
# eboss_ra_match = eboss_ra[d2d.arcsecond < 5]
# eboss_dec_match = eboss_dec[d2d.arcsecond < 5]

# # s82_niiha = np.ones(np.shape(ananna_id))
# # s82_oiiihb = np.ones(np.shape(ananna_id))
# # s82_niiha[s82_niiha == 1] = np.nan
# # s82_oiiihb[s82_oiiihb == 1] = np.nan

# s82x_niiha = []
# s82x_oiiihb = []

# for i in range(len(ananna_id)):
#     ind = np.where(ananna_id[idx[d2d.arcsecond < 5]] == ananna_id[i])[0]
#     if len(ind) > 0:
#         s82x_niiha.append(eboss_nii_ha_match[ind][0])
#         s82x_oiiihb.append(eboss_oiii_hb_match[ind][0])
#         s82x_Lx_sp_full[i] = 10**eboss_Lx_match[ind][0]
#         s82x_Lx_use[i] = 10**eboss_Lx_match[ind][0]
#         s82x_z_use[i] = eboss_z_match[ind][0]

#     else:
#         s82x_niiha.append(np.nan)
#         s82x_oiiihb.append(np.nan)

# s82x_niiha = np.asarray(s82x_niiha)
# s82x_oiiihb = np.asarray(s82x_oiiihb)

# s82x_Nh = []
# for i in range(len(ananna_id)):
#     ind = np.where(peca_ID == ananna_id[i])[0]
#     if len(ind) > 0:
#         s82x_Lx_sp_full[i] = peca_Lx_full[ind]
#         s82x_Lx_use[i] = peca_Lx_full[ind]
#         s82x_Nh.append(peca_Nh[ind][0])

#     else:
#         s82x_Nh.append(0.0)

# s82x_Nh = np.asarray(s82x_Nh)

# print('S82X All: ', len(ananna_id))


# # Select S82X sources from the Ananna2017 catalog based on conditions listed at the top of the file
# # s82x_condition = (s82x_z_sp >= z_min) & (s82x_z_sp <= z_max) & (np.log10(s82x_Lx_sp_full) >= Lx_min) & (np.log10(s82x_Lx_sp_full) <= Lx_max) & (ananna_cat == 'AO13')
# s82x_condition = (s82x_z_use > z_min) & (s82x_z_use < z_max) & (np.logical_and(ananna_ra >= 13, ananna_ra <= 37)) & (
#     np.logical_and(np.log10(s82x_Lx_use) <= Lx_max, np.log10(s82x_Lx_use) >= Lx_min))
# # s82x_condition = (s82x_z_sp > z_min) & (s82x_z_sp <= z_max) & (np.log10(s82x_Lx_sp_full) >= Lx_min) & (np.log10(s82x_Lx_sp_full) <= Lx_max) & (np.logical_and(ananna_ra >= 13,ananna_ra <= 37))
# # s82x_condition = (s82x_z_sp >= z_min) & (s82x_z_sp <= z_max) & (np.log10(s82x_Lx_sp_full) >= Lx_min) & (np.log10(s82x_Lx_sp_full) <= Lx_max)
# # s82x_condition = (np.log10(s82x_Lx_sp_full) >= Lx_min) & (np.log10(s82x_Lx_sp_full) <= Lx_max) & (np.logical_and(ananna_ra >= 13,ananna_ra <= 37))


# ananna_id = ananna_id[s82x_condition]
# ananna_cat = ananna_cat[s82x_condition]
# s82x_z_sp = s82x_z_sp[s82x_condition]
# s82x_z_use = s82x_z_use[s82x_condition]
# ananna_ra = ananna_ra[s82x_condition]
# ananna_dec = ananna_dec[s82x_condition]
# s82x_Lx_sp_full = s82x_Lx_sp_full[s82x_condition]
# s82x_Lx_use = s82x_Lx_use[s82x_condition]
# s82x_Fx_hard = s82x_Fx_hard[s82x_condition]
# s82x_Fx_soft = s82x_Fx_soft[s82x_condition]
# s82x_Fx_full = s82x_Fx_full[s82x_condition]
# s82x_Nh = s82x_Nh[s82x_condition]
# s82x_niiha = s82x_niiha[s82x_condition]
# s82x_oiiihb = s82x_oiiihb[s82x_condition]
# s82x_spec_class = s82x_spec_class[s82x_condition]
# print('S82X Lx z: ', len(ananna_id))

# # s82x_ix, s82x_iy = match(ananna_id, lamassa_id)
# # ananna_id_match = ananna_id[s82x_ix]
# # ananna_ra_match = ananna_ra[s82x_ix]
# # ananna_dec_match = ananna_dec[s82x_ix]
# # ananna_cat_match = ananna_cat[s82x_ix]
# # s82x_z_sp_match = s82x_z_sp[s82x_ix]
# # s82x_z_use_match = s82x_z_use[s82x_ix]
# # s82x_Lx_sp_full_match = s82x_Lx_sp_full[s82x_ix]
# # s82x_Fx_hard_match = s82x_Fx_hard[s82x_ix]
# # s82x_Fx_soft_match = s82x_Fx_soft[s82x_ix]
# # s82x_Fx_full_match = s82x_Fx_full[s82x_ix]
# # s82x_Nh_match = s82x_Nh[s82x_ix]
# # s82x_niiha_match = s82x_niiha[s82x_ix]
# # s82x_oiiihb_match = s82x_oiiihb[s82x_ix]
# # s82x_spec_class_match = s82x_spec_class[s82x_ix]

# # lamassa_id_match = lamassa_id[s82x_iy]
# # lamassa_z_match = lamassa_z[s82x_iy]

# s82x_phot_ix, s82x_phot_iy = match(ananna_id, ananna_phot_id)


# ananna_id_match = ananna_id[s82x_phot_ix]
# ananna_ra_match = ananna_ra[s82x_phot_ix]
# ananna_dec_match = ananna_dec[s82x_phot_ix]
# ananna_cat_match = ananna_cat[s82x_phot_ix]
# s82x_z_sp_match = s82x_z_sp[s82x_phot_ix]
# s82x_z_use_match = s82x_z_use[s82x_phot_ix]
# s82x_Lx_sp_full_match = s82x_Lx_sp_full[s82x_phot_ix]
# s82x_Lx_use_match = s82x_Lx_use[s82x_phot_ix]
# s82x_Fx_hard_match = s82x_Fx_hard[s82x_phot_ix]
# s82x_Fx_soft_match = s82x_Fx_soft[s82x_phot_ix]
# s82x_Fx_full_match = s82x_Fx_full[s82x_phot_ix]
# s82x_Nh_match = s82x_Nh[s82x_phot_ix]
# s82x_niiha_match = s82x_niiha[s82x_phot_ix]
# s82x_oiiihb_match = s82x_oiiihb[s82x_phot_ix]
# s82x_spec_class_match = s82x_spec_class[s82x_phot_ix]

# s82x_F250, s82x_F350, s82x_F500 = [], [], []
# s82x_F250_err, s82x_F350_err, s82x_F500_err = [], [], []
# for i in range(len(ananna_id_match)):
#     ind = np.where(lamassa_id_use == ananna_id_match[i])[0]
#     if len(ind) > 0:
#         s82x_F250.append(lamassa_data['F250'][ind][0])
#         s82x_F350.append(lamassa_data['F350'][ind][0])
#         s82x_F500.append(lamassa_data['F500'][ind][0])
#         s82x_F250_err.append(lamassa_data['F250_ERR'][ind][0])
#         s82x_F350_err.append(lamassa_data['F350_ERR'][ind][0])
#         s82x_F500_err.append(lamassa_data['F500_ERR'][ind][0])
#     else:
#         s82x_F250.append(-999.)
#         s82x_F350.append(-999.)
#         s82x_F500.append(-999.)
#         s82x_F250_err.append(-999.)
#         s82x_F350_err.append(-999.)
#         s82x_F500_err.append(-999.)

# s82x_F250 = np.asarray(s82x_F250)
# s82x_F350 = np.asarray(s82x_F350)
# s82x_F500 = np.asarray(s82x_F500)
# s82x_F250_err = np.asarray(s82x_F250_err)
# s82x_F350_err = np.asarray(s82x_F350_err)
# s82x_F500_err = np.asarray(s82x_F500_err)


# print('S82X match: ', len(ananna_id_match))

# s82x_Fx_hard_match_mjy = s82x_Fx_hard_match*4.136E8/(10-2)
# s82x_Fx_soft_match_mjy = s82x_Fx_soft_match*4.136E8/(2-0.5)
# s82x_Fx_full_match_mjy = s82x_Fx_full_match*4.136E8/(10-0.5)
# s82x_Fx_hard_err_match_mjy = s82x_Fx_hard_match_mjy*0.2
# s82x_Fx_soft_err_match_mjy = s82x_Fx_soft_match_mjy*0.2
# s82x_Fx_full_err_match_mjy = s82x_Fx_full_match_mjy*0.2


# # Create nan array with length == to the number of sources to be input to the photometry array
# s82x_nan_array = np.zeros(np.shape(ananna_id_match))
# s82x_nan_array[s82x_nan_array == 0] = np.nan

# # Flux array for Stripe82X sources in Jy
# s82x_flux_array = np.array([
# 	s82x_Fx_hard_match_mjy*1000, s82x_Fx_soft_match_mjy*1000,
# 	s82x_nan_array,
# 	mag_to_flux(ananna_phot_data['FUVmag'][s82x_phot_iy], 'FUV')*1E6,
# 	mag_to_flux(ananna_phot_data['NUVmag'][s82x_phot_iy], 'FUV')*1E6,
# 	mag_to_flux(ananna_phot_data['umag'][s82x_phot_iy], 'sloan_u')*1E6,
# 	mag_to_flux(ananna_phot_data['gmag'][s82x_phot_iy], 'sloan_g')*1E6,
# 	mag_to_flux(ananna_phot_data['rmag'][s82x_phot_iy], 'sloan_r')*1E6,
# 	mag_to_flux(ananna_phot_data['imag'][s82x_phot_iy], 'sloan_i')*1E6,
# 	mag_to_flux(ananna_phot_data['zmag'][s82x_phot_iy], 'sloan_z')*1E6,
# 	mag_to_flux(ananna_phot_data['Jmag'][s82x_phot_iy], 'JVHS')*1E6,
# 	mag_to_flux(ananna_phot_data['Hmag'][s82x_phot_iy], 'HVHS')*1E6,
# 	mag_to_flux(ananna_phot_data['Kmag'][s82x_phot_iy], 'HVHS')*1E6,
# 	mag_to_flux(s82x_W1[s82x_phot_iy], 'W1')*1E6,
# 	mag_to_flux(s82x_W2[s82x_phot_iy], 'W2')*1E6,
# 	mag_to_flux(s82x_W3[s82x_phot_iy], 'W3')*1E6,
# 	mag_to_flux(s82x_W4[s82x_phot_iy], 'W4')*1E6,
# 	s82x_nan_array,
# 	s82x_F250*1000,
# 	s82x_F350*1000,
# 	s82x_F500*1000
# ])


# s82x_flux_err_array = np.array([
# 	s82x_Fx_hard_err_match_mjy*1000, s82x_Fx_soft_err_match_mjy*1000,
# 	s82x_nan_array,
# 	magerr_to_fluxerr(ananna_phot_data['FUVmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_FUVmag'][s82x_phot_iy], 'FUV')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['NUVmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_NUVmag'][s82x_phot_iy], 'FUV')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['umag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_umag'][s82x_phot_iy], 'sloan_u')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['gmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_gmag'][s82x_phot_iy], 'sloan_g')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['rmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_rmag'][s82x_phot_iy], 'sloan_r')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['imag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_imag'][s82x_phot_iy], 'sloan_i')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['zmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_zmag'][s82x_phot_iy], 'sloan_z')*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['Jmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_Jmag'][s82x_phot_iy], 'JVHS', AB=True)*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['Hmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_Hmag'][s82x_phot_iy], 'HVHS', AB=True)*1E6,
# 	magerr_to_fluxerr(ananna_phot_data['Kmag'][s82x_phot_iy],
# 	                  ananna_phot_data['e_Kmag'][s82x_phot_iy], 'HVHS', AB=True)*1E6,
# 	magerr_to_fluxerr(s82x_W1[s82x_phot_iy],
# 	                  s82x_W1_err[s82x_phot_iy], 'W1', AB=True)*1E6,
# 	magerr_to_fluxerr(s82x_W2[s82x_phot_iy],
# 	                  s82x_W2_err[s82x_phot_iy], 'W2', AB=True)*1E6,
# 	magerr_to_fluxerr(s82x_W3[s82x_phot_iy],
# 	                  s82x_W3_err[s82x_phot_iy], 'W3', AB=True)*1E6,
# 	magerr_to_fluxerr(s82x_W4[s82x_phot_iy],
# 	                  s82x_W4_err[s82x_phot_iy], 'W4', AB=True)*1E6,
# 	s82x_nan_array,
# 	s82x_F250_err*1000,
# 	s82x_F350_err*1000,
# 	s82x_F500_err*1000
# ])

# # Hra = lamassa_data['HERS_RA'][s82x_iy]
# # Hdec = lamassa_data['HERS_DEC'][s82x_iy]

# s82x_flux_array = s82x_flux_array.T
# s82x_flux_err_array = s82x_flux_err_array.T
