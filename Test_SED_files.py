import numpy as np
import matplotlib.pyplot as plt
from match import match
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord 
from astropy import units as u
from SED_v8 import AGN
from SED_plots_v2 import Plotter


def only_one(x,f):
    # out = [i for i in x if np.where(x == i)[0] == 1]
    out = list(set(x).symmetric_difference(set(f)))
    return np.array(out)

# def only_one(x):
#     out = []
#     for i in range(len(x)):
#         ind = np.where(x == x[i])[0]
#         if ind ==1:
#             out.append(x[i])
#         else:
#             continue
#     return np.array(out)
 
path = '/Users/connor_auge/Research/Disertation/catalogs/output/'
path2 = '/Users/connor_auge/Research/Disertation/catalogs/'
path3 = '/Users/connor_auge/'

# with fits.open(path+'AGN_properties_final.fits') as hdul:
#     prop_data = hdul[1].data
#     prop_cols = hdul[1].columns

# with fits.open(path+'AGN_photometry_cosmos_final.fits') as hdul:
#     cosmos_data = hdul[1].data
#     cosmos_cols = hdul[1].columns

# with fits.open(path+'AGN_photometry_s82x_final.fits') as hdul:
#     s82x_data = hdul[1].data
#     s82x_cols = hdul[1].columns

# with fits.open(path+'AGN_photometry_GOODS_final.fits') as hdul:
#     goods_data = hdul[1].data
#     goods_cols = hdul[1].columns

prop = ascii.read(path+'/AGN_properties_final.csv')
s82x = ascii.read(path3+'/AGN_photometry_s82x_final.csv')
cosmos = ascii.read(path3+'/AGN_photometry_cosmos_final.csv')
goods = ascii.read(path3+'/AGN_photometry_goods_final.csv')

print(len(prop))
print(len(s82x),len(cosmos),len(goods),len(s82x)+len(cosmos)+len(goods))


prop_field = np.array(prop['Field'])
prop_z = np.array(prop['z_spec'])
prop_Lx = np.array(prop['L0510_c'])
prop_Lbol = np.array(prop['Lbol'])
goods_field = np.array(goods['Field'])


cosmos_Lx = prop_Lx[prop_field == 'COSMOS']
cosmos_z = prop_z[prop_field == 'COSMOS']
cosmos_id = np.array(cosmos['phot_id'])
nan_array = np.ones(np.shape(np.array(cosmos['Fxh'])))
nan_array[nan_array == 1] = np.nan
cosmos_flux_array = np.array([
    np.array(cosmos['Fxh']),
    np.array(cosmos['Fxs']),
    nan_array,
    np.array(cosmos['FUV']),
    np.array(cosmos['NUV']),
    np.array(cosmos['u']),
    np.array(cosmos['g']),
    np.array(cosmos['r']),
    np.array(cosmos['i']),
    np.array(cosmos['z']),
    np.array(cosmos['y']),
    np.array(cosmos['J']),
    np.array(cosmos['H']),
    np.array(cosmos['Ks']),
    np.array(cosmos['IRAC1']),
    np.array(cosmos['IRAC2']),
    np.array(cosmos['IRAC3']),
    np.array(cosmos['IRAC4']),
    np.array(cosmos['F24']),
    np.array(cosmos['F100']),
    np.array(cosmos['F160']),
    np.array(cosmos['F250']),
    np.array(cosmos['F350']),
    np.array(cosmos['F500']),
])

cosmos_flux_err_array = np.array([
    np.array(cosmos['Fxh'])*0.2,
    np.array(cosmos['Fxs'])*0.2,
    nan_array,
    np.array(cosmos['FUV_err']),
    np.array(cosmos['NUV_err']),
    np.array(cosmos['u_err']),
    np.array(cosmos['g_err']),
    np.array(cosmos['r_err']),
    np.array(cosmos['i_err']),
    np.array(cosmos['z_err']),
    np.array(cosmos['y_err']),
    np.array(cosmos['J_err']),
    np.array(cosmos['H_err']),
    np.array(cosmos['Ks_err']),
    np.array(cosmos['IRAC1_err']),
    np.array(cosmos['IRAC2_err']),
    np.array(cosmos['IRAC3_err']),
    np.array(cosmos['IRAC4_err']),
    np.array(cosmos['F24_err']),
    np.array(cosmos['F100_err']),
    np.array(cosmos['F160_err']),
    np.array(cosmos['F250_err']),
    np.array(cosmos['F350_err']),
    np.array(cosmos['F500_err']),
])

cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T







s82x_Lx = prop_Lx[prop_field == 'Stripe82X']
s82x_z = prop_z[prop_field == 'Stripe82X']
s82x_id = np.array(s82x['phot_id'])
nan_array = np.ones(np.shape(np.array(s82x['Fxh'])))
nan_array[nan_array == 1] = np.nan
s82x_flux_array = np.array([
    np.array(s82x['Fxh']),
    np.array(s82x['Fxs']),
    nan_array,
    np.array(s82x['FUV']),
    np.array(s82x['NUV']),
    np.array(s82x['u']),
    np.array(s82x['g']),
    np.array(s82x['r']),
    np.array(s82x['i']),
    np.array(s82x['z']),
    np.array(s82x['J']),
    np.array(s82x['H']),
    np.array(s82x['Ks']),
    np.array(s82x['W1']),
    np.array(s82x['W2']),
    np.array(s82x['W3']),
    np.array(s82x['W4']),
    nan_array,
    np.array(s82x['F250']),
    np.array(s82x['F350']),
    np.array(s82x['F500']),
])

s82x_flux_err_array = np.array([
    np.array(s82x['Fxh'])*0.2,
    np.array(s82x['Fxs'])*0.2,
    nan_array,
    np.array(s82x['FUV_err']),
    np.array(s82x['NUV_err']),
    np.array(s82x['u_err']),
    np.array(s82x['g_err']),
    np.array(s82x['r_err']),
    np.array(s82x['i_err']),
    np.array(s82x['z_err']),
    np.array(s82x['J_err']),
    np.array(s82x['H_err']),
    np.array(s82x['Ks_err']),
    np.array(s82x['W1_err']),
    np.array(s82x['W2_err']),
    np.array(s82x['W3_err']),
    np.array(s82x['W4_err']),
    nan_array,
    np.array(s82x['F250_err']),
    np.array(s82x['F350_err']),
    np.array(s82x['F500_err']),
])

s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T


goodsN_Lx = prop_Lx[prop_field == 'GOODS-N']
goodsN_z = prop_z[prop_field == 'GOODS-N']
goodsN_id = np.array(goods['phot_id'])[goods_field == 'GOODS-N']
nan_array = np.ones(np.shape(np.array(goods['Fxh'])[goods_field == 'GOODS-N']))
nan_array[nan_array == 1] = np.nan
goodsN_flux_array = np.array([
    np.array(goods['Fxh'])[goods_field == 'GOODS-N'],
    np.array(goods['Fxs'])[goods_field == 'GOODS-N'],
    nan_array,
    np.array(goods['FUV'])[goods_field == 'GOODS-N'],
    np.array(goods['NUV'])[goods_field == 'GOODS-N'],
    np.array(goods['U'])[goods_field == 'GOODS-N'],
    np.array(goods['F435W'])[goods_field == 'GOODS-N'],
    np.array(goods['B'])[goods_field == 'GOODS-N'],
    np.array(goods['V'])[goods_field == 'GOODS-N'],
    np.array(goods['F606W'])[goods_field == 'GOODS-N'],
    np.array(goods['R'])[goods_field == 'GOODS-N'],
    np.array(goods['I'])[goods_field == 'GOODS-N'],
    np.array(goods['F775W'])[goods_field == 'GOODS-N'],
    np.array(goods['F814W'])[goods_field == 'GOODS-N'],
    np.array(goods['z'])[goods_field == 'GOODS-N'],
    np.array(goods['F105W'])[goods_field == 'GOODS-N'],
    np.array(goods['F125W'])[goods_field == 'GOODS-N'],
    np.array(goods['J'])[goods_field == 'GOODS-N'],
    np.array(goods['F140W'])[goods_field == 'GOODS-N'],
    np.array(goods['F160W'])[goods_field == 'GOODS-N'],
    np.array(goods['H'])[goods_field == 'GOODS-N'],
    np.array(goods['Ks'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC1'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC2'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC3'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC4'])[goods_field == 'GOODS-N'],
    np.array(goods['F24'])[goods_field == 'GOODS-N'],
    np.array(goods['F70'])[goods_field == 'GOODS-N'],
    np.array(goods['F100'])[goods_field == 'GOODS-N'],
    np.array(goods['F160'])[goods_field == 'GOODS-N'],
    np.array(goods['F250'])[goods_field == 'GOODS-N'],
    np.array(goods['F350'])[goods_field == 'GOODS-N'],
    np.array(goods['F500'])[goods_field == 'GOODS-N'],
])

goodsN_flux_err_array = np.array([
    np.array(goods['Fxh'])[goods_field == 'GOODS-N']*0.2,
    np.array(goods['Fxs'])[goods_field == 'GOODS-N']*0.2,
    nan_array,
    np.array(goods['FUV_err'])[goods_field == 'GOODS-N'],
    np.array(goods['NUV_err'])[goods_field == 'GOODS-N'],
    np.array(goods['U_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F435W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['B_err'])[goods_field == 'GOODS-N'],
    np.array(goods['V_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F606W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['R_err'])[goods_field == 'GOODS-N'],
    np.array(goods['I_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F775W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F814W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['z_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F105W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F125W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['J_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F140W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F160W_err'])[goods_field == 'GOODS-N'],
    np.array(goods['H_err'])[goods_field == 'GOODS-N'],
    np.array(goods['Ks_err'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC1_err'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC2_err'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC3_err'])[goods_field == 'GOODS-N'],
    np.array(goods['IRAC4_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F24_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F70_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F100_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F160_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F250_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F350_err'])[goods_field == 'GOODS-N'],
    np.array(goods['F500_err'])[goods_field == 'GOODS-N'],
])

goodsN_flux_array = goodsN_flux_array.T
goodsN_flux_err_array = goodsN_flux_err_array.T



goodsS_Lx = prop_Lx[prop_field == 'GOODS-S']
goodsS_z = prop_z[prop_field == 'GOODS-S']
goodsS_id = np.array(goods['phot_id'])[goods_field == 'GOODS-S']
nan_array = np.ones(np.shape(np.array(goods['Fxh'])[goods_field == 'GOODS-S']))
nan_array[nan_array == 1] = np.nan
goodsS_flux_array = np.array([
    np.array(goods['Fxh'])[goods_field == 'GOODS-S'],
    np.array(goods['Fxs'])[goods_field == 'GOODS-S'],
    nan_array,
    np.array(goods['FUV'])[goods_field == 'GOODS-S'],
    np.array(goods['NUV'])[goods_field == 'GOODS-S'],
    np.array(goods['U'])[goods_field == 'GOODS-S'],
    np.array(goods['F435W'])[goods_field == 'GOODS-S'],
    np.array(goods['B'])[goods_field == 'GOODS-S'],
    np.array(goods['V'])[goods_field == 'GOODS-S'],
    np.array(goods['F606W'])[goods_field == 'GOODS-S'],
    np.array(goods['R'])[goods_field == 'GOODS-S'],
    np.array(goods['I'])[goods_field == 'GOODS-S'],
    np.array(goods['F775W'])[goods_field == 'GOODS-S'],
    np.array(goods['F814W'])[goods_field == 'GOODS-S'],
    np.array(goods['z'])[goods_field == 'GOODS-S'],
    np.array(goods['F850LP'])[goods_field == 'GOODS-S'],
    np.array(goods['F098M'])[goods_field == 'GOODS-S'],
    np.array(goods['F105W'])[goods_field == 'GOODS-S'],
    np.array(goods['F125W'])[goods_field == 'GOODS-S'],
    np.array(goods['J'])[goods_field == 'GOODS-S'],
    np.array(goods['F140W'])[goods_field == 'GOODS-S'],
    np.array(goods['F160W'])[goods_field == 'GOODS-S'],
    np.array(goods['H'])[goods_field == 'GOODS-S'],
    np.array(goods['Ks'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC1'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC2'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC3'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC4'])[goods_field == 'GOODS-S'],
    np.array(goods['F24'])[goods_field == 'GOODS-S'],
    np.array(goods['F70'])[goods_field == 'GOODS-S'],
    np.array(goods['F100'])[goods_field == 'GOODS-S'],
    np.array(goods['F160'])[goods_field == 'GOODS-S'],
    np.array(goods['F250'])[goods_field == 'GOODS-S'],
    np.array(goods['F350'])[goods_field == 'GOODS-S'],
    np.array(goods['F500'])[goods_field == 'GOODS-S'],
])

goodsS_flux_err_array = np.array([
    np.array(goods['Fxh'])[goods_field == 'GOODS-S']*0.2,
    np.array(goods['Fxs'])[goods_field == 'GOODS-S']*0.2,
    nan_array,
    np.array(goods['FUV_err'])[goods_field == 'GOODS-S'],
    np.array(goods['NUV_err'])[goods_field == 'GOODS-S'],
    np.array(goods['U_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F435W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['B_err'])[goods_field == 'GOODS-S'],
    np.array(goods['V_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F606W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['R_err'])[goods_field == 'GOODS-S'],
    np.array(goods['I_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F775W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F814W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['z_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F850LP_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F098M_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F105W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F125W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['J_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F140W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F160W_err'])[goods_field == 'GOODS-S'],
    np.array(goods['H_err'])[goods_field == 'GOODS-S'],
    np.array(goods['Ks_err'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC1_err'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC2_err'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC3_err'])[goods_field == 'GOODS-S'],
    np.array(goods['IRAC4_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F24_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F70_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F100_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F160_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F250_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F350_err'])[goods_field == 'GOODS-S'],
    np.array(goods['F500_err'])[goods_field == 'GOODS-S'],
])

goodsS_flux_array = goodsS_flux_array.T
goodsS_flux_err_array = goodsS_flux_err_array.T







COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'JVHS', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
S82X_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
                          'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'nan', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])
GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I',
                                  'F775W', 'F814W', 'Z', 'F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100_goodsS', 'FLUX_160_goodsS', 'FLUX_250_goodsS', 'FLUX_350_goodsS', 'FLUX_500_goodsS'])

GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                  'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100_goodsN', 'FLUX_160_goodsN', 'FLUX_250_goodsN', 'FLUX_350_goodsN', 'FLUX_500_goodsN'])


out_id, out_x, out_y, out_z, out_frac_err, out_upcheck = [],[],[],[],[],[]
out_shape, out_f1 = [], []
out_intx, out_inty, out_wfir, out_ffir = [], [], [], []
out_Lx = []
out_F025, out_F6, out_F100 = [], [], []
out_field = []
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(COSMOS_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(cosmos_id)):
    source = AGN(cosmos_id[i],cosmos_z[i],COSMOS_filters,cosmos_flux_array[i],cosmos_flux_err_array[i])
    source.MakeSED(data_replace_filt=['FLUX_24'])
    source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160',
                      'FLUX_250', 'FLUX_350', 'FLUX_500'])
    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    f1 = source.Find_value(1.0)
    f025 = source.Find_value(0.25)
    f6 = source.Find_value(6.0)
    lbol = source.Find_Lbol()
    shape = source.SED_shape()
    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    w = np.append(w, fill_nan)
    f = np.append(f, fill_nan)
    frac_err = np.append(frac_err, fill_nan)
    out_id.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_z.append(redshift)
    out_frac_err.append(frac_err)
    out_upcheck.append(up_check)
    out_shape.append(shape)
    out_f1.append(f1)
    out_intx.append(ix)
    out_inty.append(iy)
    out_wfir.append(wfir)
    out_ffir.append(ffir)
    out_Lx.append(cosmos_Lx[i])
    out_F025.append(f025)
    out_F6.append(f6)
    out_F100.append(f100)
    out_field.append('COSMOS')


fill_nan = np.zeros(len(GOODSS_auge_filters)-len(S82X_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(s82x_id)):
    source = AGN(s82x_id[i],s82x_z[i],S82X_filters,s82x_flux_array[i],s82x_flux_err_array[i])
    source.MakeSED(data_replace_filt=['W4'])
    source.FIR_extrap(
        ['W4', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'], stack=True)
    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    f1 = source.Find_value(1.0)
    f025 = source.Find_value(0.25)
    f6 = source.Find_value(6.0)
    lbol = source.Find_Lbol()
    shape = source.SED_shape()
    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    w = np.append(w, fill_nan)
    f = np.append(f, fill_nan)
    frac_err = np.append(frac_err, fill_nan)
    out_id.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_z.append(redshift)
    out_frac_err.append(frac_err)
    out_upcheck.append(up_check)
    out_shape.append(shape)
    out_f1.append(f1)
    out_intx.append(ix)
    out_inty.append(iy)
    out_wfir.append(wfir)
    out_ffir.append(ffir)
    out_Lx.append(s82x_Lx[i])
    out_F025.append(f025)
    out_F6.append(f6)
    out_F100.append(f100)
    out_field.append('S82X')


fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_auge_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(goodsN_id)):
    # if goodsN_id[i] == 56002:
    #     print(goodsN_flux_array[i])
    #     print(goodsN_flux_err_array[i])
    source = AGN(goodsN_id[i],goodsN_z[i],GOODSN_auge_filters,goodsN_flux_array[i],goodsN_flux_err_array[i])
    source.MakeSED(data_replace_filt=['FLUX_24'])
    source.FIR_extrap(['FLUX_24', 'MIPS2', 'FLUX_100_goodsN', 'FLUX_160_goodsN',
                      'FLUX_250_goodsN', 'FLUX_350_goodsN', 'FLUX_500_goodsN'])
    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    f1 = source.Find_value(1.0)
    f025 = source.Find_value(0.25)
    f6 = source.Find_value(6.0)
    lbol = source.Find_Lbol()
    shape = source.SED_shape()
    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    w = np.append(w, fill_nan)
    f = np.append(f, fill_nan)
    frac_err = np.append(frac_err, fill_nan)
    out_id.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_z.append(redshift)
    out_frac_err.append(frac_err)
    out_upcheck.append(up_check)
    out_shape.append(shape)
    out_f1.append(f1)
    out_intx.append(ix)
    out_inty.append(iy)
    out_wfir.append(wfir)
    out_ffir.append(ffir)
    out_Lx.append(goodsN_Lx[i])
    out_F025.append(f025)
    out_F6.append(f6)
    out_F100.append(f100)
    out_field.append('GOODS-N')

for i in range(len(goodsS_id)):
    source = AGN(goodsS_id[i],goodsS_z[i],GOODSS_auge_filters,goodsS_flux_array[i],goodsS_flux_err_array[i])
    source.MakeSED(data_replace_filt=['FLUX_24'])
    source.FIR_extrap(['FLUX_24', 'MIPS2', 'FLUX_100_goodsS', 'FLUX_160_goodsS', 
                       'FLUX_250_goodsS', 'FLUX_350_goodsS', 'FLUX_500_goodsS'])
    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    f1 = source.Find_value(1.0)
    f025 = source.Find_value(0.25)
    f6 = source.Find_value(6.0)
    lbol = source.Find_Lbol()
    shape = source.SED_shape()
    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    frac_err = np.append(frac_err, fill_nan)
    out_id.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_z.append(redshift)
    out_frac_err.append(frac_err)
    out_upcheck.append(up_check)
    out_shape.append(shape)
    out_f1.append(f1)
    out_intx.append(ix)
    out_inty.append(iy)
    out_wfir.append(wfir)
    out_ffir.append(ffir)
    out_Lx.append(goodsS_Lx[i])
    out_F025.append(f025)
    out_F6.append(f6)
    out_F100.append(f100)
    out_field.append('GOODS-S')

out_id, out_x, out_y, out_z, out_frac_err, out_upcheck, out_shape = np.asarray(out_id), np.asarray(out_x), np.asarray(out_y), np.asarray(out_z), np.asarray(out_frac_err), np.asarray(out_upcheck), np.asarray(out_shape)
out_intx, out_inty, out_wfir, out_ffir = np.asarray(out_intx), np.asarray(out_inty), np.asarray(out_wfir), np.asarray(out_ffir)
out_F025, out_F6, out_F100, out_field = np.asarray(out_F025), np.asarray(out_F6), np.asarray(out_F100), np.asarray(out_field)

# print(np.shape(out_id),np.shape(out_x),np.shape(out_y),np.shape(cosmos_Lx))
plot = Plotter(out_id,out_z,out_x,out_y,out_Lx,out_f1,out_upcheck)
plot.multi_SED('All_multi',out_intx,out_inty,out_wfir,out_ffir,wave_labels=True,)
plot.L_scatter_3panels_vert('All_AGN_emission_test','Lx','UV-MIR-FIR','X-axis',out_f1,out_F025,out_F6,out_F100,out_shape,out_Lx,error=False,stack_color=False,field=out_field)

plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_full_CHECK','Lbol','Lbol/Lx','X-axis',out_f1,out_F025,out_F6,out_F100,out_shape,prop_Lbol)


# with fits.open(path2+'Ananna2017_S82X.fit') as hdul:
#     ananna_data = hdul[1].data
#     ananna_cols = hdul[1].columns

# with fits.open(path2+'chandra_COSMOS_legacy_opt_NIR_counterparts_20160113_4d.fits') as hdul:
#     chandra_data = hdul[1].data 
#     chandra_cols = hdul[1].columns

# with fits.open(path2+'COSMOS2020_CLASSIC_R1_v2.2_p3.fits') as hdul:
#     cosmos_cat_data = hdul[1].data
#     cosmos_cat_cols = hdul[1].columns

# print(cosmos_cat_cols)
# print(cosmos_cat_data['HSC_g_FLUX_APER2'])
# print(cosmos_cat_data['HSC_g_FLUXERR_APER2'])
# print(ananna_cols)
# print(chandra_cols)

# with fits.open(path2+'Peca_S82X_full.fit') as hdul:
#     peca_data = hdul[1].data
#     peca_cols = hdul[1].columns

# with fits.open(path2+'LaMassa2019_S82x_eboss2.fit') as hdul:
#     lamassa_data = hdul[1].data
#     lamassa_cols = hdul[1].columns

# print(lamassa_data)
# print(lamassa_cols)
# print(lamassa_cols)
# print(peca_cols)

# lamassa_z = lamassa_data['z']
# lamassa_id = lamassa_data['L16']
# lamassa_ra = lamassa_data['RAJ2000']
# lamassa_dec = lamassa_data['DEJ2000']
# lamassa_Lx = lamassa_data['Lx0_5-10']

# print(lamassa_id)

# peca_z = peca_data['z']
# peca_fz = peca_data['f_z']
# peca_id = peca_data['Source']

# ananna_z = ananna_data['zsp']
# ananna_id = ananna_data['ID']

# prop_field = prop_data['field']
# s82x_z = prop_data['z_spec'][prop_field == 'Stripe82X']
# s82x_id = np.asarray(prop_data['x_ID'][prop_field == 'Stripe82X'],dtype='int')
# s82x_ra = prop_data['RAJ2000'][prop_field == 'Stripe82X']
# s82x_dec = prop_data['DEJ2000'][prop_field == 'Stripe82X']
# s82x_Lx = prop_data['L0510_c'][prop_field == 'Stripe82X']

# catalog = SkyCoord(ra = lamassa_ra*u.degree, dec = lamassa_dec*u.degree)
# c = SkyCoord(ra=s82x_ra, dec=s82x_dec, unit=(u.hourangle, u.deg))
# idx, d2d, d3d = c.match_to_catalog_sky(catalog)
# idx_match = idx[d2d.arcsecond < 3]

# lamassa_z_match = lamassa_z[idx_match]
# lamassa_id_match = s82x_id[d2d.arcsecond < 3]

# print(len(lamassa_id_match), len(lamassa_z_match))




# ix, iy = match(ananna_id,s82x_id)
# ix2, iy2 = match(peca_id,s82x_id)

# s82x_z_match = s82x_z[iy]
# ananna_z_match = ananna_z[ix]
# s82x_z_match2 = s82x_z[iy2]
# peca_z_match = peca_z[ix2]
# s82x_id_match = s82x_id[iy]
# s82x_id_match2 = s82x_id[iy2]
# ananna_id_match = ananna_id[ix]
# peca_id_match = peca_id[ix2]
# peca_fz_match = peca_fz[ix2]

# print('here')
# print(len(s82x_id))
# print(len(s82x_z_match))
# print(len(ananna_z_match))
# print(len(peca_z_match))

# s82x_unique_id = only_one(s82x_id,ananna_id_match)
# s82x_unique_id = np.append(s82x_unique_id,ananna_id_match[ananna_z_match <= 0])
# print(len(s82x_unique_id))

# ix3, iy3 = match(s82x_unique_id,peca_id)

# peca_id_match2 = peca_id[iy3]
# peca_z_match2 = peca_z[iy3]
# peca_fz_match2 = peca_fz[iy3]
# print(len(peca_id_match2))
# for i in range(len(peca_z_match2)):
#     print(peca_id_match2[i],peca_z_match2[i],peca_fz_match2[i])
# print(len(ananna_z_match[ananna_z_match <= 0]))
# print('HERE')
# for i in range(len(s82x_z)):
#     print(s82x_id[i],s82x_z[i])

# print('here')
# print(s82x_id_match[ananna_z_match <= 0])

# for i in range(len(peca_id_match)):
#     print(peca_id_match[i],peca_z_match[i],peca_fz_match[i])


# for i in range(len(prop_data)):
#     print(i,prop_data[i])

# for i in range(len(lamassa_id_match)):
#     print(lamassa_id_match[i],lamassa_z_match[i])


# prop_field = prop_data['field']
# print(len(prop_field),len(prop_field[prop_field == 'Stripe82X']),len(prop_field[prop_field == 'COSMOS']),len(prop_field[prop_field == 'GOODS-N'])+len(prop_field[prop_field == 'GOODS-S']))
# print(len(s82x_data),len(cosmos_data),len(goods_data))