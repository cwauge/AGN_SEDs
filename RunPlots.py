import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits 
from astropy.io import ascii
from SED_plots_v2 import Plotter
from SED_shape_plots import SED_shape_Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from scipy.stats import kstest
from match import match
from SED_v8 import Flux_to_Lum
from match import match



path = '/Users/connor_auge/Research/Disertation/catalogs/'
# AHA_SEDs_out_good_1sig
# AHA_SEDs_out_ALL
with fits.open(path+'AHA_SEDs_out_ALL_F6_FINAL5.fits') as hdul:
    cols = hdul[1].columns
    data = hdul[1].data 


temps = ascii.read('/Users/connor_auge/Research/templets/A10_templates.txt')
temp_wave = np.asarray(temps['Wave'])
temp_flux_agn = np.asarray(temps['AGN2'])*1E-14
temp_freq = 3E10/temp_wave

# temp_flux_agn = temp_flux_agn/temp_flux_agn[temp_wave == 1.0050]
temp_nuFnu = temp_flux_agn*temp_freq
temp_nuLnu = Flux_to_Lum(temp_nuFnu,np.nan,d=10,distance=True)
temp_nuFnu_norm = temp_nuFnu/temp_nuFnu[temp_wave == 1.0050]
# plt.plot(temp_wave, temp_nuFnu)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


field = data['field']
id = data['ID'] #[field == 'S82X']
z = data['z'] #[field == 'S82X']
x = data['x'] #[field == 'S82X']
y = data['y'] #[field == 'S82X']
Lx = data['Lx'] #[field == 'S82X']
norm = data['norm'] #[field == 'S82X']
FIR_upper_lims = data['FIR_upper_lims'] #[field == 'S82X']
frac_err = data['frac_err'] #[field == 'S82X']

uv_slope = data['uv_slope'] #[field == 'S82X']
mir_slope1 = data['mir_slope1'] #[field == 'S82X']
mir_slope2 = data['mir_slope2'] #[field == 'S82X']

F025 = data['F025'] #[field == 'S82X']
F1 = data['F1'] #[field == 'S82X']
F6 = data['F6'] #[field == 'S82X'] 
F10 = data['F10'] #[field == 'S82X']
F100 = data['F100'] #[field == 'S82X']
F100_ratio = data['F100_ratio']
# F025_boot = data['F025_boot'][field == 'COSMOS']
# F1_boot = data['F1_boot'][field == 'COSMOS']
# F6_boot = data['F6_boot'][field == 'COSMOS']
# F10_boot = data['F10_boot'][field == 'COSMOS']
# F100_boot = data['F100_boot'][field == 'COSMOS']
uv_lum = data['UV_lum'] #[field == 'S82X']
mir_lum = data['MIR_lum'] #[field == 'S82X']
fir_lum = data['FIR_lum'] #[field == 'S82X']

check_sed = data['sed_check']
check_sed6 = data['check6']

# F025_err = (np.std(np.log10(F025_boot),axis=1)/1000**(1/2))*3
# F6_err = (np.std(np.log10(F6_boot),axis=1)/1000**(1/2))*3

# F100_err = (np.std(np.log10(F100_boot),axis=1)/1000**(1/2))*3


shape = data['shape'] #[field == 'S82X']
Lbol = data['Lbol'] #[field == 'S82X']
Lbol_sub = data['Lbol_sub'] #[field == 'S82X']
Lbol_sf_sub = data['Lbol_sf_sub']

Nh = data['Nh'] #[field == 'S82X']
Nh_upper = data['Nh_check'] #[field == 'S82X']

int_x = data['int_x'] #[field == 'S82X']
int_y = data['int_y'] #[field == 'S82X']
wfir = data['wfir'] #[field == 'S82X']
ffir = data['ffir'] #[field == 'S82X']

spec_type = data['spec_class'] #[field == 'S82X']

F24 = data['F24_lum'] #[field == 'S82X']
stack_bin = data['stack_bin'] #[field == 'S82X']

field = field #[field == 'S82X']
s82x_z = z #[field == 'S82X']



check_sed = data['sed_check']



sort = Lx.argsort()
field = field[sort]
id = id[sort]
z = z[sort]
x = x[sort]
y = y[sort]
Lx = Lx[sort]
norm = norm[sort]
FIR_upper_lims = FIR_upper_lims[sort]
shape = shape[sort]
Lbol = Lbol[sort]
Lbol_sub = Lbol_sub[sort]
Lbol_sf_sub = Lbol_sf_sub[sort]
F025 = F025[sort]
F1 = F1[sort]
F6 = F6[sort]
F10 = F10[sort]
F24 = F24[sort]
F100 = F100[sort]
Nh = Nh[sort]
Nh_upper = Nh_upper[sort]
spec_type = spec_type[sort]
int_x = int_x[sort]
int_y = int_y[sort]
wfir = wfir[sort]
ffir = ffir[sort]
uv_lum = uv_lum[sort]
mir_lum = mir_lum[sort]
fir_lum = fir_lum[sort]
stack_bin = stack_bin[sort]
check = check_sed[sort]
check_sed6 = check_sed6[sort]
F100_ratio = F100_ratio[sort]
frac_err = frac_err[sort]
uv_slope = uv_slope[sort]
mir_slope1 = mir_slope1[sort]
mir_slope2 = mir_slope2[sort]

GOOD_SED = check_sed == 'BAD'
BAD_SED = check_sed == 'BAD'
GOOD_6 = check_sed6 == 'GOOD'
BAD_6 = check_sed6 == 'BAD'

# field_condition = (field == 'S82X') | (field == 'COSMOS') | (field == 'GOODS-N') | (field == 'GOODS-S')
field_condition = (field == 'S82X') 
# field_condition = (field == 'COSMOS') 
# field_condition = (field == 'GOODS-N') | (field == 'GOODS-S')

# plt.hist(z[GOOD_6][field[GOOD_6] == 'S82X'],bins=np.arange(0,1.3,0.1))
# plt.show()



# GOOD_SED = check_sed6 == 'GOOD'
# BAD_SED = check_sed6 == 'BAD'

# plot = Plotter(id, z, x, y, Lx, norm, FIR_upper_lims)
# plot2 = Plotter_Letter2(id, z, x, y, Lx, Lbol_sub)
# plot_shape = SED_shape_Plotter(id, z, x, y, Lx, norm, FIR_upper_lims, shape)


# plot = Plotter(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], norm[GOOD_SED], FIR_upper_lims[GOOD_SED])
# plot2 = Plotter_Letter2(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], Lbol_sub[GOOD_SED])
# plot_shape = SED_shape_Plotter(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], norm[GOOD_SED], FIR_upper_lims[GOOD_SED], shape[GOOD_SED])


plot = Plotter(id[field_condition][GOOD_6[field_condition]], z[field_condition][GOOD_6[field_condition]], x[field_condition][GOOD_6[field_condition]], y[field_condition][GOOD_6[field_condition]], Lx[field_condition][GOOD_6[field_condition]], norm[field_condition][GOOD_6[field_condition]], FIR_upper_lims[field_condition][GOOD_6[field_condition]])
plot2 = Plotter_Letter2(id[GOOD_6], z[GOOD_6], x[GOOD_6], y[GOOD_6], Lx[GOOD_6], Lbol_sub[GOOD_6])
plot_shape = SED_shape_Plotter(id[GOOD_6], z[GOOD_6], x[GOOD_6], y[GOOD_6], Lx[GOOD_6], norm[GOOD_6], FIR_upper_lims[GOOD_6], shape[GOOD_6])

print('Total SEDs')
print('Stripe82X: ', len(id[field == 'S82X']))
print('COSMOS: ', len(id[field == 'COSMOS']))
print('GOODS-N: ', len(id[field == 'GOODS-N']))
print('GOODS-S: ', len(id[field == 'GOODS-S']))
print('All: ', len(id))
print('~~~~~~~~~~')
# print('Total GOOD SEDs')
# print('Stripe82X: ', len(id[GOOD_SED][field[GOOD_SED] == 'S82X']))
# print('COSMOS: ', len(id[GOOD_SED][field[GOOD_SED] == 'COSMOS']))
# print('GOODS-N: ', len(id[GOOD_SED][field[GOOD_SED] == 'GOODS-N']))
# print('GOODS-S: ', len(id[GOOD_SED][field[GOOD_SED] == 'GOODS-S']))
# print('All: ', len(id[GOOD_SED]))
# print('~~~~~~~~~~')
print('Total GOOD 6 SEDs')
print('Stripe82X: ', len(id[GOOD_6][field[GOOD_6] == 'S82X']))
print('COSMOS: ', len(id[GOOD_6][field[GOOD_6] == 'COSMOS']))
print('GOODS-N: ', len(id[GOOD_6][field[GOOD_6] == 'GOODS-N']))
print('GOODS-S: ', len(id[GOOD_6][field[GOOD_6] == 'GOODS-S']))
print('All: ', len(id[GOOD_6]))
print('~~~~~~~~~~')
# print('Total BAD SEDs')
# print('Stripe82X: ', len(id[BAD_SED][field[BAD_SED] == 'S82X']))
# print('COSMOS: ', len(id[BAD_SED][field[BAD_SED] == 'COSMOS']))
# print('GOODS-N: ', len(id[BAD_SED][field[BAD_SED] == 'GOODS-N']))
# print('GOODS-S: ', len(id[BAD_SED][field[BAD_SED] == 'GOODS-S']))
# print('All: ', len(id[BAD_SED]))


# def sed_shape(uv_slope, mir_slope1, mir_slope2):
    # Pre-defined conditions. Check slope values to determine SED shape bin and return bin
    # if (uv_slope < -0.3) & (mir_slope1 >= -0.2):
    #     bin = 1
    # elif (uv_slope >= -0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2):
    #     bin = 2
    # elif (uv_slope > 0.2) & (mir_slope1 >= -0.2):
    #     bin = 3
    # elif (uv_slope >=-0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0):
    #     bin = 4
    # elif (uv_slope >=-0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0):
    #     bin = 5
    # else:
    #     bin = 6
    # return bin

def sed_shape(uv_slope, mir_slope1, mir_slope2):
#     # Pre-defined conditions. Check slope values to determine SED shape bin and return bin
    if (uv_slope < -0.3) & (mir_slope1 >= -0.35):
        bin = 1
    elif (uv_slope >= -0.3) & (uv_slope <= 0.21) & (mir_slope1 >= -0.35):
        bin = 2
    elif (uv_slope > 0.21) & (mir_slope1 >= -0.35):
        bin = 3
    elif (uv_slope > 0.21) & (mir_slope1 < -0.35) & (mir_slope2 > 0.0):
        bin = 4
    elif (uv_slope > 0.21) & (mir_slope1 < -0.35) & (mir_slope2 <= 0.0):
        bin = 5
    else:
        bin = 6
    return bin

# ix, iy = match(np.asarray(id[GOOD_6][test_shape2]), np.asarray(id[GOOD_6][test_shape3]))
# print(len(id[GOOD_6][ix]),len(id[GOOD_6][iy]))

# print(id[GOOD_6][test_shape2])
# print(id[GOOD_6][test_shape3])

sed_shape_check = []
for i in range(len(uv_slope)):
    s = sed_shape(uv_slope[i], mir_slope1[i], mir_slope2[i])
    sed_shape_check.append(s)
sed_shape_check = np.asarray(sed_shape_check)[GOOD_6]
print(np.shape(sed_shape_check))
print('shape 1: ',len(sed_shape_check[sed_shape_check == 1]), len(sed_shape_check[sed_shape_check == 1])/len(sed_shape_check)*100)
print('shape 2: ',len(sed_shape_check[sed_shape_check == 2]), len(sed_shape_check[sed_shape_check == 2])/len(sed_shape_check)*100)
print('shape 3: ',len(sed_shape_check[sed_shape_check == 3]), len(sed_shape_check[sed_shape_check == 3])/len(sed_shape_check)*100)
print('shape 4: ',len(sed_shape_check[sed_shape_check == 4]), len(sed_shape_check[sed_shape_check == 4])/len(sed_shape_check)*100)
print('shape 5: ',len(sed_shape_check[sed_shape_check == 5]), len(sed_shape_check[sed_shape_check == 5])/len(sed_shape_check)*100)
print('shape 6: ',len(sed_shape_check[sed_shape_check == 6]), len(sed_shape_check[sed_shape_check == 6])/len(sed_shape_check)*100)
print('Total: ', len(sed_shape_check[sed_shape_check == 1])+len(sed_shape_check[sed_shape_check == 2])+len(
    sed_shape_check[sed_shape_check == 3])+len(sed_shape_check[sed_shape_check == 4])+len(sed_shape_check[sed_shape_check == 5]))
print('Total: ', len(sed_shape_check[sed_shape_check == 1])+len(sed_shape_check[sed_shape_check == 2])+len(
    sed_shape_check[sed_shape_check == 3])+len(sed_shape_check[sed_shape_check == 4])+len(sed_shape_check[sed_shape_check == 5])+len(sed_shape_check[sed_shape_check == 6]))




# print('~~~~~~~~~~')
# print('~~~~~~~~~~')
# print('~~~~~~~~~~')

# print('Total shape break down')
# print('shape 1: ',len(id[shape == 1]),len(id[shape == 1])/len(id)*100)
# print('shape 2: ',len(id[shape == 2]),len(id[shape == 2])/len(id)*100)
# print('shape 3: ',len(id[shape == 3]),len(id[shape == 3])/len(id)*100)
# print('shape 4: ',len(id[shape == 4]),len(id[shape == 4])/len(id)*100)
# print('shape 5: ',len(id[shape == 5]),len(id[shape == 5])/len(id)*100)
# print('shape 4 + 5: ',len(id[shape == 4])+len(id[shape == 5]),(len(id[shape == 4])+len(id[shape == 5]))/len(id)*100)
# print('~~~~~~~~~~')

# print('GOOD shape break down')
# print('shape 1: ',len(id[GOOD_SED][shape[GOOD_SED] == 1]),len(id[GOOD_SED][shape[GOOD_SED] == 1])/len(id[GOOD_SED])*100)
# print('shape 2: ',len(id[GOOD_SED][shape[GOOD_SED] == 2]),len(id[GOOD_SED][shape[GOOD_SED] == 2])/len(id[GOOD_SED])*100)
# print('shape 3: ',len(id[GOOD_SED][shape[GOOD_SED] == 3]),len(id[GOOD_SED][shape[GOOD_SED] == 3])/len(id[GOOD_SED])*100)
# print('shape 4: ',len(id[GOOD_SED][shape[GOOD_SED] == 4]),len(id[GOOD_SED][shape[GOOD_SED] == 4])/len(id[GOOD_SED])*100)
# print('shape 5: ',len(id[GOOD_SED][shape[GOOD_SED] == 5]),len(id[GOOD_SED][shape[GOOD_SED] == 5])/len(id[GOOD_SED])*100)
# print('shape 4 + 5: ',len(id[GOOD_SED][shape[GOOD_SED] == 4])+len(id[GOOD_SED][shape[GOOD_SED] == 5]),(len(id[GOOD_SED][shape[GOOD_SED] == 4])+len(id[GOOD_SED][shape[GOOD_SED] == 5]))/len(id[GOOD_SED])*100)
# print('~~~~~~~~~~')

print('GOOD 6 shape break down')
print('shape 1: ',len(id[GOOD_6][shape[GOOD_6] == 1]),len(id[GOOD_6][shape[GOOD_6] == 1])/len(id[GOOD_6])*100)
print('shape 2: ',len(id[GOOD_6][shape[GOOD_6] == 2]),len(id[GOOD_6][shape[GOOD_6] == 2])/len(id[GOOD_6])*100)
print('shape 3: ',len(id[GOOD_6][shape[GOOD_6] == 3]),len(id[GOOD_6][shape[GOOD_6] == 3])/len(id[GOOD_6])*100)
print('shape 4: ',len(id[GOOD_6][shape[GOOD_6] == 4]),len(id[GOOD_6][shape[GOOD_6] == 4])/len(id[GOOD_6])*100)
print('shape 5: ',len(id[GOOD_6][shape[GOOD_6] == 5]),len(id[GOOD_6][shape[GOOD_6] == 5])/len(id[GOOD_6])*100)
print('Total: ', len(id[GOOD_6][shape[GOOD_6] == 1])+len(id[GOOD_6]
      [shape[GOOD_6] == 2])+len(id[GOOD_6][shape[GOOD_6] == 3])+len(id[GOOD_6][shape[GOOD_6] == 4])+len(id[GOOD_6][shape[GOOD_6] == 5]))
print('shape 6: ',len(id[GOOD_6][shape[GOOD_6] == -99.]),len(id[GOOD_6][shape[GOOD_6] == -99.])/len(id[GOOD_6])*100)
# print('shape 4 + 5: ',len(id[GOOD_6][shape[GOOD_6] == 4])+len(id[GOOD_6][shape[GOOD_6] == 5]),(len(id[GOOD_6][shape[GOOD_6] == 4])+len(id[GOOD_6][shape[GOOD_6] == 5]))/len(id[GOOD_6])*100)
print('~~~~~~~~~~')


# print('BAD shape break down')
# print('shape 1: ',len(id[BAD_SED][shape[BAD_SED] == 1]),len(id[BAD_SED][shape[BAD_SED] == 1])/len(id[BAD_SED])*100)
# print('shape 2: ',len(id[BAD_SED][shape[BAD_SED] == 2]),len(id[BAD_SED][shape[BAD_SED] == 2])/len(id[BAD_SED])*100)
# print('shape 3: ',len(id[BAD_SED][shape[BAD_SED] == 3]),len(id[BAD_SED][shape[BAD_SED] == 3])/len(id[BAD_SED])*100)
# print('shape 4: ',len(id[BAD_SED][shape[BAD_SED] == 4]),len(id[BAD_SED][shape[BAD_SED] == 4])/len(id[BAD_SED])*100)
# print('shape 5: ',len(id[BAD_SED][shape[BAD_SED] == 5]),len(id[BAD_SED][shape[BAD_SED] == 5])/len(id[BAD_SED])*100)
# print('shape 4 + 5: ',len(id[BAD_SED][shape[BAD_SED] == 4])+len(id[BAD_SED][shape[BAD_SED] == 5]),(len(id[BAD_SED][shape[BAD_SED] == 4])+len(id[BAD_SED][shape[BAD_SED] == 5]))/len(id[BAD_SED])*100)
# print('~~~~~~~~~~')






print('FIR Detections')
print('Total number of sources:                 ', len(FIR_upper_lims[GOOD_6]))
print('Number of sources with detections:       ',len(FIR_upper_lims[GOOD_6][FIR_upper_lims[GOOD_6] == 0]))
print('Number of sources with upper limits:     ',len(FIR_upper_lims[GOOD_6][FIR_upper_lims[GOOD_6] == 1]))
print('Percentage of sources with detection:    ',len(FIR_upper_lims[GOOD_6][FIR_upper_lims[GOOD_6] == 0])/len(FIR_upper_lims[GOOD_6])*100)
print('Percentage of sources with no detection: ',len(FIR_upper_lims[GOOD_6][FIR_upper_lims[GOOD_6] == 1])/len(FIR_upper_lims[GOOD_6])*100)
print('~~~~~~~~~~')


print('NH Detections')
print('Total number of sources:                 ', len(Nh_upper[GOOD_6]))
print('Number of sources with detections:       ',len(Nh_upper[GOOD_6][Nh_upper[GOOD_6] == 0]))
print('Number of sources with upper limits:     ',len(Nh_upper[GOOD_6][Nh_upper[GOOD_6] == 1]))
print('Number of sources with lower limits:     ',len(Nh_upper[GOOD_6][Nh_upper[GOOD_6] == 2]))
print('Percentage of sources with detection:    ',len(Nh_upper[GOOD_6][Nh_upper[GOOD_6] == 0])/len(Nh_upper[GOOD_6])*100)
print('Percentage of sources with no detection: ',len(Nh_upper[GOOD_6][Nh_upper[GOOD_6] == 1])/len(Nh_upper[GOOD_6])*100)
print('~~~~~~~~~~')

id_check = id == 25734
print('Source Check:')
print(field[id_check],id[id_check],10**Lx[id_check],Nh[id_check],Lbol_sub[id_check],shape[id_check])
print('~~~~~~~~~~')


### 
# Test Medians


def stern(mir):
    a = 40.981
    b = 1.024
    c = 0.047
    # x = np.log10(mir/1E41)
    x = mir - 41

    Lx = a+b*x-c*x**2
    Lx += np.log10(1.64)

    return Lx


# plt.plot(np.log10(F6[GOOD_6]), Lx[GOOD_6], '.')
# plt.plot(np.arange(40, 48), stern(np.arange(40, 48)), color='k')
# plt.xlim(40, 47)
# plt.ylim(40, 47)
# plt.grid()
# plt.show()

# plt.plot(Lx[GOOD_6], np.log10(F6[GOOD_6]), '.')
# plt.plot(stern(np.arange(40, 48)), np.arange(40, 48), color='k')
# plt.xlim(40, 47)
# plt.ylim(40, 47)
# plt.grid()
# plt.show()


# plt.figure(figsize=(10,10),facecolor='w')
# ax1 = plt.subplot(111,)

# Fig 1 (opt)
# # plt.plot(xue_z_match,np.log10(xue_Lx_match),'.',color='gray',rasterized=True)
# # plt.plot(luo_z_match,np.log10(luo_Lx_match),'.',color='gray',rasterized=True,label='GOODS-N/S')
# plt.plot(z[GOOD_6][field[GOOD_6] == 'COSMOS'],np.log10(Lx[GOOD_6][field[GOOD_6] == 'COSMOS']),'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# # plt.plot(s82x_z_sp_match,np.log10(s82x_≥Lx_sp_full_match),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.plot(z[GOOD_6][field[GOOD_6] == 'S82X'],np.log10(Lx[GOOD_6][field[GOOD_6] == 'S82X']),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.plot(z[GOOD_6][field[GOOD_6] == 'GOODS-N'],np.log10(Lx[GOOD_6][field[GOOD_6] == 'GOODS-N']),'.',ms=12,color='gray',rasterized=True)
# plt.plot(z[GOOD_6][field[GOOD_6] == 'GOODS-S'],np.log10(Lx[GOOD_6][field[GOOD_6] == 'GOODS-S']),'.',ms=12,color='gray',rasterized=True,label='GOODS-N/S')
# # plt.plot(ulirg_z,ulirg_Lx,'*',color='g',ms=12,rasterized=True,label='GOALS')
# plt.xlabel('Spectroscopic Redshift')
# plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# # plt.text(2.15, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)+len(ulirg_z)}')
# # plt.text(3.25, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(chandra_cosmos_z_match)+len(s82x_z)}')
# # plt.text(3.25, 40.55, f'n = {len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)}')
# plt.legend()
# # plt.axvline(1.2,color='k',ls='--',lw=3)
# plt.xlim(-0.05,1.25)
# plt.grid()
# plt.tight_layout()
# # plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_new/Lx_z_spec3.pdf')
# plt.show()

# Fig 2
# plt.figure(figsize=(10,10),facecolor='w')
# plt.hist(Lx[GOOD_6],color='gray',alpha=0.45)
# plt.hist(Lx[GOOD_6][field[GOOD_6] == 'S82X'],bins=np.arange(43,46,0.25),histtype='step',color='r',lw=5,label='Stripe 82X')
# plt.hist(Lx[GOOD_6][field[GOOD_6] == 'COSMOS'],bins=np.arange(43,46,0.25),histtype='step',color='b',lw=5,label='COSMOS')
# plt.hist(Lx[GOOD_6][np.logical_or(field[GOOD_6]=='GOODS-S',field[GOOD_6]=='GOODS-N')],bins=np.arange(43,46,0.25),histtype='step',color='k',lw=5,label='GOODS-N/S')
# plt.axvline(np.nanmedian(Lx[GOOD_6][field[GOOD_6] == 'S82X']),color='r',ls='--',lw=3)
# plt.axvline(np.nanmedian(Lx[GOOD_6][field[GOOD_6] == 'COSMOS']),color='b',ls='--',lw=3)
# plt.axvline(np.nanmedian(Lx[GOOD_6][np.logical_or(field[GOOD_6]=='GOODS-S',field[GOOD_6]=='GOODS-N')]),color='k',ls='--',lw=3)
# plt.legend()
# plt.grid()
# plt.xlabel(r'log $L_{\rm X}$ [erg s$^{-1}$]')
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_six/a_final/Lx_sample.pdf')
# plt.show()

# Fig 3
# plot.multi_SED('a_check/All_SEDs_temp',int_x,int_y,wfir,ffir,wave_labels=True,temp_comp=True,temp_comp_x=temp_wave,temp_comp_y=temp_nuFnu_norm)
# plot.multi_SED('a_check/All_SEDs_sample_bad', int_x[GOOD_SED], int_y[GOOD_SED], wfir[GOOD_SED], ffir[GOOD_SED], wave_labels=True)
plot.multi_SED('vision_slides/All_SEDs_S82X2', int_x[field_condition][GOOD_6[field_condition]], int_y[field_condition][GOOD_6[field_condition]], wfir[field_condition][GOOD_6[field_condition]], ffir[field_condition][GOOD_6[field_condition]], wave_labels=True)

# Fig 4
# plot.L_hist('a_six/a_final/Lone_hist',np.log10(F1[GOOD_6]),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=True)

# Fig 5
# plot.multi_SED_bins('a_six/a_final/All_z_bins_norm','redshift',field[GOOD_6],median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],scale=True)

# Fig 6 
# plot.L_hist_zbins('a_six/a_final/Lone_hist_zbins',np.log10(F1[GOOD_6]),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot_shape.L_hist_bins('a_six/Lone_hist_Lx_bins',np.log10(F1[GOOD_6]),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False,bins='Lx')

# Fig 7 
# plot.multi_SED_bins('a_six/a_final/All_Lx_bins_norm2',bin='Lx',field=field[GOOD_6],median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],Median_line=True,FIR_upper='upper lims',scale=True)

# Fig 8 
# plot.L_scatter_3panels('a_new/AGN_emission_s82x_3sig','UV-MIR-FIR','Lx','X-axis',norm,F025,F6,F100,shape,Lx,uv_err=F025_err,mir_err=F6_err,fir_err=F100_err,error=False)
# plot.L_scatter_3panels('a_new/AGN_emission_1sig','UV-MIR-FIR','Lx','X-axis',norm,F025,F6,F100,shape,Lx,error=False,stack_color=True,stack_bins=stack_bin,F100_ratio=F100_ratio,field = field,fir_field=True)
# plot.L_scatter_3panels('a_new/AGN_emission_medians', 'Lx', 'UV-MIR-FIR', 'X-axis', norm, F025, F6, F100, shape,Lx, error=False, stack_color=True, stack_bins=stack_bin, F100_ratio=F100_ratio, field=field, fir_field=True)

# plot.L_scatter_3panels('a_six/AGN_emission_sample', 'Lx', 'UV-MIR-FIR', 'X-axis', norm[GOOD_6], F025[GOOD_6], F6[GOOD_6], F100[GOOD_6], shape[GOOD_6],Lx[GOOD_6], error=False, stack_color=True, stack_bins=stack_bin[GOOD_6], F100_ratio=F100_ratio[GOOD_6], field=field[GOOD_6], fir_field=True)
# plot.L_scatter_3panels_vert('a_six/a_final/AGN_emission_ALL', 'Lx', 'UV-MIR-FIR', 'X-axis', norm[GOOD_6], F025[GOOD_6], F6[GOOD_6], F100[GOOD_6], shape[GOOD_6],Lx[GOOD_6], error=False, stack_color=True, stack_bins=stack_bin[GOOD_6], F100_ratio=F100_ratio[GOOD_6], field=field[GOOD_6], fir_field=True)


# Fig 9
# plot.L_hist('a_six/a_final/Nh_hist',np.log10(Nh[GOOD_6]),r'log $N_{\rm H}/(\rm{cm}^{-2})$', [20,24.5], [20,24.5,0.25],split=True,split_param=Nh_upper[GOOD_6])

# Fig 10
# plot.L_hist('a_six/a_final/Lbol_hist_new',np.log10(Lbol[GOOD_6]),r'Total log $L_{\rm bol}/({\rm erg \; s^{-1}})$',[43,47],[43,47,0.25],std=True,top_label=True,xlabel2=r'Total log $L_{\rm bol}/\rm{L_\odot}$')

# Fig 11 (Lbol/Lx scatter)
# plt.figure(figsize=(6,6))
# plt.plot(np.log10(Lbol[GOOD_6]), np.log10(Lbol_sub[GOOD_6]), '.')
# plt.plot(np.arange(42,48),np.arange(42,48),color='k')
# plt.xlabel('Lbol')
# plt.ylabel('Lbol sub')
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.plot(np.log10(Lbol[GOOD_6]), np.log10(Lbol_sf_sub[GOOD_6]), '.')
# plt.plot(np.arange(42, 48), np.arange(42, 48), color='k')
# plt.xlabel('Lbol')
# plt.ylabel('Lbol sub')
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.plot(np.log10(Lbol_sub[GOOD_6]), np.log10(Lbol_sf_sub[GOOD_6]), '.')
# plt.plot(np.arange(42, 48), np.arange(42, 48), color='k')
# plt.xlabel('Lbol')
# plt.ylabel('Lbol sub')
# plt.show()

lbol_med2x = np.array([44.25, 44.75, 45.25, 45.75, 46.25])
lbol_med2y = np.array([0.81489127, 1.0489582, 1.228792, 1.37900561, 1.98974862])
lbol_med2xerr = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
lbol_med2yerr = np.array([0.34876941, 0.35842915, 0.40963542, 0.39048596, 1.47821504])

# print(np.log10(Lbol) - np.log10(Lbol_sub))
# for i in range(len(Lbol)):
#     print(np.log10(Lbol[i]) - np.log10(Lbol_sub[i]),
#           np.log10(Lbol[i]) - np.log10(Lbol_sf_sub[i]))

# plt.hist(np.log10(Lbol) - np.log10(Lbol_sub),bins=np.arange(-1,1,0.1),alpha=0.75,label='E sub')
# plt.hist(np.log10(Lbol) - np.log10(Lbol_sf_sub),bins=np.arange(-1,1,0.1),alpha=0.75,label='SF sub')
# plt.xlabel('Lbol - Lbol_sub')
# plt.legend()
# plt.show()

# plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_full_new','Lbol','Lbol/Lx','X-axis',F1[GOOD_6],F025[GOOD_6],F6[GOOD_6],F100[GOOD_6],shape[GOOD_6],np.log10(Lbol[GOOD_6]))
# plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_sub_new','Lbol','Lbol/Lx','X-axis',F1[GOOD_6],F025[GOOD_6],F6[GOOD_6],F100[GOOD_6],shape[GOOD_6],np.log10(Lbol_sub[GOOD_6]))
# plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_sf_cut','Lbol','Lbol/Lx','X-axis',F1[GOOD_6],F025[GOOD_6],F6[GOOD_6],F100[GOOD_6],shape[GOOD_6],np.log10(Lbol_sf_sub[GOOD_6]))
# plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_both_NEW','Lbol','Lbol/Lx','X-axis',F1[GOOD_6],F025[GOOD_6],F6[GOOD_6],F100[GOOD_6],shape[GOOD_6],np.log10(Lbol_sub[GOOD_6]),med2x=lbol_med2x,med2y=lbol_med2y,med2xerr=lbol_med2xerr,med2yerr=lbol_med2yerr)



# plot.L_ratio_1panel('a_six/a_final/Lx_Lbol_cut','Lbol','Lbol/Lx','X-axis',F1[GOOD_6],F025[GOOD_6],F6[GOOD_6],F100[GOOD_6],shape[GOOD_6],np.log10(Lbol[GOOD_6]))

# plot.L_ratio_1panel('a_new/Lx_Lbol_spec_type','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape,np.log10(Lbol_sub),sample=True,spec_type=spec_type)

# Fig 12
# plot_shape.shape_1bin_v('a_check/vertical_5_panel_check_all6',median_x=int_x,median_y=int_y,wfir=wfir,ffir=ffir,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims',bins='shape')
# plot_shape.shape_1bin_v('a_six/a_final/vertical_5_panel_new',median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],uv_slope=uv_slope[GOOD_6],mir_slope1=mir_slope1[GOOD_6],mir_slope2=mir_slope2[GOOD_6],Median_line=True,FIR_upper='upper lims',bins='shape')
# plot_shape.shape_1bin_h('a_six/horizontal_5_panel_check2',median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],uv_slope=uv_slope[GOOD_6],mir_slope1=mir_slope1[GOOD_6],mir_slope2=mir_slope2[GOOD_6],Median_line=True,FIR_upper='upper lims',bins='shape')

# Fig 13
# plot_shape.L_hist_panels('a_six/a_final/Lone_hist_panels_update',np.log10(F1[GOOD_6]),r'log L (1 $\mu$m)/(erg s$^{-1}$)',[43.25,46],[43.25,46,0.25],z_label=True,top_label=True,xlabel2=r'log L (1 $\mu$m)/$\rm{L_\odot}$')

# Fig 14 (Med SED)
# plot.median_SED_1panel('a_six/a_final/median_SED_new_norm',int_x[GOOD_6],int_y[GOOD_6]-np.log10(F1[GOOD_6][:,None]),wfir[GOOD_6],ffir[GOOD_6]/F1[GOOD_6][:,None],shape[GOOD_6],plot_temp=False,temp_x=temp_wave,temp_y=temp_nuLnu*1E6)

# Fig 15
# plot_shape.L_hist_panels('a_six/a_final/Lx_hist_panels_all',Lx[GOOD_6],r'log $L_{\rm X}$/(erg s$^{-1}$)',[43,46],[43,46,0.25],z_label=True)
# plot_shape.L_hist_panels('a_six/a_final/LbolLx_hist_panels_all',np.log10(Lbol_sub[GOOD_6])-Lx[GOOD_6],r'log $L_{\rm bol}$ / $L_{\rm X}$',[-0.5,3],[0,2.75,0.25],z_label=True)

# Fig 16
# plot2.scatter_1panel('a_six/a_final/UV_MIR','MIR6','UV/MIR6','None','Bins',Nh[GOOD_6],Lx[GOOD_6],np.log10(Lbol_sub[GOOD_6]),F1[GOOD_6],np.log10(F025[GOOD_6]),np.log10(F6[GOOD_6]),np.log10(F100[GOOD_6]),np.log10(F10[GOOD_6]),uv_slope[GOOD_6],mir_slope1[GOOD_6],mir_slope2[GOOD_6],FIR_upper_lims[GOOD_6],shape[GOOD_6])

# Fig 17 
# plot.L_ratio_multi_panel('a_six/a_final/ratio_multipanel','Lx','AGN','bins',np.log10(F1[GOOD_6]),np.log10(F025[GOOD_6]),np.log10(F6[GOOD_6]),np.log10(F100[GOOD_6]),np.log10(Nh[GOOD_6]),np.log10(Lbol_sub[GOOD_6]),shape[GOOD_6],FIR_upper_lims[GOOD_6],F100_ratio=F100_ratio[GOOD_6],field=field[GOOD_6])

# Fig 18
# plot_shape.L_hist_panels('a_six/a_final/Nh_hist_panel_update',np.log10(Nh[GOOD_6]),r'log $N_{\rm H}/(\rm{cm}^{-2})$', [20,24.5],[20,24.5,0.25],split=True,split_param=Nh_upper[GOOD_6])

# Fig 19
# plot_shape.L_hist_panels2('a_six/a_final/Lbol_Lx_hist_panels_update_new', np.log10(Lbol_sub[GOOD_6]), np.log10(Lbol_sub[GOOD_6])-Lx[GOOD_6], r'log $L_{\rm bol-gal,e}$',[44,47],[44,47,0.25],[0,3],[0,3,0.25])

# Panel Scatter
# plot2.scatter_1panel('a_new/FIR_Lx_update_check2','Lx','FIR/Lx',None,'Bins',Nh,Lx,np.log10(Lbol_sub),np.log10(F1),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape)

# U-panel plots
# plot2.Upanels_ratio_plots('a_new/Nh_Upanels_update','Nh','UV/MIR-UV/Lx-MIR/Lx','Bins',Nh,Lx,Lbol_sub,np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=shape,Nh_upper=Nh_upper)
# plot.Upanels_ratio('a_six/a_final/Lum_Lbol_update_new','Lbol','UV-MIR-FIR','Bins',np.log10(Lbol_sub[GOOD_6]),np.log10(uv_lum[GOOD_6]),np.log10(mir_lum[GOOD_6]),np.log10(fir_lum[GOOD_6]),np.log10(F1[GOOD_6]),shape[GOOD_6],FIR_upper_lims[GOOD_6],field=field[GOOD_6])
