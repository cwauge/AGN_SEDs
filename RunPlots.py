import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits 
from astropy.io import ascii
from SED_plots_v2 import Plotter
from SED_shape_plots import SED_shape_Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from match import match
from SED_v8 import Flux_to_Lum



path = '/Users/connor_auge/Research/Disertation/catalogs/'
# AHA_SEDs_out_good_1sig
# AHA_SEDs_out_ALL
with fits.open(path+'AHA_SEDs_out_ALL_F6.fits') as hdul:
    cols = hdul[1].columns
    data = hdul[1].data 


temps = ascii.read('/Users/connor_auge/Desktop/templets/A10_templates.txt')
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


# plt.plot(z[FIR_upper_lims == 0],np.log10(F100[FIR_upper_lims == 0]),'.',label='Detection')
# plt.plot(z[FIR_upper_lims == 1],np.log10(F100[FIR_upper_lims == 1]),'.',color='r',label='Upper Limits')
# plt.xlabel('redshift')
# plt.ylabel('FIR Luminosity Upper limits')
# plt.title('S82X')
# plt.show()


# plt.figure(figsize=(10,10),facecolor='white')
# plt.plot(s82x_z,np.log10(F24[field == 'S82X']),'o')
# # plt.plot(s82x_z[s82x_250 > 0],np.log10(L24_1[s82x_250 > 0])-off,'o',color='r')
# plt.axvline(0.45,ls='--',color='k')
# plt.axvline(0.70,ls='--',color='k')
# plt.axhline(44,ls='--',color='b')
# plt.axhline(45,ls='--',color='b')
# plt.xlabel('Redshift')
# plt.ylabel(r'log L$_{24}$ (erg/s)')
# plt.xlim(0,1.22)
# plt.ylim(43,46.2)
# plt.show()

check_sed = data['sed_check']


# sort = Lx.argsort()

# id = id[sort]
# z = z[sort]
# x = x[sort]
# y = y[sort]
# Lx = Lx[sort]
# norm = norm[sort]
# FIR_upper_lims = FIR_upper_lims[sort]
# Lbol_sub = Lbol_sub[sort]
# shape = shape[sort]
# check_sed = check_sed[sort]
# F025 = F025[sort]
# F6 = F6[sort]
# F100 = F100[sort]
# int_x = int_x[sort]
# int_y = int_y[sort]
# wfir = wfir[sort]
# ffir = ffir[sort]
# field = field[sort]
# F100_ratio = F100_ratio[sort]


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

# GOOD_SED = check_sed6 == 'GOOD'
# BAD_SED = check_sed6 == 'BAD'

# plot = Plotter(id, z, x, y, Lx, norm, FIR_upper_lims)
# plot2 = Plotter_Letter2(id, z, x, y, Lx, Lbol_sub)
# plot_shape = SED_shape_Plotter(id, z, x, y, Lx, norm, FIR_upper_lims, shape)


# plot = Plotter(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], norm[GOOD_SED], FIR_upper_lims[GOOD_SED])
# plot2 = Plotter_Letter2(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], Lbol_sub[GOOD_SED])
# plot_shape = SED_shape_Plotter(id[GOOD_SED], z[GOOD_SED], x[GOOD_SED], y[GOOD_SED], Lx[GOOD_SED], norm[GOOD_SED], FIR_upper_lims[GOOD_SED], shape[GOOD_SED])


plot = Plotter(id[GOOD_6], z[GOOD_6], x[GOOD_6], y[GOOD_6], Lx[GOOD_6], norm[GOOD_6], FIR_upper_lims[GOOD_6])
plot2 = Plotter_Letter2(id[GOOD_6], z[GOOD_6], x[GOOD_6], y[GOOD_6], Lx[GOOD_6], Lbol_sub[GOOD_6])
plot_shape = SED_shape_Plotter(id[GOOD_6], z[GOOD_6], x[GOOD_6], y[GOOD_6], Lx[GOOD_6], norm[GOOD_6], FIR_upper_lims[GOOD_6], shape[GOOD_6])

print('Total SEDs')
print('Stripe82X: ', len(id[field == 'S82X']))
print('COSMOS: ', len(id[field == 'COSMOS']))
print('GOODS-N: ', len(id[field == 'GOODS-N']))
print('GOODS-S: ', len(id[field == 'GOODS-S']))
print('All: ', len(id))
print('~~~~~~~~~~')
print('Total GOOD SEDs')
print('Stripe82X: ', len(id[GOOD_SED][field[GOOD_SED] == 'S82X']))
print('COSMOS: ', len(id[GOOD_SED][field[GOOD_SED] == 'COSMOS']))
print('GOODS-N: ', len(id[GOOD_SED][field[GOOD_SED] == 'GOODS-N']))
print('GOODS-S: ', len(id[GOOD_SED][field[GOOD_SED] == 'GOODS-S']))
print('All: ', len(id[GOOD_SED]))
print('~~~~~~~~~~')
print('Total GOOD 6 SEDs')
print('Stripe82X: ', len(id[GOOD_6][field[GOOD_6] == 'S82X']))
print('COSMOS: ', len(id[GOOD_6][field[GOOD_6] == 'COSMOS']))
print('GOODS-N: ', len(id[GOOD_6][field[GOOD_6] == 'GOODS-N']))
print('GOODS-S: ', len(id[GOOD_6][field[GOOD_6] == 'GOODS-S']))
print('All: ', len(id[GOOD_6]))
print('~~~~~~~~~~')
print('Total BAD SEDs')
print('Stripe82X: ', len(id[BAD_SED][field[BAD_SED] == 'S82X']))
print('COSMOS: ', len(id[BAD_SED][field[BAD_SED] == 'COSMOS']))
print('GOODS-N: ', len(id[BAD_SED][field[BAD_SED] == 'GOODS-N']))
print('GOODS-S: ', len(id[BAD_SED][field[BAD_SED] == 'GOODS-S']))
print('All: ', len(id[BAD_SED]))

print('~~~~~~~~~~')
print('~~~~~~~~~~')
print('~~~~~~~~~~')

print('Total shape break down')
print('shape 1: ',len(id[shape == 1]),len(id[shape == 1])/len(id)*100)
print('shape 2: ',len(id[shape == 2]),len(id[shape == 2])/len(id)*100)
print('shape 3: ',len(id[shape == 3]),len(id[shape == 3])/len(id)*100)
print('shape 4: ',len(id[shape == 4]),len(id[shape == 4])/len(id)*100)
print('shape 5: ',len(id[shape == 5]),len(id[shape == 5])/len(id)*100)
print('shape 4 + 5: ',len(id[shape == 4])+len(id[shape == 5]),(len(id[shape == 4])+len(id[shape == 5]))/len(id)*100)
print('~~~~~~~~~~')

print('GOOD shape break down')
print('shape 1: ',len(id[GOOD_SED][shape[GOOD_SED] == 1]),len(id[GOOD_SED][shape[GOOD_SED] == 1])/len(id[GOOD_SED])*100)
print('shape 2: ',len(id[GOOD_SED][shape[GOOD_SED] == 2]),len(id[GOOD_SED][shape[GOOD_SED] == 2])/len(id[GOOD_SED])*100)
print('shape 3: ',len(id[GOOD_SED][shape[GOOD_SED] == 3]),len(id[GOOD_SED][shape[GOOD_SED] == 3])/len(id[GOOD_SED])*100)
print('shape 4: ',len(id[GOOD_SED][shape[GOOD_SED] == 4]),len(id[GOOD_SED][shape[GOOD_SED] == 4])/len(id[GOOD_SED])*100)
print('shape 5: ',len(id[GOOD_SED][shape[GOOD_SED] == 5]),len(id[GOOD_SED][shape[GOOD_SED] == 5])/len(id[GOOD_SED])*100)
print('shape 4 + 5: ',len(id[GOOD_SED][shape[GOOD_SED] == 4])+len(id[GOOD_SED][shape[GOOD_SED] == 5]),(len(id[GOOD_SED][shape[GOOD_SED] == 4])+len(id[GOOD_SED][shape[GOOD_SED] == 5]))/len(id[GOOD_SED])*100)
print('~~~~~~~~~~')

print('GOOD 6 shape break down')
print('shape 1: ',len(id[GOOD_6][shape[GOOD_6] == 1]),len(id[GOOD_6][shape[GOOD_6] == 1])/len(id[GOOD_6])*100)
print('shape 2: ',len(id[GOOD_6][shape[GOOD_6] == 2]),len(id[GOOD_6][shape[GOOD_6] == 2])/len(id[GOOD_6])*100)
print('shape 3: ',len(id[GOOD_6][shape[GOOD_6] == 3]),len(id[GOOD_6][shape[GOOD_6] == 3])/len(id[GOOD_6])*100)
print('shape 4: ',len(id[GOOD_6][shape[GOOD_6] == 4]),len(id[GOOD_6][shape[GOOD_6] == 4])/len(id[GOOD_6])*100)
print('shape 5: ',len(id[GOOD_6][shape[GOOD_6] == 5]),len(id[GOOD_6][shape[GOOD_6] == 5])/len(id[GOOD_6])*100)
print('shape 4 + 5: ',len(id[GOOD_6][shape[GOOD_6] == 4])+len(id[GOOD_6][shape[GOOD_6] == 5]),(len(id[GOOD_6][shape[GOOD_6] == 4])+len(id[GOOD_6][shape[GOOD_6] == 5]))/len(id[GOOD_6])*100)
print('~~~~~~~~~~')


print('BAD shape break down')
print('shape 1: ',len(id[BAD_SED][shape[BAD_SED] == 1]),len(id[BAD_SED][shape[BAD_SED] == 1])/len(id[BAD_SED])*100)
print('shape 2: ',len(id[BAD_SED][shape[BAD_SED] == 2]),len(id[BAD_SED][shape[BAD_SED] == 2])/len(id[BAD_SED])*100)
print('shape 3: ',len(id[BAD_SED][shape[BAD_SED] == 3]),len(id[BAD_SED][shape[BAD_SED] == 3])/len(id[BAD_SED])*100)
print('shape 4: ',len(id[BAD_SED][shape[BAD_SED] == 4]),len(id[BAD_SED][shape[BAD_SED] == 4])/len(id[BAD_SED])*100)
print('shape 5: ',len(id[BAD_SED][shape[BAD_SED] == 5]),len(id[BAD_SED][shape[BAD_SED] == 5])/len(id[BAD_SED])*100)
print('shape 4 + 5: ',len(id[BAD_SED][shape[BAD_SED] == 4])+len(id[BAD_SED][shape[BAD_SED] == 5]),(len(id[BAD_SED][shape[BAD_SED] == 4])+len(id[BAD_SED][shape[BAD_SED] == 5]))/len(id[BAD_SED])*100)
print('~~~~~~~~~~')


# plt.figure(figsize=(8,8))
# plt.hist(shape[spec_type==1],bins=np.arange(0,7,1),color='orange',lw=3,histtype='step',alpha=0.75,label='Type 1')
# plt.hist(shape[spec_type==2],bins=np.arange(0,7,1),color='green',lw=3,histtype='step',alpha=0.75,label='Type 2')
# plt.xlabel('SED Shape')
# plt.legend(loc='upper left')
# plt.show()

# plt.figure(figsize=(10,10))
# plt.hist(Lx[check_sed == 'GOOD'],bins=np.arange(42,47,0.25),color='gray',alpha=0.5)
# plt.hist(Lx[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 0],bins=np.arange(42,47,0.25),color='red',histtype='step',lw=3,label='FIR detections')
# plt.hist(Lx[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 1],bins=np.arange(42,47,0.25),color='blue',histtype='step',lw=3,label='FIR non-detections')
# plt.axvline(np.nanmedian(Lx[FIR_upper_lims == 0]),ls='--',color='r',lw=3)
# plt.axvline(np.nanmedian(Lx[FIR_upper_lims == 1]), ls='--', color='b',lw=3)
# plt.xlabel(r'log L$_{\rm X}$ [erg/s]')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,10))
# plt.hist(Lx,bins=np.arange(42,47,0.25),color='gray',alpha=0.5)
# plt.hist(Lx[check_sed6 == 'GOOD'],bins=np.arange(42,47,0.25),color='red',histtype='step',lw=3,label='Sample')
# plt.hist(Lx[check_sed6 == 'BAD'],bins=np.arange(42,47,0.25),color='blue',histtype='step',lw=3,label='Removed sources')
# plt.axvline(np.nanmedian(Lx[check_sed6 == 'GOOD']), ls='--', color='r',lw=3)
# plt.axvline(np.nanmedian(Lx[check_sed6 == 'BAD']), ls='--', color='b',lw=3)
# plt.xlabel(r'log L$_{\rm X}$ [erg/s]')
# plt.legend()
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_check/Lx_hist_sample6_3.pdf')
# plt.show()

# plt.figure(figsize=(10,10))
# # plt.plot(z, Lx, 'o', color='gray',alpha=0.5,label='Total Sample')
# plt.plot(z[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 0], Lx[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 0], '.', color='red', label='FIR detections')
# plt.plot(z[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 1], Lx[check_sed == 'GOOD'][FIR_upper_lims[check_sed == 'GOOD'] == 1], '.', color='blue', label='FIR non-detections')
# plt.xlabel('Redshift')
# plt.ylabel(r'log L$_{\rm X}$ [erg/s]')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,10))
# # plt.plot(z, Lx, 'o', color='gray',alpha=0.5,label='Total Sample')
# plt.plot(z[check_sed6 == 'GOOD'], Lx[check_sed6 == 'GOOD'], '.', color='red', label='Sample')
# plt.plot(z[check_sed6 == 'BAD'], Lx[check_sed6 == 'BAD'], '.', color='blue', label='Removed Sources')
# plt.xlabel('Redshift')
# plt.ylabel(r'log L$_{\rm X}$ [erg/s]')
# plt.grid()
# plt.legend()
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_check/Lx_z_sample6.pdf')
# plt.show()

# print(np.shape(F100_boot))
# plt.hist(np.log10(F100_boot[15]),bins=np.arange(42,47,0.25))
# plt.axvline(np.log10(F100[15]),color='k')
# plt.axvline(np.nanmean(np.log10(F100_boot[15])),ls='--',color='red')
# plt.show()

# ind = (shape == 1) & (Lx > 44.5) & (FIR_upper_lims == 0)
# ID = 777034
# ind = (id == ID)
# plot_single = Plotter(id[ind][0],z[ind][0],x[ind][0],y[ind][0]*norm[ind][0],10**Lx[ind][0],norm[ind][0],FIR_upper_lims[ind][0])
# plot_single.PlotSED()

# plt.hist(np.log10(norm),alpha=0.5,label='F1')
# plt.hist(np.log10(F100),alpha=0.5,label='F100')
# plt.hist(np.log10(F100_ratio),alpha=0.5,label='F100 ratio')
# plt.legend()
# plt.show()

# plt.plot(np.log10(norm),np.log10(F100),'.')
# plt.plot(np.log10(norm),np.log10(F100_ratio),'.')
# plt.plot(np.arange(40,50),np.arange(40,50),color='k')
# plt.xlabel('F1')
# plt.ylabel('F100')
# plt.show()

# plt.plot(np.log10(Lbol_sub),np.log10(Lbol),'.')
# plt.plot(np.arange(44,47),np.arange(44,47),color='k')
# plt.xlabel('Lbol gal sub')
# plt.ylabel('Lbol total')
# plt.grid()
# plt.show()






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


def Just_alpha_ox(Luv):
    a = -0.140
    b = 2.705
    c = -0.093
    d = 0.899
    w = 2500*1E-8  # observed wavelength from Angstroms to cm
    nu_uv = 3E10/w  # convert obs wavelength to a frequ

    wx = 6.2*1E-8
    nu_x = 3E10/wx

    # Luv =- np.log10(nu)
    Lnu = Luv - np.log10(nu_uv)

    # Lx = (1/c)*(a*Lnu+b-d)
    Lx = 0.709*Lnu+4.822
    Lx_out = Lx

    Lx_out = Lx + np.log10(nu_x)
    Lx_out += 0.67

    return Lx_out

# F6 = F6[GOOD_SED]
# F025 = F025[GOOD_SED]
# Lx = Lx[GOOD_SED]

# F6 = F6[BAD_SED]
# F025 = F025[BAD_SED]
# Lx = Lx[BAD_SED]
# b1 =(Lx > 45) & (Lx < 45.5)
b1 =(Lx > 45)
b2 = (Lx > 44.5) & (Lx < 45)
b3 =(Lx > 44) & (Lx < 44.5) 
b4 = (Lx > 43.5) & (Lx < 44)
b5 = (Lx > 43) & (Lx < 43.5)

med1 = np.nanmedian(np.log10(F6[b1]))
med2 = np.nanmedian(np.log10(F6[b2]))
med3 = np.nanmedian(np.log10(F6[b3]))
med4 = np.nanmedian(np.log10(F6[b4]))
med5 = np.nanmedian(np.log10(F6[b5]))


per1_25 = np.nanpercentile(np.log10(F6[b1]),20)
per2_25 = np.nanpercentile(np.log10(F6[b2]),20)
per3_25 = np.nanpercentile(np.log10(F6[b3]),20)
per4_25 = np.nanpercentile(np.log10(F6[b4]),20)
per5_25 = np.nanpercentile(np.log10(F6[b5]),20)


per1_75 = np.nanpercentile(np.log10(F6[b1]), 80)
per2_75 = np.nanpercentile(np.log10(F6[b2]), 80)
per3_75 = np.nanpercentile(np.log10(F6[b3]), 80)
per4_75 = np.nanpercentile(np.log10(F6[b4]), 80)
per5_75 = np.nanpercentile(np.log10(F6[b5]), 80)


med1_2 = np.nanmedian(np.log10(F6[b1][(np.log10(F6[b1]) > per1_25)&(np.log10(F6[b1]) < per1_75)]))
med2_2 = np.nanmedian(np.log10(F6[b2][(np.log10(F6[b2]) > per2_25)&(np.log10(F6[b2]) < per2_75)])) 
med3_2 = np.nanmedian(np.log10(F6[b3][(np.log10(F6[b3]) > per3_25)&(np.log10(F6[b3]) < per3_75)]))
med4_2 = np.nanmedian(np.log10(F6[b4][(np.log10(F6[b4]) > per4_25)&(np.log10(F6[b4]) < per4_75)]))
med5_2 = np.nanmedian(np.log10(F6[b5][(np.log10(F6[b5]) > per5_25)&(np.log10(F6[b5]) < per5_75)]))

print(np.log10(F6[b1]))
print(np.log10(F6[b1][(np.log10(F6[b1]) > per1_25)&(np.log10(F6[b1]) < per1_75)]))
print(med1,med1_2)
print(len(F6[b1]))
print(len(F6[b1][(np.log10(F6[b1]) > per1_25) & (np.log10(F6[b1]) < per1_75)]))
# med1_2 = np.nanmedian(np.log10(F6[(Lx > 44.75) & (Lx < 45.75)]))
# med2_2 = np.nanmedian(np.log10(F6[(Lx > 44.25) & (Lx < 45.25)]))
# med3_2 = np.nanmedian(np.log10(F6[(Lx > 43.75) & (Lx < 44.75)]))
# med4_2 = np.nanmedian(np.log10(F6[(Lx > 43.25) & (Lx < 44.25)]))
# med5_2 = np.nanmedian(np.log10(F6[(Lx > 42.27) & (Lx < 43.75)]))

# med1_3 = np.nanmedian(np.log10(F6[(Lx > 44.75) & (Lx < 45.25)]))
# med2_3 = np.nanmedian(np.log10(F6[(Lx > 44.25) & (Lx < 44.75)]))
# med3_3 = np.nanmedian(np.log10(F6[(Lx > 43.75) & (Lx < 44.25)]))
# med4_3 = np.nanmedian(np.log10(F6[(Lx > 43.25) & (Lx < 43.75)]))
# med5_3 = np.nanmedian(np.log10(F6[(Lx > 42.75) & (Lx < 43.25)]))

z = np.polyfit(Lx,np.log10(F6),1)
p = np.poly1d(z)

xrange = np.linspace(43,47,100)
yout = p(xrange)

# plt.figure(figsize=(8,8))
# plt.plot(Lx[GOOD_SED], np.log10(F6[GOOD_SED]), '.',ms=8,alpha=0.7,label='all')
# plt.plot(Lx[BAD_SED],np.log10(F6[BAD_SED]),'.',ms=7,alpha=0.7,label='Removed Sources')
# # plt.plot(Lx[GOOD_6], np.log10(F6[GOOD_6]), '.', color='purple', label='6micron sample')
# # plt.plot(45.25, med1_2, 'X', color='k',ms=10)
# # plt.plot(44.75, med2_2, 'X', color='k',ms=10)
# # plt.plot(44.25, med3_2, 'X', color='k',ms=10)
# # plt.plot(43.75, med4_2, 'X', color='k',ms=10)
# # plt.plot(43.25, med5_2, 'X', color='k', ms=10)
# plt.plot(45.25, med1, 'X', color='k', ms=13)
# plt.plot(44.75, med2, 'X', color='k', ms=13)
# plt.plot(44.25, med3, 'X', color='k', ms=13)
# plt.plot(43.75, med4, 'X', color='k', ms=13)
# plt.plot(43.25, med5, 'X', color='k', ms=13)
# # plt.plot(45.25, med1_2, 'X', color='green', ms=10)
# # plt.plot(44.75, med2_2, 'X', color='green', ms=10)
# # plt.plot(44.25, med3_2, 'X', color='green', ms=10)
# # plt.plot(43.75, med4_2, 'X', color='green', ms=10)
# # plt.plot(43.25, med5_2, 'X', color='green', ms=10)
# # plt.plot(45, med1_3, 'X', color='b', ms=10)
# # plt.plot(44.5, med2_3, 'X', color='b', ms=10)
# # plt.plot(44, med3_3, 'X', color='b', ms=10)
# # plt.plot(43.5, med4_3, 'X', color='b', ms=10)
# # plt.plot(43, med5_3, 'X', color='b', ms=10)
# # plt.plot(xrange,yout,'--',color='r',lw=2)
# plt.plot(stern(xrange),xrange,color='red',lw=3,label='Stern 2015')
# # plt.xlim(42.5, 46)
# # plt.ylim(42.25, 45.75)
# plt.xlabel(r'L$_{\rm X}$')
# plt.ylabel(r'L (6$\mu$m)')
# plt.legend()
# plt.grid()
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_check/MIR_Lx_sample1.pdf')
# plt.show()

# med1 = np.nanmedian(np.log10(F025[(Lx > 45) & (Lx < 45.5)]))
# med2 = np.nanmedian(np.log10(F025[(Lx > 44.5) & (Lx < 45)]))
# med3 = np.nanmedian(np.log10(F025[(Lx > 44) & (Lx < 44.5)]))
# med4 = np.nanmedian(np.log10(F025[(Lx > 43.5) & (Lx < 44)]))
# med5 = np.nanmedian(np.log10(F025[(Lx > 43) & (Lx < 43.5)]))

med1 = np.nanmedian(np.log10(F025[b1]))
med2 = np.nanmedian(np.log10(F025[b2]))
med3 = np.nanmedian(np.log10(F025[b3]))
med4 = np.nanmedian(np.log10(F025[b4]))
med5 = np.nanmedian(np.log10(F025[b5]))

per1_25 = np.nanpercentile(np.log10(F025[b1]),20)
per2_25 = np.nanpercentile(np.log10(F025[b2]),20)
per3_25 = np.nanpercentile(np.log10(F025[b3]),20)
per4_25 = np.nanpercentile(np.log10(F025[b4]),20)
per5_25 = np.nanpercentile(np.log10(F025[b5]),20)

per1_75 = np.nanpercentile(np.log10(F025[b1]), 80)
per2_75 = np.nanpercentile(np.log10(F025[b2]), 80)
per3_75 = np.nanpercentile(np.log10(F025[b3]), 80)
per4_75 = np.nanpercentile(np.log10(F025[b4]), 80)
per5_75 = np.nanpercentile(np.log10(F025[b5]), 80)

med1_2 = np.nanmedian(np.log10(F6[b1][(np.log10(F6[b1]) > per1_25)&(np.log10(F6[b1]) < per1_75)]))
med2_2 = np.nanmedian(np.log10(F6[b2][(np.log10(F6[b2]) > per2_25)&(np.log10(F6[b2]) < per2_75)])) 
med3_2 = np.nanmedian(np.log10(F6[b3][(np.log10(F6[b3]) > per3_25)&(np.log10(F6[b3]) < per3_75)]))
med4_2 = np.nanmedian(np.log10(F6[b4][(np.log10(F6[b4]) > per4_25)&(np.log10(F6[b4]) < per4_75)]))
med5_2 = np.nanmedian(np.log10(F6[b5][(np.log10(F6[b5]) > per5_25)&(np.log10(F6[b5]) < per5_75)]))

z = np.polyfit(Lx,np.log10(F025),1)
p = np.poly1d(z)

xrange = np.linspace(43,47,100)
yout = p(xrange)

# plt.figure(figsize=(8, 8))
# plt.plot(Lx[GOOD_SED], np.log10(F025[GOOD_SED]), '.',ms=8,alpha=0.7,label='all')
# plt.plot(Lx[BAD_SED], np.log10(F025[BAD_SED]), '.',ms=7,alpha=0.7,label='Removed Sources')
# plt.plot(45.25, med1, 'X', color='k', ms=12)
# plt.plot(44.75, med2, 'X', color='k', ms=12)
# plt.plot(44.25, med3, 'X', color='k', ms=12)
# plt.plot(43.75, med4, 'X', color='k', ms=12)
# plt.plot(43.25, med5, 'X', color='k', ms=12)
# # plt.plot(45.25, med1_2, 'X', color='green', ms=10)
# # plt.plot(44.75, med2_2, 'X', color='green', ms=10)
# # plt.plot(44.25, med3_2, 'X', color='green', ms=10)
# # plt.plot(43.75, med4_2, 'X', color='green', ms=10)
# # plt.plot(43.25, med5_2, 'X', color='green', ms=10)
# # plt.plot(xrange,yout,'--',color='r',lw=2)
# # plt.xlim(42.5,46)
# # plt.ylim(42.25,45.75)
# plt.plot(Just_alpha_ox(xrange),xrange,color='blue',lw=3,label='Just 2007')
# plt.xlabel(r'L$_{\rm X}$')
# plt.ylabel(r'L (0.25$\mu$m)')
# plt.legend()
# plt.grid()
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_check/UV_Lx_sample1.pdf')
# plt.show()






good_shape = shape[GOOD_SED]
bad_shape = shape[BAD_SED]
print('GOOD: ',len(shape[GOOD_6]))
print('BAD: ',len(shape[BAD_6]))
print('TOTAL: ',len(shape[GOOD_6])+len(shape[BAD_6]))

# plt.figure(figsize=(8,8))
# plt.hist(shape[GOOD_SED]-0.5,bins=np.arange(0.5,7.5,1),histtype='step',color='b',lw=3.5,label='GOOD')
# plt.hist(shape[BAD_SED]-0.5,bins=np.arange(0.5,7.5,1),histtype='step',color='r',lw=2,label='Bad')
# plt.legend()
# plt.show()


# print('shape 1: ',len(good_shape[good_shape == 1]), len(bad_shape[bad_shape == 1]))
# print('shape 2: ',len(good_shape[good_shape == 2]), len(bad_shape[bad_shape == 2]))
# print('shape 3: ',len(good_shape[good_shape == 3]), len(bad_shape[bad_shape == 3]))
# print('shape 4: ',len(good_shape[good_shape == 4]), len(bad_shape[bad_shape == 4]))
# print('shape 5: ',len(good_shape[good_shape == 5]), len(bad_shape[bad_shape == 5]))



print(len(Lx[field == 'S82X']))
print(len(Lx[field == 'COSMOS']))
print(len(Lx[np.logical_or(field == 'GOODS-S', field == 'GOODS-N')]))

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
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_six/Lx_sample.pdf')
# plt.show()

# Fig 3
# plot.multi_SED('a_check/All_SEDs_temp',int_x,int_y,wfir,ffir,wave_labels=True,temp_comp=True,temp_comp_x=temp_wave,temp_comp_y=temp_nuFnu_norm)
# plot.multi_SED('a_check/All_SEDs_sample_bad', int_x[GOOD_SED], int_y[GOOD_SED], wfir[GOOD_SED], ffir[GOOD_SED], wave_labels=True)
# plot.multi_SED('a_six/All_SEDs', int_x[GOOD_6], int_y[GOOD_6], wfir[GOOD_6], ffir[GOOD_6], wave_labels=True)

# Fig 4
# plot.L_hist('a_six/Lone_hist',np.log10(F1[GOOD_6]),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=True)

# Fig 5
# plot.multi_SED_bins('a_six/All_z_bins_norm','redshift',field[GOOD_6],median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6])

# Fig 6 


# Fig 7 
# plot.multi_SED_bins('a_six/All_Lx_bins_norm',bin='Lx',field=field[GOOD_6],median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],Median_line=True,FIR_upper='upper lims',scale=True)

# Fig 8 
# plot.L_scatter_3panels('a_new/AGN_emission_s82x_3sig','UV-MIR-FIR','Lx','X-axis',norm,F025,F6,F100,shape,Lx,uv_err=F025_err,mir_err=F6_err,fir_err=F100_err,error=False)
# plot.L_scatter_3panels('a_new/AGN_emission_1sig','UV-MIR-FIR','Lx','X-axis',norm,F025,F6,F100,shape,Lx,error=False,stack_color=True,stack_bins=stack_bin,F100_ratio=F100_ratio,field = field,fir_field=True)
# plot.L_scatter_3panels('a_new/AGN_emission_medians', 'Lx', 'UV-MIR-FIR', 'X-axis', norm, F025, F6, F100, shape,Lx, error=False, stack_color=True, stack_bins=stack_bin, F100_ratio=F100_ratio, field=field, fir_field=True)
# plot.L_scatter_3panels_vert('a_six/AGN_emission', 'Lx', 'UV-MIR-FIR', 'X-axis', norm[GOOD_6], F025[GOOD_6], F6[GOOD_6], F100[GOOD_6], shape[GOOD_6],Lx[GOOD_6], error=False, stack_color=True, stack_bins=stack_bin[GOOD_6], F100_ratio=F100_ratio[GOOD_6], field=field[GOOD_6], fir_field=True)

# plot.L_scatter_3panels('a_six/AGN_emission_sample1', 'Lx', 'UV-MIR-FIR', 'X-axis', norm[GOOD_6], F025[GOOD_6], F6[GOOD_6], F100[GOOD_6], shape[GOOD_6],Lx[GOOD_6], error=False, stack_color=True, stack_bins=stack_bin[GOOD_6], F100_ratio=F100_ratio[GOOD_6], field=field[GOOD_6], fir_field=True)


# One panel hist
# plot.L_hist('a_new/Lbol_hist_update',np.log10(Lbol),r'Total log $L_{\rm bol}/({\rm erg \; s^{-1}})$',[43,47],[43,47,0.25],std=True,top_label=True,xlabel2=r'Total log $L_{\rm bol}/\rm{L_\odot}$')
# plot.L_hist('a_new/Nh_hist_update',np.log10(Nh),r'log $N_{\rm H}/(\rm{cm}^{-2})$', [20,24.5], [20,24.5,0.25],split=True,split_param=Nh_upper)

# Fig 11 (Lbol/Lx scatter)
# plot.L_ratio_1panel('a_new/Lx_Lbol_update','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape,np.log10(Lbol_sub))
# plot.L_ratio_1panel('a_new/Lx_Lbol_spec_type','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape,np.log10(Lbol_sub),sample=True,spec_type=spec_type)

# Fig 12
# plot_shape.shape_1bin_v('a_check/vertical_5_panel_check_all6',median_x=int_x,median_y=int_y,wfir=wfir,ffir=ffir,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims',bins='shape')
# plot_shape.shape_1bin_v('a_six/vertical_5_panel',median_x=int_x[GOOD_6],median_y=int_y[GOOD_6],wfir=wfir[GOOD_6],ffir=ffir[GOOD_6],uv_slope=uv_slope[GOOD_6],mir_slope1=mir_slope1[GOOD_6],mir_slope2=mir_slope2[GOOD_6],Median_line=True,FIR_upper='upper lims',bins='shape')

# Fig 14 (Med SED)
# plot.median_SED_1panel('a_check/median_SED_update',int_x[GOOD_SED],int_y[GOOD_SED],wfir[GOOD_SED],ffir[GOOD_SED],shape[GOOD_SED],plot_temp=True,temp_x=temp_wave,temp_y=temp_nuLnu*1E6)

# Multi-panel hist
# plot_shape.L_hist_panels('a_new/Lone_hist_panels_update',np.log10(F1),r'log L (1 $\mu$m)/(erg s$^{-1}$)',[43.25,46],[43.25,46,0.25],z_label=True,top_label=True,xlabel2=r'log L (1 $\mu$m)/$\rm{L_\odot}$')
# plot_shape.L_hist_panels('a_check/Lx_hist_panels_all',Lx,r'log $L_{\rm X}$/(erg s$^{-1}$)',[43,46],[43,46,0.25],z_label=True)
# plot_shape.L_hist_panels('a_new/Nh_hist_panel_update',np.log10(Nh),r'log $N_{\rm H}/(\rm{cm}^{-2})$', [20,24.5],[20,24.5,0.25],split=True,split_param=Nh_upper)

# plot_shape.L_hist_panels2('a_new/Lbol_Lx_hist_panels_update', np.log10(Lbol_sub), np.log10(Lbol_sub)-Lx, r'log $L_{\rm bol-gal,e}$',[44,47],[44,47,0.25],[0,3],[0,3,0.25])

# Panel Scatter
# plot2.scatter_1panel('a_new/FIR_Lx_update','Lx','FIR/Lx',None,'Bins',Nh,Lx,np.log10(Lbol_sub),np.log10(F1),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape)

# Ratio Plots
# plot.L_ratio_multi_panel('a_new/ratio_multipanel_1sig_nostack','Lx','AGN','bins',np.log10(F1),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(Nh),np.log10(Lbol_sub),shape,FIR_upper_lims,F100_ratio=F100_ratio,field=field)

# U-panel plots
# plot2.Upanels_ratio_plots('a_new/Nh_Upanels_update','Nh','UV/MIR-UV/Lx-MIR/Lx','Bins',Nh,Lx,Lbol_sub,np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=shape,Nh_upper=Nh_upper)
# plot.Upanels_ratio('a_new/Lum_Lbol_update','Lbol','UV-MIR-FIR','Bins',np.log10(Lbol_sub),np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F1),shape,FIR_upper_lims)
