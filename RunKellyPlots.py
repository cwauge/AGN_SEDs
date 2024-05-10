import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii
from SED_plots_v2 import Plotter
from SED_shape_plots import SED_shape_Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import kstest
from match import match
from SED_v8 import Flux_to_Lum
from match import match


path = '/Users/connor_auge/Research/Disertation/catalogs/'
# AHA_SEDs_out_good_1sig
# AHA_SEDs_out_ALL
with fits.open(path+'Kelly_SEDs_out5.fits') as hdul:
    cols = hdul[1].columns
    data = hdul[1].data


field = data['field']
id = data['ID']  # [field == 'S82X']
z = data['z']  # [field == 'S82X']
x = data['x']  # [field == 'S82X']
y = data['y']  # [field == 'S82X']
Lx = data['Lx']  # [field == 'S82X']
kelly_Lx = data['kelly_Lx']
norm = data['norm']  # [field == 'S82X']
FIR_upper_lims = data['FIR_upper_lims']  # [field == 'S82X']
frac_err = data['frac_err']  # [field == 'S82X']
group = data['group']
Nh = data['Nh']

uv_slope = data['uv_slope']  # [field == 'S82X']
mir_slope1 = data['mir_slope1']  # [field == 'S82X']
mir_slope2 = data['mir_slope2']  # [field == 'S82X']

F025 = data['F025']  # [field == 'S82X']
F1 = data['F1']  # [field == 'S82X']
F6 = data['F6']  # [field == 'S82X']
F10 = data['F10']  # [field == 'S82X']
F100 = data['F100']  # [field == 'S82X']
F100_ratio = data['F100_ratio']
# F025_boot = data['F025_boot'][field == 'COSMOS']
# F1_boot = data['F1_boot'][field == 'COSMOS']
# F6_boot = data['F6_boot'][field == 'COSMOS']
# F10_boot = data['F10_boot'][field == 'COSMOS']
# F100_boot = data['F100_boot'][field == 'COSMOS']
uv_lum = data['UV_lum']  # [field == 'S82X']
mir_lum = data['MIR_lum']  # [field == 'S82X']
fir_lum = data['FIR_lum']  # [field == 'S82X']

abs_corr = data['abs_corr']
abs_check = data['abs_check']

check_sed = data['sed_check']
check_sed6 = data['check6']

ch1 = data['irac_ch1']
ch2 = data['irac_ch2']
ch3 = data['irac_ch3']
ch4 = data['irac_ch4']

# F025_err = (np.std(np.log10(F025_boot),axis=1)/1000**(1/2))*3
# F6_err = (np.std(np.log10(F6_boot),axis=1)/1000**(1/2))*3

# F100_err = (np.std(np.log10(F100_boot),axis=1)/1000**(1/2))*3

shape = data['shape']  # [field == 'S82X']
Lbol = data['Lbol']  # [field == 'S82X']
Lbol_sub = data['Lbol_sub']  # [field == 'S82X']
Lbol_sf_sub = data['Lbol_sf_sub']

Nh = data['Nh']  # [field == 'S82X']
Nh_upper = data['Nh_check']  # [field == 'S82X']

int_x = data['int_x']  # [field == 'S82X']
int_y = data['int_y']  # [field == 'S82X']
wfir = data['wfir']  # [field == 'S82X']
ffir = data['ffir']  # [field == 'S82X']

spec_type = data['spec_class']  # [field == 'S82X']

F24 = data['F24_lum']  # [field == 'S82X']
stack_bin = data['stack_bin']  # [field == 'S82X']

field = field  # [field == 'S82X']
s82x_z = z  # [field == 'S82X']


check_sed = data['sed_check']

# samp = (group == 'BLU')
# samp = (group == 'GRN')
samp = (group == 'RED')
# samp = (group == 'RED') | (group == 'GRN') | (group == 'BLU')  


plot = Plotter(id[samp], z[samp], x[samp], y[samp], Lx[samp], norm[samp], FIR_upper_lims[samp])
plot_shape = SED_shape_Plotter(id[samp], z[samp], x[samp], y[samp], Lx[samp], norm[samp], FIR_upper_lims[samp], shape[samp])

print('HERE: ',np.shape(x),np.shape(x[samp]))

# outf = open('/Users/connor_auge/Desktop/KellySample_AugeLx.csv','w')
# outf.writelines('ID,Group,z,Kelly_Lx,Auge_Lx\n')
# for i in range(len(id)):
#     outf.writelines('%s,%s,%f,%f,%f\n' % (id[i],group[i],z[i],kelly_Lx[i],Lx[i]))
# outf.close()

def Flux_to_Lum(F, z, d=np.nan, distance=False):
    '''Function to convert flux to luminosity'''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

    dl = cosmo.luminosity_distance(z).value  # Distance in Mpc
    if distance:
        dl = d/1E6
    dl_cgs = dl*(3.0856E24)  # Distance from Mpc to cm

    # convert flux to luminosity
    L = F*4*np.pi*dl_cgs**2

    return L

def stern(mir):
    a = 40.981
    b = 1.024
    c = 0.047
    # x = np.log10(mir/1E41)
    x = mir - 41

    Lx = a+b*x-c*x**2
    Lx += np.log10(1.64)

    return Lx

def chen(mir):
    if mir < 44.63:
        lx = 0.84*(mir-45)+44.58

    else:
        lx = 0.54*(mir-45)+44.47

    Lx = lx + np.log10(1.64)

    return Lx

def Durras_Lbol(L,typ,err=False):
    '''X-ray to bolometric correction from Durras et al. 2020'''
    if typ == 'Lx':
        a, b, c = 15.33, 11.48, 16.20
        alo, blo, clo = 15.33-0.06, 11.48-0.01, 16.20-0.16
        aup, bup, cup = 15.33+0.06, 11.48+0.01, 16.20+0.16
        std = 0.37
    elif typ == 'Lbol':    
        a, b, c = 10.96, 11.93, 17.79
        alo, blo, clo = 10.96-0.06, 11.93-0.01, 17.79-0.10
        aup, bup, cup = 10.96+0.06, 11.93+0.01, 17.79+0.10
        std = 0.27
    else:
        print('Specify typ. Options are:    Lx    Lbol')
        return

    # L += np.log10(0.611)

    kx = a*(1+((L - np.log10(3.8E33))/b)**c)
    kx *= 1/1.64
    kx_lo = alo*(1+((L - np.log10(3.8E33))/blo)**clo)
    kx_up = aup*(1+((L - np.log10(3.8E33))/bup)**cup)
    kx_lo = kx - std
    kx_up = kx + std
    if err:
        return kx, kx_up, kx_lo
    else:
        return kx

# xtot_values, xtot_bins = np.histogram(kelly_Lx[(group == 'BLU') | (group == 'GRN')],bins=np.arange(42.0,46,0.25))
# xIR_values, xIR_bins = np.histogram(kelly_Lx[(group == 'GRN')],bins=np.arange(42.0,46,0.25))

# irtot_values, irtot_bins = np.histogram(kelly_Lx[(group == 'RED') | (group == 'GRN')],bins=np.arange(42.0,46,0.25))
Lx_use = Lx.copy()
for i in range(len(Lx)):
    if np.isnan(Lx[i]):
        Lx_use[i] = chen(np.log10(F6[i]))
    else:
        continue

Kx = Durras_Lbol(Lx_use,typ='Lx')
print(Kx)

Lbol = 10**Lx_use + Kx
print(Lbol)


xtot_values, xtot_bins = np.histogram(np.log10(F6[(group == 'BLU') | (group == 'GRN')]),bins=np.arange(42.0,50,0.25))
xIR_values, xIR_bins = np.histogram(np.log10(F6[(group == 'GRN')]),bins=np.arange(42.0,50,0.25))

xxtot_values, xxtot_bins = np.histogram(Lx[(group == 'BLU') | (group == 'GRN')],bins=np.arange(42.0,50,0.25))
xxIR_values, xxIR_bins = np.histogram(Lx[(group == 'GRN')],bins=np.arange(42.0,50,0.25))

irtot_values, irtot_bins = np.histogram(np.log10(F6[(group == 'RED') | (group == 'GRN')]),bins=np.arange(42.0,50,0.25))


lbol_xtot_values, lbol_xtot_bins = np.histogram(np.log10(Lbol[(group == 'BLU') | (group == 'GRN')]),bins=np.arange(42.0,50,0.25))
lbol_xIR_values, lbol_xIR_bins = np.histogram(np.log10(Lbol[(group == 'GRN')]),bins=np.arange(42.0,50,0.25))
lbol_irtot_values, lbol_irtot_bins = np.histogram(np.log10(Lbol[(group == 'RED') | (group == 'GRN')]),bins=np.arange(42.0,50,0.25))


xcumulative = np.cumsum(xIR_values/xtot_values)
ircumulative = np.cumsum(xIR_values/irtot_values)

plt.figure(figsize=(10,10))
plt.bar(xIR_bins[:-1],xIR_values,alpha=0.25,width=np.diff(xIR_bins), align='edge',label='Green')
plt.bar(xtot_bins[:-1],xtot_values,alpha=0.25,width=np.diff(xtot_bins), align='edge',label='Blue+Green')
plt.legend()
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm 6\mu m}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Number')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(xIR_bins[:-1],xIR_values/xtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm 6\mu m}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of X-ray AGN identifed by IR')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(xxIR_bins[:-1],xxIR_values/xxtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm X}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of X-ray AGN identifed by IR')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(lbol_xIR_bins[:-1],lbol_xIR_values/lbol_xtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm bol}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of X-ray AGN identifed by IR')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(xIR_bins[:-1],xIR_values,alpha=0.25,width=np.diff(xIR_bins), align='edge',label='Green')
plt.bar(irtot_bins[:-1],irtot_values,alpha=0.25,width=np.diff(irtot_bins), align='edge',label='Red+Green')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm 6\mu m}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Number')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.bar(xIR_bins[:-1],xIR_values/irtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm 6\mu m}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of IR AGN identifed by X-ray')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(xxIR_bins[:-1],xxIR_values/irtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm X}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of IR AGN identifed by X-ray')
plt.show()

plt.figure(figsize=(10,10))
plt.bar(lbol_xIR_bins[:-1],lbol_xIR_values/lbol_irtot_values,width=np.diff(xIR_bins), align='edge', color='gray')
# plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
plt.xlabel(r'log L$_{\rm bol}$ [$\rm{erg\,s^{-1}}$]')
plt.ylabel('Fraction of IR AGN identifed by X-ray')
plt.show()

outf = open('/Users/connor_auge/Desktop/Completeness.csv','w')
outf.writelines('Lx_bin_size,L6_bin_size,Frac_of_IR_IDby_Lx_L6,Frac_of_Xray_IDby_IR_Lx,Frac_of_Xray_IDby_IR_L6\n')
for i in range(len(xIR_bins[:-1])):
    outf.writelines('%f,%f,%f,%f,%f\n' % (xxIR_bins[:-1][i],xIR_bins[:-1][i],xIR_values[i]/irtot_values[i],xxIR_values[i]/xxtot_values[i],xIR_values[i]/xtot_values[i]))

outf.close()


# plt.figure(figsize=(10,10))
# plt.plot(xIR_bins[:-1],xcumulative,color='b')
# plt.plot(xIR_bins[:-1],ircumulative,color='r')
# # plt.xlabel(r'L$_{\rm X(0.5 - 10keV)}[\rm{erg\,s^{-1}}]')
# plt.xlabel(r'log L$_{\rm 6\mu m}$ [$\rm{erg\,s^{-1}}$]')
# plt.show()

n, bins, patches = plt.hist(xIR_values/xtot_values, xIR_bins[:-1], density=True, histtype="step",
                               cumulative=True, label="Cumulative histogram")
plt.show()


x1d = np.linspace(0.08, 1.5)
x2d = np.linspace(0.35, 2.0)

x1L = np.linspace(-0.3, 1.5)
# plt.figure(figsize=(12,12))
# plt.plot(x1d, 1.21*x1d + 0.27,color='k',lw=3,label='Donley et al. 2012')
# plt.plot(x2d, 1.21*x2d - 0.27,color='k',lw=3)
# plt.vlines(0.08,ymin=0.15,ymax=0.37,color='k',lw=3)
# plt.hlines(0.15,xmin=0.08,xmax=0.35,color='k',lw=3)
# plt.plot(np.log10(ch3[spec_type == 1]/ch1[spec_type == 1]),np.log10(ch4[spec_type == 1]/ch2[spec_type == 1]),'.',color='blue',label='type 1',ms=10)
# plt.plot(np.log10(ch3[spec_type == 2]/ch1[spec_type == 2]),np.log10(ch4[spec_type == 2]/ch2[spec_type == 2]),'.',color='r', label='type 2',ms=10)
# plt.xlim(-0.5,1.)
# plt.ylim(-0.5,1.)
# plt.xlabel(r'log $\frac{f_{5.8}}{f_{3.4}}$',fontsize=26)
# plt.ylabel(r'log $\frac{f_{8.0}}{f_{4.5}}$',fontsize=26)
# plt.grid()
# plt.legend(fontsize=13)
# plt.savefig('/Users/connor_auge/Desktop/ir_color_type2.png')
# plt.show()
# plt.close()

# plot.IR_colors('ir_color_spec_type',np.log10(ch3/ch1),np.log10(ch4/ch2),kelly_Lx,np.log10(ch3/ch1),np.log10(ch4/ch2),colorbar=True,colorbar_label=r'L$_{\rm X}$ [erg/s]',agn=(group=='BLU') | (group=='GRN'))



Fx_w_cgs = 2.36*1E-8
Fx_freq = 3E10/Fx_w_cgs
Fx_lim = 8.9E-16
Fx_lim_jy = (Fx_lim*4.136E8/(10-0.5))/1000

Fx_nu = Fx_lim_jy*1E-23
Fx_lim_nuFnu = Fx_nu*Fx_freq

stern_Lx_int = []
stern_Lx_obs = []
stern_F6 = []

chen_Lx_int = []
chen_Lx_obs = []
chen_F6 = []
abs_corr_use = []
for i in range(len(F6)):
    if np.isnan(Lx[i]):
        stern_Lx_int.append(10**stern(np.log10(F6[i])))
        stern_Lx_obs.append(Flux_to_Lum(Fx_lim_nuFnu, z[i]))
        stern_F6.append(F6[i])

        chen_Lx_int.append(10**chen(np.log10(F6[i])))
        chen_Lx_obs.append(Flux_to_Lum(Fx_lim_nuFnu, z[i]))
        chen_F6.append(F6[i])

        abs_corr_use.append(Flux_to_Lum(Fx_lim_nuFnu, z[i])/(10**chen(np.log10(F6[i]))))
    else:
        chen_Lx_int.append(10**Lx[i])
        chen_Lx_obs.append((10**Lx[i])*abs_corr[i])
        chen_F6.append(F6[i])

        stern_Lx_int.append(10**Lx[i])
        stern_Lx_obs.append(10**(Lx[i])*abs_corr[i])
        stern_F6.append(F6[i])

        abs_corr_use.append(abs_corr[i])


stern_Lx_int = np.asarray(stern_Lx_int)
stern_Lx_obs = np.asarray(stern_Lx_obs)

chen_Lx_int = np.asarray(chen_Lx_int)
chen_Lx_obs = np.asarray(chen_Lx_obs)

abs_corr_use = np.asarray(abs_corr_use)



# plt.figure(figsize=(7,7))
# plt.plot(np.log10(F6),kelly_Lx,'.')
# plt.plot(np.log10(stern_F6),np.log10(stern_Lx_int),'.')
# plt.plot(np.log10(chen_F6),np.log10(chen_Lx_obs),'x',color='k')
# plt.plot(np.arange(40,49),stern(np.arange(40,49)),color='k')
# plt.xlim(40,47)
# plt.ylim(40,47)
# plt.text(42, 41.5, f'N={len(F6)}')
# plt.grid()
# plt.ylabel(r'log L$_{\rm{X}}$')
# plt.xlabel(r'log L$_{6\mu \rm{m}}$')
# plt.show()


# plt.figure(figsize=(9,9))
# plt.plot(np.log10(abs_corr),np.log10(Nh),'.')
# plt.xlabel(r'log L$_{\rm X, obs}$/L$_{\rm X, int}$')
# plt.ylabel(r'log N$_{\rm H}$')
# plt.show()


# plot_shape.shape_1bin_v('Kelly_plots/RED_5panel',median_x=int_x[samp],median_y=int_y[samp],wfir=wfir[samp],ffir=ffir[samp],uv_slope=uv_slope[samp],mir_slope1=mir_slope1[samp],mir_slope2=mir_slope2[samp],Median_line=True,FIR_upper='data only',bins='shape')

# # print(min(np.log10(F1)))

# plt.hist(np.log10(F1[z <= 1]),bins=np.arange(37,47,0.5),color='gray',alpha=0.5)
# plt.hist(np.log10(F1[z <= 1][group[z <= 1] == 'RED']),bins=np.arange(38,48,0.5), histtype='step',color='red',alpha=0.75,lw=2)
# plt.hist(np.log10(F1[z <= 1][group[z <= 1] == 'GRN']),bins=np.arange(38,48,0.5), histtype='step',color='green',alpha=0.75,lw=2)
# plt.hist(np.log10(F1[z <= 1][group[z <= 1] == 'BLU']), bins=np.arange(38, 48, 0.5), histtype='step', color='blue', alpha=0.75,lw=2)
# plt.axvline(np.nanmean(np.log10(F1[z <= 1][group[z <= 1] == 'RED'])),color='r',ls='--')
# plt.axvline(np.nanmean(np.log10(F1[z <= 1][group[z <= 1] == 'GRN'])),color='b',ls='--')
# plt.axvline(np.nanmean(np.log10(F1[z <= 1][group[z <= 1] == 'BLU'])), color='green', ls='--')

# plt.xlabel(r'log L$_{1\mu \rm{m}}$')
# plt.xlim(37.75,48)
# plt.show()


# plt.hist(np.log10(F6[z <= 10]),bins=np.arange(37,47,0.5),color='gray',alpha=0.5)
# plt.hist(np.log10(F6[z <= 10][group[z <= 10] == 'RED']),bins=np.arange(38,48,0.5), histtype='step',color='red',alpha=0.75,lw=2)
# plt.hist(np.log10(F6[z <= 10][group[z <= 10] == 'GRN']),bins=np.arange(38,48,0.5), histtype='step',color='green',alpha=0.75,lw=2)
# plt.hist(np.log10(F6[z <= 10][group[z <= 10] == 'BLU']), bins=np.arange(38, 48, 0.5), histtype='step', color='blue', alpha=0.75,lw=2)
# plt.axvline(np.log10(np.nanmean(F6[z <= 10][group[z <= 10] == 'RED'])),color='r',ls='--')
# plt.axvline(np.log10(np.nanmean(F6[z <= 10][group[z <= 10] == 'GRN'])),color='green',ls='--')
# plt.axvline(np.log10(np.nanmean(F6[z <= 10][group[z <= 10] == 'BLU'])), color='b', ls='--')

# plt.xlabel(r'log L$_{6\mu \rm{m}}$')
# plt.xlim(37.75,48)
# plt.show()

# plt.plot(Lx,kelly_Lx,'.',color='r')
# # plt.plot(Lx[abs_check==1],kelly_Lx[abs_check==1],'.',color='b')
# # plt.plot(Lx[abs_check==2],kelly_Lx[abs_check==2],'.',color='g')
# plt.plot(np.arange(40,49),np.arange(40,49),color='k')
# plt.xlim(40,48)
# plt.ylim(40,48)
# plt.show()

print(stern_Lx_obs)
print(stern_Lx_int)
print(stern_Lx_obs/stern_Lx_int)

print(np.log10(min(abs_corr[abs_corr > 0])))


# plt.figure(figsize=(8,8))
# plt.hist(np.log10(abs_corr[group == 'BLU']),bins=np.arange(-3,1,0.1),histtype='step',color='b',lw=3,alpha=0.75,label='X-ray Exclusive')
# plt.hist(np.log10(abs_corr[group == 'GRN']),bins=np.arange(-3,1,0.1),histtype='step',color='g',lw=3,alpha=0.75,label='X-ray Inclusive')
# # plt.hist(abs_corr[group == 'RED'],bins=np.arange(0,1,0.1),histtype='step',color='orange',lw=3,alpha=0.75,label='MIR Inclusive')
# # plt.hist(np.log10(stern_Lx_obs/stern_Lx_int),bins=np.arange(-3,1,0.1),histtype='step',color='r',lw=3,alpha=0.75,label='MIR Exclusive')
# plt.hist(np.log10(chen_Lx_obs/chen_Lx_int),bins=np.arange(-3,1,0.1),histtype='step',color='r',lw=3,alpha=0.75,label='MIR Exclusive')

# plt.axvline(np.nanmedian(np.log10(abs_corr[group == 'BLU'])),ls='--',color='b',lw=3)
# plt.axvline(np.nanmedian(np.log10(abs_corr[group == 'GRN'])),ls='--',color='g',lw=3)
# # plt.axvline(np.nanmedian(np.log10(stern_Lx_obs/stern_Lx_int)),ls='--',color='r',lw=3)
# plt.axvline(np.nanmedian(np.log10(chen_Lx_obs/chen_Lx_int)),ls='--',color='r',lw=3)

# plt.xlabel(r'log L$_{\rm X, obs}$/L$_{\rm X, int}$')
# plt.grid()
# plt.legend(loc='upper left')
# plt.show()

plt.figure(figsize=(8,8))
plt.hist(np.log10(abs_corr_use[group == 'BLU']),bins=np.arange(-3,1,0.1),histtype='step',color='b',lw=3,alpha=0.75,label='X-ray Exclusive')
plt.hist(np.log10(abs_corr_use[group == 'GRN']),bins=np.arange(-3,1,0.1),histtype='step',color='g',lw=3,alpha=0.75,label='X-ray Inclusive')
plt.hist(np.log10(abs_corr_use[group == 'RED']),bins=np.arange(-3,1,0.1),histtype='step',color='r',lw=3,alpha=0.75,label='MIR Exclusive')

# plt.hist(abs_corr[group == 'RED'],bins=np.arange(0,1,0.1),histtype='step',color='orange',lw=3,alpha=0.75,label='MIR Inclusive')
# plt.hist(np.log10(stern_Lx_obs/stern_Lx_int),bins=np.arange(-3,1,0.1),histtype='step',color='r',lw=3,alpha=0.75,label='MIR Exclusive')
# plt.hist(np.log10(chen_Lx_obs/chen_Lx_int),bins=np.arange(-3,1,0.1),histtype='step',color='r',lw=3,alpha=0.75,label='MIR Exclusive')

plt.axvline(np.nanmedian(np.log10(abs_corr_use[group == 'BLU'])),ls='--',color='b',lw=3)
plt.axvline(np.nanmedian(np.log10(abs_corr_use[group == 'GRN'])),ls='--',color='g',lw=3)
plt.axvline(np.nanmedian(np.log10(abs_corr_use[group == 'RED'])),ls='--',color='r',lw=3)
# plt.axvline(np.nanmedian(np.log10(stern_Lx_obs/stern_Lx_int)),ls='--',color='r',lw=3)
# plt.axvline(np.nanmedian(np.log10(chen_Lx_obs/chen_Lx_int)),ls='--',color='r',lw=3)

plt.xlabel(r'log L$_{\rm X, obs}$/L$_{\rm X, int}$')
plt.grid()
plt.legend(loc='upper left')
# plt.show()
plt.close()


# outf = open('/Users/connor_auge/Research/REU/2022/Thresa/Lx_correction_plot_data_update3.csv','w')
# outf.writelines('# ID,Lx_corr,log_Lx_int,log_Lx_obs,group\n')
# for i in range(len(group[group == 'BLU'])):
#     outf.writelines('%s,%f,%f,%f,%s\n' % (id[group == 'BLU'][i],abs_corr_use[group == 'BLU'][i],np.log10(chen_Lx_int[group == 'BLU'][i]),np.log10(chen_Lx_obs[group == 'BLU'][i]),'BLU'))
# for i in range(len(group[group == 'GRN'])):
#     outf.writelines('%s,%f,%f,%f,%s\n' % (id[group == 'GRN'][i],abs_corr_use[group == 'GRN'][i],np.log10(chen_Lx_int[group == 'GRN'][i]),np.log10(chen_Lx_obs[group == 'GRN'][i]),'GRN'))
# for i in range(len(group[group == 'RED'])):
#     outf.writelines('%s,%f,%f,%f,%s\n' % (id[group == 'RED'][i],abs_corr_use[group == 'RED'][i],np.log10(chen_Lx_int[group == 'RED'][i]),np.log10(chen_Lx_obs[group == 'RED'][i]),'RED'))
# outf.close()

