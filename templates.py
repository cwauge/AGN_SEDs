import numpy as np 
import matplotlib.pyplot as plt 
from SED_v7 import Flux_to_Lum
from astropy.io import ascii
from scipy import integrate



inf = ascii.read('/Users/connor_auge/Desktop/A10_templates.txt')

wave = np.asarray(inf['Wave'])
wave_cgs = wave*1E-4
freq = 3E10/wave_cgs

E_flux = np.asarray(inf['E'])*1E-16 # erg/s/Hz
E_nuFnu = E_flux*freq

Im_flux = np.asarray(inf['Im'])*1E-15 # erg/s/Hz
Im_nuFnu = Im_flux*freq

Sbc_flux = np.asarray(inf['Sbc'])*1E-17 # erg/s/Hz
Sbc_nuFnu = Sbc_flux*freq

AGN1_flux = np.asarray(inf['AGN'])*1E-11 # erg/s/Hz
AGN1_nuFnu = AGN1_flux*freq

AGN2_flux = np.asarray(inf['AGN2'])*1E-14 # erg/s/Hz
AGN2_nuFnu = AGN2_flux*freq

E_Im_nuFnu = Im_nuFnu+E_nuFnu
E_Im_Sbc_nuFnu = Im_nuFnu+E_nuFnu+Sbc_nuFnu

# AGN1_nuFnu /= AGN1_nuFnu[wave == 1.0050]
# AGN2_nuFnu /= AGN2_nuFnu[wave == 1.0050]
# Sbc_nuFnu /= Sbc_nuFnu[wave == 1.0050]
# Im_nuFnu /= Im_nuFnu[wave == 1.0050]
# E_nuFnu /= E_nuFnu[wave == 1.0050]

E_L = E_nuFnu*4*np.pi*3E19**2
E_AGN1 = AGN1_nuFnu*4*np.pi*3E19**2
E_AGN2 = AGN2_nuFnu*4*np.pi*3E19**2
E_Sbc = Sbc_nuFnu*4*np.pi*3E19**2
E_Im = Im_nuFnu*4*np.pi*3E19**2
E_Im_Sbc = E_Im_Sbc_nuFnu*4*np.pi*3E19**2

def Lbol(freq,nuLnu):

    nuLnu /= freq
     
    bol = integrate.trapz(nuLnu[::-1],freq[::-1])

    bol = bol*4*np.pi*3E19**2
    bol /= 3.8E33

    # plt.plot(freq[::-1],nuLnu[::-1])
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # return bol
    return np.log10(bol)

# plt.plot(wave,E_nuFnu/freq)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


print('Eliptical:',Lbol(freq,E_nuFnu))
print('AGN 1:    ',Lbol(freq,AGN1_nuFnu))
print('AGN 2:    ',Lbol(freq,AGN2_nuFnu))
print('Spiral:   ',Lbol(freq,Sbc_nuFnu))
print('Irregular:',Lbol(freq,Im_nuFnu))
print('E+Im:     ',Lbol(freq,E_Im_nuFnu))
# print('E+Im:     ',Lbol(freq,E_nuFnu)+Lbol(freq,Im_nuFnu))



plt.plot(wave,E_L,label='E')
plt.plot(wave,E_Im,label='Im')
plt.plot(wave,E_Sbc,label='Sbc')
plt.plot(wave,E_L+E_Im,label='E+Im')
plt.plot(wave,E_L+E_Im+E_Sbc,label='E+Im+Sbc')

plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'$\nu$L$_{\nu}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

plt.plot(wave,E_L,label='E')
plt.plot(wave,E_Im,label='Im')
plt.plot(wave,E_Sbc,label='Sbc')
plt.plot(wave,E_AGN1,label='AGN1')
plt.plot(wave,E_AGN2,label='AGN2')
plt.xlabel(r'Wavelength [$\mu$m]')
plt.ylabel(r'$\nu$L$_{\nu}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
