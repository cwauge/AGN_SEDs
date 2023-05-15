import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from match import match 
from Herschel_flux import Herschel

path = '/Users/connor_auge/Research/Disertation/catalogs/'

with fits.open(path+'AHA_SEDs_out_check.fits') as hdul:
    sample = hdul[1].data 

with fits.open(path+'cosmos2020/classic/COSMOS2020_CLASSIC_R1_v2.0_master.fits') as hdul:
    cosmos = hdul[1].data 

sample_id = sample['ID']
sample_field = sample['field']
sample_fir_check = sample['FIR_upper_lims']

cosmos_id = cosmos['ID_COSMOS2015']
cosmos_ra = cosmos['ALPHA_J2000']
cosmos_dec = cosmos['DELTA_J2000']

cond = (sample_field == 'c') & (sample_fir_check == 1)
sample_id = sample_id[cond]
sample_fir_check = sample_fir_check[cond]
sample_field = sample_field[cond]

ix, iy = match(sample_id, cosmos_id)
cosmos_id_match = cosmos_id[iy]
cosmos_ra_match = cosmos_ra[iy]
cosmos_dec_match = cosmos_dec[iy]

print('start boot')
F250_boot, F350_boot, F500_boot = [], [], []
for i in range(len(cosmos_id_match)):
    F250 = Herschel('/Users/connor_auge/Desktop/Herschel_COSMOS/COSMOS-Nest_image_250_SMAP_v6.0.fits',cosmos_ra_match[i],cosmos_dec_match[i])
    F350 = Herschel('/Users/connor_auge/Desktop/Herschel_COSMOS/COSMOS-Nest_image_350_SMAP_v6.0.fits',cosmos_ra_match[i],cosmos_dec_match[i])
    F500 = Herschel('/Users/connor_auge/Desktop/Herschel_COSMOS/COSMOS-Nest_image_500_SMAP_v6.0.fits',cosmos_ra_match[i],cosmos_dec_match[i])
    f250_boot = F250.MC(1000)
    f350_boot = F350.MC(1000)
    f500_boot = F500.MC(1000)

    F250_boot.append(f250_boot)
    F350_boot.append(f350_boot)
    F500_boot.append(f500_boot)
print('done with boot')

F250_boot = np.asarray(F250_boot)
F350_boot = np.asarray(F350_boot)
F500_boot = np.asarray(F500_boot)

F250_mean, F350_mean, F500_mean = [], [], []
for i in range(len(F250_boot)):
    F250_mean.append(np.median(F250_boot[i])) 
    F350_mean.append(np.median(F350_boot[i])) 
    F500_mean.append(np.median(F500_boot[i])) 

F250_mean = np.asarray(F250_mean)
F350_mean = np.asarray(F350_mean)
F500_mean = np.asarray(F500_mean)


plt.hist(F250_mean)
# plt.axvline(x=1.77,color='k')
plt.xlabel('F250 flux [mJy]')
plt.show()

plt.hist(F350_mean)
# plt.axvline(x=2.68,color='k')
plt.xlabel('F350 flux [mJy]')
plt.show()

plt.hist(F500_mean)
# plt.axvline(x=2.91, color='k')
plt.xlabel('F500 flux [mJy]')
plt.show()



