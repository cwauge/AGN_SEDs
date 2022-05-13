import numpy as np 

def mag_to_flux(mag,band,AB=True):

	mag = np.asarray(mag,dtype=float)
	# mag = mag[0]

	if AB == True:
		F_zero = {
		'FUV':3631.00,
		'NUV':3631.00,
		'sloan_u':3631.000,
		'sloan_g':3631.000,
		'sloan_r':3631.000,
		'sloan_i':3631.000,
		'sloan_z':3631.000,
		'U':3631.000,
		'B':3631.000,
		'V':3631.000,
		'R':3631.000,
		'I':3631.000,
		'JVHS':3631.0,
		'HVHS':3631.0,
		'KVHS':3631.0,
		'JUK':3631.0,
		'HUK':3631.0,
		'KUK':3631.0,
		'J2MASS':3631.0,
		'H2MASS':3631.0,
		'Ks2MASS':3631.0,
		'W1':3631.0,
		'W2':3631.0,
		'W3':3631.0,
		'W4':3631.0,
		'Ch1Spies':3631.0,
		'Ch2Spies':3631.0,
		'IRAC1':3631.0,
		'IRAC2':3631.0,
		'IRAC3':3631.0,
		'IRAC4':3631.0,
		'MIPS1':3631.0,
		'AB':3631.0
	}

	else:
		F_zero = {
		'FUV':520.73,
		'NUV':788.55,
		'sloan_u':1628.72,
		'sloan_g':3971.19,
		'sloan_r':3142.02,
		'sloan_i':2571.22,
		'sloan_z':2227.03,
		'U':1790.000,
		'B':4063.00,
		'V':3636.000,
		'R':3064.000,
		'I':2416.000,
		'JVHS':1550.69,
		'HVHS':1027.33,
		'KVHS':669.56,
		'JUK':1553.01,
		'HUK':1034.19,
		'KUK':641.60,
		'J2MASS':1594.0,
		'H2MASS':1024.0,
		'Ks2MASS':666.80,
		'W1':309.54,
		'W2':171.79,
		'W3':31.67,
		'W4':8.36,
		'IRAC1':277.22,
		'IRAC2':179.04,
		'IRAC3':113.85,
		'IRAC4':62.0,
		'MIPS1':7.07,
		'MIPS2':0.77,
		'MIPS3':0.158
		}

	mag[mag<=0] = np.nan

	# if band == 'JVHS':
	# 	mag -= 0.916
	# elif band == 'HVHS':
	# 	mag -= 1.366
	# elif band == 'KVHS':
	# 	mag -= 1.827

	# if band == 'W1':
	# 	mag += 2.699
	# elif band == 'W2':
	# 	mag += 3.339
	# elif band == 'W3':
	# 	mag += 5.174
	# elif band == 'W4':
	# 	mag += 6.620
	# else:
	# 	mag = mag

	F = F_zero[band]*(10**(-1.0*mag/2.5))
	# print(F_zero[band])

	return F

def magerr_to_fluxerr(mag,mag_err,band,AB=True):
	mag = np.asarray(mag,dtype=float)
	# mag = mag[0]

	if AB == True:
		F_zero = {
		'FUV':3631.00,
		'NUV':3631.00,
		'sloan_u':3631.000,
		'sloan_g':3631.000,
		'sloan_r':3631.000,
		'sloan_i':3631.000,
		'sloan_z':3631.000,
		'U':3631.000,
		'B':3631.000,
		'V':3631.000,
		'R':3631.000,
		'I':3631.000,
		'JVHS':3631.0,
		'HVHS':3631.0,
		'KVHS':3631.0,
		'JUK':3631.0,
		'HUK':3631.0,
		'KUK':3631.0,
		'J2MASS':3631.0,
		'H2MASS':3631.0,
		'Ks2MASS':3631.0,
		'W1':3631.0,
		'W2':3631.0,
		'W3':3631.0,
		'W4':3631.0,
		'Ch1Spies':3631.0,
		'Ch2Spies':3631.0,
		'IRAC1':3631.0,
		'IRAC2':3631.0,
		'IRAC3':3631.0,
		'IRAC4':3631.0,
		'MIPS1':3631.0,
		'AB':3631.0
	}

	else:
		F_zero = {
		'FUV':520.73,
		'NUV':788.55,
		'sloan_u':1628.72,
		'sloan_g':3971.19,
		'sloan_r':3142.02,
		'sloan_i':2571.22,
		'sloan_z':2227.03,
		'U':3631.000,
		'B':4025.79,
		'V':3631.000,
		'R':3631.000,
		'I':3631.000,
		'JVHS':1550.69,
		'HVHS':1027.33,
		'KVHS':669.56,
		'JUK':1553.01,
		'HUK':1034.19,
		'KUK':641.60,
		'J2MASS':1594.0,
		'H2MASS':1024.0,
		'Ks2MASS':666.80,
		'W1':309.54,
		'W2':171.79,
		'W3':31.67,
		'W4':8.36,
		'IRAC1':277.22,
		'IRAC2':179.04,
		'IRAC3':113.85,
		'IRAC4':62.0,
		'MIPS1':7.07,
		'MIPS2':0.77,
		'MIPS3':0.158
		}

	mag[mag<=0] = np.nan

	F = F_zero[band]*(10**(-1.0*mag/2.5))
	F_err = F*abs(mag_err)

	return F_err


def flux_to_mag(F,band):

	F = np.asarray(F,dtype=float)

	# F_zero = {
	# 'FUV':520.7,
	# 'NUV':788.5,
	# 'sloan_u':1568.5,
	# 'sloan_g':3965.9,
	# 'sloan_r':3162.0,
	# 'sloan_i':2602.0,
	# 'sloan_z':2244.7,
	# 'JVHS':3631.0,
	# 'HVHS':3631.0,
	# 'KVHS':3631.0,
	# 'JUK':1556.8,
	# 'HUK':2920.0,
	# 'KUK':3510.0,
	# 'W1':309.540,
	# 'W2':171.787,
	# 'W3':31.674,
	# 'W4':8.363
	# }

	# F_zero = {
	# 'FUV':3631.00,
	# 'NUV':3631.00,
	# 'sloan_u':3631.000,
	# 'sloan_g':3631.000,
	# 'sloan_r':3631.000,
	# 'sloan_i':3631.000,
	# 'sloan_z':3564.727,
	# 'U':3631.000,
	# 'B':3631.000,
	# 'V':3631.000,
	# 'R':3631.000,
	# 'I':3631.000,
	# 'JVHS':3631.0,
	# 'HVHS':3631.0,
	# 'KVHS':3631.0,
	# 'JUK':1556.75,
	# 'HUK':1038.30,
	# 'KUK':644.07,
	# 'W1':309.540,
	# 'W2':171.787,
	# 'W3':31.674,
	# 'W4':8.363
	# }

	F_zero = {
	'FUV':3631.00,
	'NUV':3631.00,
	'sloan_u':3631.000,
	'sloan_g':3631.000,
	'sloan_r':3631.000,
	'sloan_i':3631.000,
	'sloan_z':3631.000,
	'U':3631.000,
	'B':3631.000,
	'V':3631.000,
	'R':3631.000,
	'I':3631.000,
	'JVHS':3631.0,
	'HVHS':3631.0,
	'KVHS':3631.0,
	'JUK':3631.0,
	'HUK':3631.0,
	'KUK':3631.0,
	'W1':3631.0,
	'W2':3631.0,
	'W3':3631.0,
	'W4':3631.0,
	'Ch1Spies':3631.0,
	'MIPS1':3631.0
	}



	# mag[mag<0] = np.nan

	# if band == 'JVHS':
	# 	mag -= 0.916
	# elif band == 'HVHS':
	# 	mag -= 1.366
	# elif band == 'KVHS':
	# 	mag -= 1.827

	# if band == 'W1':
	# 	mag += 2.699
	# elif band == 'W2':
	# 	mag += 3.339
	# elif band == 'W3':
	# 	mag += 5.174
	# elif band == 'W4':
	# 	mag += 6.620
	# else:
	# 	mag = mag

	mag = -2.5*np.log10(F/F_zero[band])
	# print(F_zero[band])

	return mag
