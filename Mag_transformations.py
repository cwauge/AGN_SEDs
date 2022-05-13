import numpy as np
def sloan_JC(out,u,g,r):

	u[u < 0] = np.nan
	g[g < 0] = np.nan
	r[r < 0] = np.nan

	B = g + 0.17*(u - g) + 0.11
	V = g - 0.52*(g - r) - 0.02

	if out == 'B':
		return B
	elif out == 'V':
		return V


def MIR_trans(mir1,mir2,w1,w2,out1,out2):

	if np.isnan(mir1) == True:
		return np.nan, np.nan
	
	elif np.isnan(mir2) == True:
		return np.nan, np.nan
	
	else:
		y = np.array([mir1,mir2])
		x = np.array([w1,w2])

		z = np.polyfit(x, y, 1)
		p = np.poly1d(z)

		return p(out1), p(out2)
