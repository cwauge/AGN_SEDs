import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import math
from pylab import *
from matplotlib.collections import LineCollection
from astropy.cosmology import FlatLambdaCDM
from filters import Filters
from astropy.io import ascii
from astropy.io import fits
from match import match
# from SED_v7 import Flux_to_Lum
import matplotlib.patheffects as pe
import Lit_functions

class Plotter_Letter2():
	
	def __init__(self,ID,z,wavelength,flux,Lx=None,Lbol=None,spec_type=None):
		self.ID = np.asarray(ID)
		self.z = np.asarray(z)
		self.wavelength_array = np.asarray(wavelength)
		self.flux_array = np.asarray(flux)
		self.Lx = np.asarray(Lx)
		self.Lbol = np.asarray(Lbol)
		self.spec_type = np.asarray(spec_type)


	def multilines(self,xs,ys,cs,ax=None,**kwargs):
		ax = plt.gca() if ax is None else ax # find axes
		segments = [np.column_stack([x,y]) for x, y in zip(xs,ys)] # Create LineCollection
		lc = LineCollection(segments, **kwargs)
		lc.set_array(np.asarray(cs)) # set coloring of line segments
		ax.add_collection(lc) # add lines to axes and rescale
		ax.autoscale()
		return lc

	def make_median_bins(self, array1, array2, bins):
		j = 0
		sort = array1.argsort()
		array1 = array1[sort]
		array2 = array2[sort]
    
		x_out_med_list = []
		y_out_med_list = []
		x_out_lims_list = []
		y_out_lims_list = []
		x_fill_lists = [[] for i in range(bins)]
		y_fill_lists = [[] for i in range(bins)]
    
		bin_size = int(len(array1)/bins)
		print('bin size: ',bin_size)
		for i in range(bins):
			try:
				while len(x_fill_lists[i]) < bin_size:
					x_fill_lists[i].append(array1[j])
					y_fill_lists[i].append(array2[j])
					j += 1
				x_out_med_list.append(np.nanmedian(x_fill_lists[i]))
				y_out_med_list.append(np.nanmedian(y_fill_lists[i]))
 
				x_out_lims_list.append([min(x_fill_lists[i]),max(x_fill_lists[i])])
				y_out_lims_list.append([min(y_fill_lists[i]),max(y_fill_lists[i])])
			except IndexError:
				continue

		return x_out_med_list, y_out_med_list, x_out_lims_list, y_out_lims_list
	

	def Lx_Scatter_Comp(self, savestring, X, Y, Norm, Median, Lx, L, f1, f2, f3, f4, F1, field, spec_z, uv_slope, mir_slope1, mir_slope2, up_check,ulirg_Lx=None,ulirg_Flux=None,ulirg_F1=None):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = (uv_slope < -0.3)&(mir_slope1 >= -0.2)
		B2 = (uv_slope >= -0.3)&(uv_slope <= 0.2)&(mir_slope1 >= -0.2)
		B3 = (uv_slope > 0.2)&(mir_slope1 >= -0.2)
		B4 = (uv_slope >= -0.3)&(mir_slope1 < -0.2)&(mir_slope2 > 0.0)
		B5 = (uv_slope >= -0.3)&(mir_slope1 < -0.2)&(mir_slope2 <= 0.0)

		zlim1 = 0.0
		zlim2 = 0.6
		zlim3 = 0.9
		zlim4 = 1.2

		zbins1 = (spec_z >= zlim1) & (spec_z <= zlim2)
		zbins2 = (spec_z >= zlim2) & (spec_z <= zlim3)
		zbins3 = (spec_z >= zlim3) & (spec_z <= zlim4)

		if Y == 'UV':
			y = f1
			y_var = r'L (0.25$\mu$m)'

		elif Y == 'MIR10':
			y = f4
			y_var = r'L (10$\mu$m)'

		elif Y == 'MIR6':
			y = f2
			y_var = r'L (6$\mu$m)'

		elif Y == 'FIR':
			y = f3
			y_var = r'L (100$\mu$m)'
		elif Y == 'Lbol':
			y = L
			y_var = r'L$_{\mathrm{bol}}$'
		else:
			print('Specify Y variable')
			return 


		if X == 'Lx':
			x = Lx
			x_var = r'L$_{\mathrm{X}}$'

		elif X == 'Lbol':
			x == L
			x_var = r'L$_{\mathrm{bol}}$'
		else:
			print('Specify X variable')
			return


		if Norm == 'None':
			if Y != 'Lbol':
				y_s = np.asarray([10**i for i in y])
				y = np.log10(y_s*F1)

				ylim1 = 42.5
				ylim2 = 46.5
				xticks = [43,44,45,46]
				yticks = [43,44,45,46]
			else:
				ylim1 = 43
				ylim2 = 48
				xticks = [43,44,45,46]
				yticks = [44,45,46,47]	

			if any(ulirg_Flux) != None:
				u_f = np.asarray([10**i for i in ulirg_Flux])
				ulirg_Flux = np.log10(u_f*ulirg_F1)

			xlim1 = 42.5
			xlim2 = 46.5

			xlabel = r'log '+x_var+' [erg/s]'
			ylabel = r'log '+y_var+' [erg/s]'

			
		elif Norm == 'Both':
			x = np.asarray([10**i for i in x])
			x = np.log10(x/F1)

			ylim1 = -2
			ylim2 = 2

			xlim1 = -3
			xlim2 = 1

			xlabel = r'log '+x_var+r'/L (1$\mu$m)'
			ylabel = r'log '+y_var+r'/L (1$\mu$m)'
			xticks = [-2,-1,0,1,2]
			yticks = [-2,-1,0,1,2]

			if any(ulirg_Lx) != None:
				u_f = np.asarray([10**i for i in ulirg_Lx])
				ulirg_Lx = np.log10(u_f/ulirg_F1)

		elif Norm == 'Y':
			ylim1 = -2
			ylim2 = 2

			xlim1 = 42.5
			xlim2 = 46.5

			xlabel = r'log '+x_var+r'[erg/s]'
			ylabel = r'log '+y_var+r'/L (1$\mu$m)'
			xticks = [43,44,45,46]
			yticks = [-2,-1,0,1,2]

		else:
			print('Specify if each variable is normalized')
			return 

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		x_11 = x[zbins1][B1[zbins1]]
		y_11 = y[zbins1][B1[zbins1]]
		up_check_11 = up_check[zbins1][B1[zbins1]]
		x_12 = x[zbins1][B2[zbins1]]
		y_12 = y[zbins1][B2[zbins1]]
		up_check_12 = up_check[zbins1][B2[zbins1]]
		x_13 = x[zbins1][B3[zbins1]]
		y_13 = y[zbins1][B3[zbins1]]
		up_check_13 = up_check[zbins1][B3[zbins1]]
		x_14 = x[zbins1][B4[zbins1]]
		y_14 = y[zbins1][B4[zbins1]]
		up_check_14 = up_check[zbins1][B4[zbins1]]
		x_15 = x[zbins1][B5[zbins1]]
		y_15 = y[zbins1][B5[zbins1]]
		up_check_15 = up_check[zbins1][B5[zbins1]]

		x_21 = x[zbins2][B1[zbins2]]
		y_21 = y[zbins2][B1[zbins2]]
		up_check_21 = up_check[zbins2][B1[zbins2]]
		x_22 = x[zbins2][B2[zbins2]]
		y_22 = y[zbins2][B2[zbins2]]
		up_check_22 = up_check[zbins2][B2[zbins2]]
		x_23 = x[zbins2][B3[zbins2]]
		y_23 = y[zbins2][B3[zbins2]]
		up_check_23 = up_check[zbins2][B3[zbins2]]
		x_24 = x[zbins2][B4[zbins2]]
		y_24 = y[zbins2][B4[zbins2]]
		up_check_24 = up_check[zbins2][B4[zbins2]]
		x_25 = x[zbins2][B5[zbins2]]
		y_25 = y[zbins2][B5[zbins2]]
		up_check_25 = up_check[zbins2][B5[zbins2]]

		x_31 = x[zbins3][B1[zbins3]]
		y_31 = y[zbins3][B1[zbins3]]
		up_check_31 = up_check[zbins3][B1[zbins3]]
		x_32 = x[zbins3][B2[zbins3]]
		y_32 = y[zbins3][B2[zbins3]]
		up_check_32 = up_check[zbins3][B2[zbins3]]
		x_33 = x[zbins3][B3[zbins3]]
		y_33 = y[zbins3][B3[zbins3]]
		up_check_33 = up_check[zbins3][B3[zbins3]]
		x_34 = x[zbins3][B4[zbins3]]
		y_34 = y[zbins3][B4[zbins3]]
		up_check_34 = up_check[zbins3][B4[zbins3]]
		x_35 = x[zbins3][B5[zbins3]]
		y_35 = y[zbins3][B5[zbins3]]
		up_check_35 = up_check[zbins3][B5[zbins3]]

		if Median == 'Bins':
			c1m = c1
			c2m = c2
			c3m = c3
			c4m = c4
			c5m = c5

			# Median
			x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
			x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
			x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
			x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
			x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

			x21_med, y21_med = np.nanmedian(x_21), np.nanmedian(y_21)
			x22_med, y22_med = np.nanmedian(x_22), np.nanmedian(y_22)
			x23_med, y23_med = np.nanmedian(x_23), np.nanmedian(y_23)
			x24_med, y24_med = np.nanmedian(x_24), np.nanmedian(y_24)
			x25_med, y25_med = np.nanmedian(x_25), np.nanmedian(y_25)

			x31_med, y31_med = np.nanmedian(x_31), np.nanmedian(y_31)
			x32_med, y32_med = np.nanmedian(x_32), np.nanmedian(y_32)
			x33_med, y33_med = np.nanmedian(x_33), np.nanmedian(y_33)
			x34_med, y34_med = np.nanmedian(x_34), np.nanmedian(y_34)
			x35_med, y35_med = np.nanmedian(x_35), np.nanmedian(y_35)

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med],dtype=float), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med],dtype=float)
			x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med],dtype=float), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med],dtype=float)
			x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med],dtype=float), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med],dtype=float)


			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
			x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
			x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
			x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
			x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

			x21_25per, y21_25per = np.nanpercentile(x_21, 25), np.nanpercentile(y_21, 25)
			x22_25per, y22_25per = np.nanpercentile(x_22, 25), np.nanpercentile(y_22, 25)
			x23_25per, y23_25per = np.nanpercentile(x_23, 25), np.nanpercentile(y_23, 25)
			x24_25per, y24_25per = np.nanpercentile(x_24, 25), np.nanpercentile(y_24, 25)
			x25_25per, y25_25per = np.nanpercentile(x_25, 25), np.nanpercentile(y_25, 25)

			x31_25per, y31_25per = np.nanpercentile(x_31, 25), np.nanpercentile(y_31, 25)
			x32_25per, y32_25per = np.nanpercentile(x_32, 25), np.nanpercentile(y_32, 25)
			x33_25per, y33_25per = np.nanpercentile(x_33, 25), np.nanpercentile(y_33, 25)
			x34_25per, y34_25per = np.nanpercentile(x_34, 25), np.nanpercentile(y_34, 25)
			x35_25per, y35_25per = np.nanpercentile(x_35, 25), np.nanpercentile(y_35, 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per],dtype=float), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per],dtype=float)
			x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per],dtype=float), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per],dtype=float)
			x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per],dtype=float), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per],dtype=float)

			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
			x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
			x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
			x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
			x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

			x21_75per, y21_75per = np.nanpercentile(x_21, 75), np.nanpercentile(y_21, 75)
			x22_75per, y22_75per = np.nanpercentile(x_22, 75), np.nanpercentile(y_22, 75)
			x23_75per, y23_75per = np.nanpercentile(x_23, 75), np.nanpercentile(y_23, 75)
			x24_75per, y24_75per = np.nanpercentile(x_24, 75), np.nanpercentile(y_24, 75)
			x25_75per, y25_75per = np.nanpercentile(x_25, 75), np.nanpercentile(y_25, 75)

			x31_75per, y31_75per = np.nanpercentile(x_31, 75), np.nanpercentile(y_31, 75)
			x32_75per, y32_75per = np.nanpercentile(x_32, 75), np.nanpercentile(y_32, 75)
			x33_75per, y33_75per = np.nanpercentile(x_33, 75), np.nanpercentile(y_33, 75)
			x34_75per, y34_75per = np.nanpercentile(x_34, 75), np.nanpercentile(y_34, 75)
			x35_75per, y35_75per = np.nanpercentile(x_35, 75), np.nanpercentile(y_35, 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per],dtype=float) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per],dtype=float) - y1_med
			x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per],dtype=float) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per],dtype=float) - y2_med
			x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per],dtype=float) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per],dtype=float) - y3_med

		elif Median == 'X-axis':

			c1m = 'gray'
			c2m = 'gray'
			c3m = 'gray'
			c4m = 'gray'
			c5m = 'gray'

			b1 = (Lx > 43)&(Lx < 43.5)
			b2 = (Lx > 43.5)&(Lx < 44)
			b3 = (Lx > 44)&(Lx < 44.5)
			b4 = (Lx > 44.5)&(Lx < 45)
			b5 = (Lx > 45)

			# Median
			x11_med, y11_med = np.nanmedian(x[zbins1][b1[zbins1]]), np.nanmedian(y[zbins1][b1[zbins1]])
			x12_med, y12_med = np.nanmedian(x[zbins1][b2[zbins1]]), np.nanmedian(y[zbins1][b2[zbins1]])
			x13_med, y13_med = np.nanmedian(x[zbins1][b3[zbins1]]), np.nanmedian(y[zbins1][b3[zbins1]])
			x14_med, y14_med = np.nanmedian(x[zbins1][b4[zbins1]]), np.nanmedian(y[zbins1][b4[zbins1]])
			x15_med, y15_med = np.nanmedian(x[zbins1][b5[zbins1]]), np.nanmedian(y[zbins1][b5[zbins1]])

			x21_med, y21_med = np.nanmedian(x[zbins2][b1[zbins2]]), np.nanmedian(y[zbins2][b1[zbins2]])
			x22_med, y22_med = np.nanmedian(x[zbins2][b2[zbins2]]), np.nanmedian(y[zbins2][b2[zbins2]])
			x23_med, y23_med = np.nanmedian(x[zbins2][b3[zbins2]]), np.nanmedian(y[zbins2][b3[zbins2]])
			x24_med, y24_med = np.nanmedian(x[zbins2][b4[zbins2]]), np.nanmedian(y[zbins2][b4[zbins2]])
			x25_med, y25_med = np.nanmedian(x[zbins2][b5[zbins2]]), np.nanmedian(y[zbins2][b5[zbins2]])

			x31_med, y31_med = np.nanmedian(x[zbins3][b1[zbins3]]), np.nanmedian(y[zbins3][b1[zbins3]])
			x32_med, y32_med = np.nanmedian(x[zbins3][b2[zbins3]]), np.nanmedian(y[zbins3][b2[zbins3]])
			x33_med, y33_med = np.nanmedian(x[zbins3][b3[zbins3]]), np.nanmedian(y[zbins3][b3[zbins3]])
			x34_med, y34_med = np.nanmedian(x[zbins3][b4[zbins3]]), np.nanmedian(y[zbins3][b4[zbins3]])
			x35_med, y35_med = np.nanmedian(x[zbins3][b5[zbins3]]), np.nanmedian(y[zbins3][b5[zbins3]])

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med],dtype=float), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med],dtype=float)
			x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med],dtype=float), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med],dtype=float)
			x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med],dtype=float), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med],dtype=float)

			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x[zbins1][b1[zbins1]], 25), np.nanpercentile(y[zbins1][b1[zbins1]], 25)
			x12_25per, y12_25per = np.nanpercentile(x[zbins1][b2[zbins1]], 25), np.nanpercentile(y[zbins1][b2[zbins1]], 25)
			x13_25per, y13_25per = np.nanpercentile(x[zbins1][b3[zbins1]], 25), np.nanpercentile(y[zbins1][b3[zbins1]], 25)
			x14_25per, y14_25per = np.nanpercentile(x[zbins1][b4[zbins1]], 25), np.nanpercentile(y[zbins1][b4[zbins1]], 25)
			x15_25per, y15_25per = np.nanpercentile(x[zbins1][b5[zbins1]], 25), np.nanpercentile(y[zbins1][b5[zbins1]], 25)

			x21_25per, y21_25per = np.nanpercentile(x[zbins2][b1[zbins2]], 25), np.nanpercentile(y[zbins2][b1[zbins2]], 25)
			x22_25per, y22_25per = np.nanpercentile(x[zbins2][b2[zbins2]], 25), np.nanpercentile(y[zbins2][b2[zbins2]], 25)
			x23_25per, y23_25per = np.nanpercentile(x[zbins2][b3[zbins2]], 25), np.nanpercentile(y[zbins2][b3[zbins2]], 25)
			x24_25per, y24_25per = np.nanpercentile(x[zbins2][b4[zbins2]], 25), np.nanpercentile(y[zbins2][b4[zbins2]], 25)
			x25_25per, y25_25per = np.nanpercentile(x[zbins2][b5[zbins2]], 25), np.nanpercentile(y[zbins2][b5[zbins2]], 25)

			x31_25per, y31_25per = np.nanpercentile(x[zbins3][b1[zbins3]], 25), np.nanpercentile(y[zbins3][b1[zbins3]], 25)
			x32_25per, y32_25per = np.nanpercentile(x[zbins3][b2[zbins3]], 25), np.nanpercentile(y[zbins3][b2[zbins3]], 25)
			x33_25per, y33_25per = np.nanpercentile(x[zbins3][b3[zbins3]], 25), np.nanpercentile(y[zbins3][b3[zbins3]], 25)
			x34_25per, y34_25per = np.nanpercentile(x[zbins3][b4[zbins3]], 25), np.nanpercentile(y[zbins3][b4[zbins3]], 25)
			x35_25per, y35_25per = np.nanpercentile(x[zbins3][b5[zbins3]], 25), np.nanpercentile(y[zbins3][b5[zbins3]], 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per],dtype=float), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per],dtype=float)
			x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per],dtype=float), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per],dtype=float)
			x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per],dtype=float), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per],dtype=float)

			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x[zbins1][b1[zbins1]], 75), np.nanpercentile(y[zbins1][b1[zbins1]], 75)
			x12_75per, y12_75per = np.nanpercentile(x[zbins1][b2[zbins1]], 75), np.nanpercentile(y[zbins1][b2[zbins1]], 75)
			x13_75per, y13_75per = np.nanpercentile(x[zbins1][b3[zbins1]], 75), np.nanpercentile(y[zbins1][b3[zbins1]], 75)
			x14_75per, y14_75per = np.nanpercentile(x[zbins1][b4[zbins1]], 75), np.nanpercentile(y[zbins1][b4[zbins1]], 75)
			x15_75per, y15_75per = np.nanpercentile(x[zbins1][b5[zbins1]], 75), np.nanpercentile(y[zbins1][b5[zbins1]], 75)

			x21_75per, y21_75per = np.nanpercentile(x[zbins2][b1[zbins2]], 75), np.nanpercentile(y[zbins2][b1[zbins2]], 75)
			x22_75per, y22_75per = np.nanpercentile(x[zbins2][b2[zbins2]], 75), np.nanpercentile(y[zbins2][b2[zbins2]], 75)
			x23_75per, y23_75per = np.nanpercentile(x[zbins2][b3[zbins2]], 75), np.nanpercentile(y[zbins2][b3[zbins2]], 75)
			x24_75per, y24_75per = np.nanpercentile(x[zbins2][b4[zbins2]], 75), np.nanpercentile(y[zbins2][b4[zbins2]], 75)
			x25_75per, y25_75per = np.nanpercentile(x[zbins2][b5[zbins2]], 75), np.nanpercentile(y[zbins2][b5[zbins2]], 75)

			x31_75per, y31_75per = np.nanpercentile(x[zbins3][b1[zbins3]], 75), np.nanpercentile(y[zbins3][b1[zbins3]], 75)
			x32_75per, y32_75per = np.nanpercentile(x[zbins3][b2[zbins3]], 75), np.nanpercentile(y[zbins3][b2[zbins3]], 75)
			x33_75per, y33_75per = np.nanpercentile(x[zbins3][b3[zbins3]], 75), np.nanpercentile(y[zbins3][b3[zbins3]], 75)
			x34_75per, y34_75per = np.nanpercentile(x[zbins3][b4[zbins3]], 75), np.nanpercentile(y[zbins3][b4[zbins3]], 75)
			x35_75per, y35_75per = np.nanpercentile(x[zbins3][b5[zbins3]], 75), np.nanpercentile(y[zbins3][b5[zbins3]], 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per],dtype=float) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per],dtype=float) - y1_med
			x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per],dtype=float) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per],dtype=float) - y2_med
			x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per],dtype=float) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per],dtype=float) - y3_med

		# elif Median == 'Lx':
		# 	x_1_med, y_1_med, x_1_lim, y_1_lim = self.make_median_bins(x[zbins1],y[zbins1],5)
		# 	x_2_med, y_2_med, x_2_lim, y_2_lim = self.make_median_bins(x[zbins2],y[zbins2],5)
		# 	x_3_med, y_3_med, x_3_lim, y_3_lim = self.make_median_bins(x[zbins3],y[zbins3],5)


		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		fig = plt.figure(figsize=(18, 7))
		ax1 = plt.subplot(131, aspect='equal', adjustable='box')
		if Y == 'FIR' or Y == 'Lbol':
			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=3, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=3, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=3, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=3, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=3, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		else:
			ax1.scatter(x_11, y_11, color=c1, marker='P', rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='P', rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='P', rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='P', rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='P', rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		if any(ulirg_Flux) != None:
			ax1.scatter(ulirg_Lx, ulirg_Flux, color='k', marker='s', s=50, rasterized=True,label='ULIRGS')


		ax1.set_xlim(xlim1,xlim2)
		ax1.set_ylim(ylim1,ylim2)
		ax1.set_ylabel(ylabel)
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		if Norm == 'None' or Norm == 'Y':
			secax1 = ax1.secondary_xaxis('top',functions=(solar, ergs))
			secax1.set_xlabel(r' ')
		ax1.legend(fontsize=14)
		ax1.set_title(f'{zlim1} < z < {zlim2}')
		ax1.grid()


		ax2 = plt.subplot(132, aspect='equal', adjustable='box')
		if Y == 'FIR' or Y == 'Lbol':
			ax2.scatter(x_21[up_check_21 == 1], y_21[up_check_21 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_22[up_check_22 == 1], y_22[up_check_22 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_23[up_check_23 == 1], y_23[up_check_23 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_24[up_check_24 == 1], y_24[up_check_24 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_25[up_check_25 == 1], y_25[up_check_25 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax2.scatter(x_21[up_check_21 == 1], y_21[up_check_21 == 1], marker=3, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_22[up_check_22 == 1], y_22[up_check_22 == 1], marker=3, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_23[up_check_23 == 1], y_23[up_check_23 == 1], marker=3, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_24[up_check_24 == 1], y_24[up_check_24 == 1], marker=3, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_25[up_check_25 == 1], y_25[up_check_25 == 1], marker=3, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax2.scatter(x_21[up_check_21 == 0], y_21[up_check_21 == 0], color=c1, marker='P', rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax2.scatter(x_22[up_check_22 == 0], y_22[up_check_22 == 0], color=c2, marker='P', rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax2.scatter(x_23[up_check_23 == 0], y_23[up_check_23 == 0], color=c3, marker='P', rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax2.scatter(x_24[up_check_24 == 0], y_24[up_check_24 == 0], color=c4, marker='P', rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax2.scatter(x_25[up_check_25 == 0], y_25[up_check_25 == 0], color=c5, marker='P', rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		else:
			ax2.scatter(x_21, y_21, color=c1, marker='P', rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax2.scatter(x_22, y_22, color=c2, marker='P', rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax2.scatter(x_23, y_23, color=c3, marker='P', rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax2.scatter(x_24, y_24, color=c4, marker='P', rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax2.scatter(x_25, y_25, color=c5, marker='P', rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax2.errorbar(x2_med, y2_med, xerr=[x2_err_min, x2_err_max], yerr=[y2_err_min, y2_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax2.scatter(x21_med, y21_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x22_med, y22_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x23_med, y23_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x24_med, y24_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x25_med, y25_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)


		ax2.set_xlim(xlim1, xlim2)
		ax2.set_ylim(ylim1, ylim2)
		ax2.set_xlabel(xlabel)
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_yticklabels([])
		if Norm == 'None' or Norm == 'Y':
			secax2 = ax2.secondary_xaxis('top', functions=(solar, ergs))
			secax2.set_xlabel(r'log '+x_var+r' [L$_{\odot}$]')
		ax2.set_title(f'{zlim2} < z < {zlim3}')
		ax2.grid()

		ax3 = plt.subplot(133, aspect='equal', adjustable='box')
		if Y == 'FIR' or Y == 'Lbol':
			ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], marker=3, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], marker=3, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], marker=3, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], marker=3, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], marker=3, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax3.scatter(x_31[up_check_31 == 0], y_31[up_check_31 == 0], color=c1, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 0], y_32[up_check_32 == 0], color=c2, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 0], y_33[up_check_33 == 0], color=c3, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 0], y_34[up_check_34 == 0], color=c4, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 0], y_35[up_check_35 == 0], color=c5, marker='P', rasterized=True, alpha=0.8,zorder=0)

		else:
			ax3.scatter(x_31, y_31, color=c1, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32, y_32, color=c2, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33, y_33, color=c3, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34, y_34, color=c4, marker='P', rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35, y_35, color=c5, marker='P', rasterized=True, alpha=0.8,zorder=0)

		if Median != 'None':
			ax3.errorbar(x3_med, y3_med, xerr=[x3_err_min, x3_err_max], yerr=[y3_err_min, y3_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax3.scatter(x31_med, y31_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x32_med, y32_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x33_med, y33_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x34_med, y34_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x35_med, y35_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)


		ax3.set_xlim(xlim1, xlim2)
		ax3.set_ylim(ylim1, ylim2)
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		ax3.set_yticklabels([])
		if Norm == 'None':
			secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(r' ')
			secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
			secax3.set_ylabel(r'log '+y_var+r' [L$_{\odot}$]')
		elif Norm == 'Y':
			secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(r' ')
		ax3.set_title(f'{zlim3} < z < {zlim4}')
		ax3.grid()

		plt.tight_layout()
		plt.savefig(f'/Users/connor_auge/Desktop/New_plots3/{savestring}.pdf')
		plt.show()


	def Emission_Scatter_Comp(self, savestring, X, Y, Norm, Median, Lx, L, f1, f2, f3, f4, F1, field, spec_z, uv_slope, mir_slope1, mir_slope2, up_check):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = (uv_slope < -0.3) & (mir_slope1 >= -0.2)
		B2 = (uv_slope >= -0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2)
		B3 = (uv_slope > 0.2) & (mir_slope1 >= -0.2)
		B4 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0)
		B5 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0)

		zlim1 = 0.0
		zlim2 = 0.6
		zlim3 = 0.9
		zlim4 = 1.2

		zbins1 = (spec_z >= zlim1) & (spec_z <= zlim2)
		zbins2 = (spec_z >= zlim2) & (spec_z <= zlim3)
		zbins3 = (spec_z >= zlim3) & (spec_z <= zlim4)

		if X == 'UV':
			x = f1
			x_var = r'0.25$\mu$m'

		elif X == 'MIR10':
			x = f4
			x_var = r'10$\mu$m'

		elif X == 'MIR6':
			x = f2
			x_var = r'6$\mu$m'

		elif X == 'FIR':
			x = f3
			x_var = r'100$\mu$m'
		else:
			print('Specify X variable')
			return

		if Y == 'UV':
			y = f1
			y_var = r'0.25$\mu$m'

		elif Y == 'MIR10':
			y = f4
			y_var = r'10$\mu$m'

		elif Y == 'MIR6':
			y = f2
			y_var = r'6$\mu$m'

		elif Y == 'FIR':
			y = f3
			y_var = r'100$\mu$m'
		else:
			print('Specify Y variable')
			return

		if Norm == 'None':
			y_s = np.asarray([10**i for i in y])
			x_s = np.asarray([10**i for i in x])
			y = np.log10(y_s*F1)
			x = np.log10(x_s*F1)

			xlim1 = 42
			xlim2 = 47
			ylim1 = 42
			ylim2 = 47

			xlabel = r'log L ('+x_var+') [erg/s]'
			ylabel = r'log L ('+y_var+') [erg/s]'
			xticks = [42, 43, 44, 45, 46, 47]
			yticks = [42, 43, 44, 45, 46, 47]

		elif Norm == 'Both':

			lx = np.asarray([10**i for i in Lx])
			l = np.asarray([10**i for i in L])

			Lx = np.log10(lx/F1)
			L = np.log10(l/F1)

			xlim1 = -2
			xlim2 = 2
			ylim1 = -2
			ylim2 = 2

			xlabel = r'log L ('+x_var+r')/L (1$\mu$m)'
			ylabel = r'log L ('+y_var+r')/L (1$\mu$m)'
			xticks = [-2, -1, 0, 1, 2]
			yticks = [-2, -1, 0, 1, 2]

		else:
			print('Specify if each variable is normalized')
			return

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		x_11 = x[zbins1][B1[zbins1]]
		y_11 = y[zbins1][B1[zbins1]]
		up_check_11 = up_check[zbins1][B1[zbins1]]
		x_12 = x[zbins1][B2[zbins1]]
		y_12 = y[zbins1][B2[zbins1]]
		up_check_12 = up_check[zbins1][B2[zbins1]]
		x_13 = x[zbins1][B3[zbins1]]
		y_13 = y[zbins1][B3[zbins1]]
		up_check_13 = up_check[zbins1][B3[zbins1]]
		x_14 = x[zbins1][B4[zbins1]]
		y_14 = y[zbins1][B4[zbins1]]
		up_check_14 = up_check[zbins1][B4[zbins1]]
		x_15 = x[zbins1][B5[zbins1]]
		y_15 = y[zbins1][B5[zbins1]]
		up_check_15 = up_check[zbins1][B5[zbins1]]

		x_21 = x[zbins2][B1[zbins2]]
		y_21 = y[zbins2][B1[zbins2]]
		up_check_21 = up_check[zbins2][B1[zbins2]]
		x_22 = x[zbins2][B2[zbins2]]
		y_22 = y[zbins2][B2[zbins2]]
		up_check_22 = up_check[zbins2][B2[zbins2]]
		x_23 = x[zbins2][B3[zbins2]]
		y_23 = y[zbins2][B3[zbins2]]
		up_check_23 = up_check[zbins2][B3[zbins2]]
		x_24 = x[zbins2][B4[zbins2]]
		y_24 = y[zbins2][B4[zbins2]]
		up_check_24 = up_check[zbins2][B4[zbins2]]
		x_25 = x[zbins2][B5[zbins2]]
		y_25 = y[zbins2][B5[zbins2]]
		up_check_25 = up_check[zbins2][B5[zbins2]]

		x_31 = x[zbins3][B1[zbins3]]
		y_31 = y[zbins3][B1[zbins3]]
		up_check_31 = up_check[zbins3][B1[zbins3]]
		x_32 = x[zbins3][B2[zbins3]]
		y_32 = y[zbins3][B2[zbins3]]
		up_check_32 = up_check[zbins3][B2[zbins3]]
		x_33 = x[zbins3][B3[zbins3]]
		y_33 = y[zbins3][B3[zbins3]]
		up_check_33 = up_check[zbins3][B3[zbins3]]
		x_34 = x[zbins3][B4[zbins3]]
		y_34 = y[zbins3][B4[zbins3]]
		up_check_34 = up_check[zbins3][B4[zbins3]]
		x_35 = x[zbins3][B5[zbins3]]
		y_35 = y[zbins3][B5[zbins3]]
		up_check_35 = up_check[zbins3][B5[zbins3]]

		if Median == 'Bins':
			c1m = c1
			c2m = c2
			c3m = c3
			c4m = c4
			c5m = c5


			# Median
			x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
			x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
			x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
			x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
			x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

			x21_med, y21_med = np.nanmedian(x_21), np.nanmedian(y_21)
			x22_med, y22_med = np.nanmedian(x_22), np.nanmedian(y_22)
			x23_med, y23_med = np.nanmedian(x_23), np.nanmedian(y_23)
			x24_med, y24_med = np.nanmedian(x_24), np.nanmedian(y_24)
			x25_med, y25_med = np.nanmedian(x_25), np.nanmedian(y_25)

			x31_med, y31_med = np.nanmedian(x_31), np.nanmedian(y_31)
			x32_med, y32_med = np.nanmedian(x_32), np.nanmedian(y_32)
			x33_med, y33_med = np.nanmedian(x_33), np.nanmedian(y_33)
			x34_med, y34_med = np.nanmedian(x_34), np.nanmedian(y_34)
			x35_med, y35_med = np.nanmedian(x_35), np.nanmedian(y_35)

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])
			x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med]), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med])
			x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med]), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med])


			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
			x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
			x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
			x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
			x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

			x21_25per, y21_25per = np.nanpercentile(x_21, 25), np.nanpercentile(y_21, 25)
			x22_25per, y22_25per = np.nanpercentile(x_22, 25), np.nanpercentile(y_22, 25)
			x23_25per, y23_25per = np.nanpercentile(x_23, 25), np.nanpercentile(y_23, 25)
			x24_25per, y24_25per = np.nanpercentile(x_24, 25), np.nanpercentile(y_24, 25)
			x25_25per, y25_25per = np.nanpercentile(x_25, 25), np.nanpercentile(y_25, 25)

			x31_25per, y31_25per = np.nanpercentile(x_31, 25), np.nanpercentile(y_31, 25)
			x32_25per, y32_25per = np.nanpercentile(x_32, 25), np.nanpercentile(y_32, 25)
			x33_25per, y33_25per = np.nanpercentile(x_33, 25), np.nanpercentile(y_33, 25)
			x34_25per, y34_25per = np.nanpercentile(x_34, 25), np.nanpercentile(y_34, 25)
			x35_25per, y35_25per = np.nanpercentile(x_35, 25), np.nanpercentile(y_35, 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
			x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per]), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per])
			x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per]), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per])

			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
			x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
			x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
			x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
			x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

			x21_75per, y21_75per = np.nanpercentile(x_21, 75), np.nanpercentile(y_21, 75)
			x22_75per, y22_75per = np.nanpercentile(x_22, 75), np.nanpercentile(y_22, 75)
			x23_75per, y23_75per = np.nanpercentile(x_23, 75), np.nanpercentile(y_23, 75)
			x24_75per, y24_75per = np.nanpercentile(x_24, 75), np.nanpercentile(y_24, 75)
			x25_75per, y25_75per = np.nanpercentile(x_25, 75), np.nanpercentile(y_25, 75)

			x31_75per, y31_75per = np.nanpercentile(x_31, 75), np.nanpercentile(y_31, 75)
			x32_75per, y32_75per = np.nanpercentile(x_32, 75), np.nanpercentile(y_32, 75)
			x33_75per, y33_75per = np.nanpercentile(x_33, 75), np.nanpercentile(y_33, 75)
			x34_75per, y34_75per = np.nanpercentile(x_34, 75), np.nanpercentile(y_34, 75)
			x35_75per, y35_75per = np.nanpercentile(x_35, 75), np.nanpercentile(y_35, 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
			x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per]) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y2_med
			x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per]) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y3_med



		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		fig = plt.figure(figsize=(18, 7))
		ax1 = plt.subplot(131, aspect='equal', adjustable='box')
		if X == 'FIR':
			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=0, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=0, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=0, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=0, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=0, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		else:
			ax1.scatter(x_11, y_11, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5', zorder=0)

		if Median != 'None':
			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)


		ax1.set_xlim(xlim1,xlim2)
		ax1.set_ylim(ylim1,ylim2)
		ax1.set_ylabel(ylabel)
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		if Norm == 'None' or Norm == 'Y':
			secax1 = ax1.secondary_xaxis('top',functions=(solar, ergs))
			secax1.set_xlabel(r' ')
		ax1.legend(fontsize=14)
		ax1.set_title(f'{zlim1} < z < {zlim2}')
		ax1.grid()

		ax2 = plt.subplot(132, aspect='equal', adjustable='box')
		if X == 'FIR':
			ax2.scatter(x_21[up_check_21 == 1], y_21[up_check_21 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_22[up_check_22 == 1], y_22[up_check_22 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_23[up_check_23 == 1], y_23[up_check_23 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_24[up_check_24 == 1], y_24[up_check_24 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_25[up_check_25 == 1], y_25[up_check_25 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax2.scatter(x_21[up_check_21 == 1], y_21[up_check_21 == 1], marker=0, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_22[up_check_22 == 1], y_22[up_check_22 == 1], marker=0, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_23[up_check_23 == 1], y_23[up_check_23 == 1], marker=0, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_24[up_check_24 == 1], y_24[up_check_24 == 1], marker=0, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax2.scatter(x_25[up_check_25 == 1], y_25[up_check_25 == 1], marker=0, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax2.scatter(x_21[up_check_21 == 0], y_21[up_check_21 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax2.scatter(x_22[up_check_22 == 0], y_22[up_check_22 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax2.scatter(x_23[up_check_23 == 0], y_23[up_check_23 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax2.scatter(x_24[up_check_24 == 0], y_24[up_check_24 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax2.scatter(x_25[up_check_25 == 0], y_25[up_check_25 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		else:
			ax2.scatter(x_21, y_21, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax2.scatter(x_22, y_22, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax2.scatter(x_23, y_23, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax2.scatter(x_24, y_24, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax2.scatter(x_25, y_25, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax2.errorbar(x2_med, y2_med, xerr=[x2_err_min, x2_err_max], yerr=[y2_err_min, y2_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax2.scatter(x21_med, y21_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x22_med, y22_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x23_med, y23_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x24_med, y24_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x25_med, y25_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)


		ax2.set_xlim(xlim1, xlim2)
		ax2.set_ylim(ylim1, ylim2)
		ax2.set_xlabel(xlabel)
		ax2.set_xticks(xticks)
		ax2.set_yticks(yticks)
		ax2.set_yticklabels([])
		if Norm == 'None' or Norm == 'Y':
			secax2 = ax2.secondary_xaxis('top', functions=(solar, ergs))
			secax2.set_xlabel(r'log '+x_var+r' [L$_{\odot}$]')
		ax2.set_title(f'{zlim2} < z < {zlim3}')
		ax2.grid()

		ax3 = plt.subplot(133, aspect='equal', adjustable='box')
		if X == 'FIR':
			ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], marker=0, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], marker=0, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], marker=0, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], marker=0, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], marker=0, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax3.scatter(x_31[up_check_31 == 0], y_31[up_check_31 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32[up_check_32 == 0], y_32[up_check_32 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33[up_check_33 == 0], y_33[up_check_33 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34[up_check_34 == 0], y_34[up_check_34 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35[up_check_35 == 0], y_35[up_check_35 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)

		else:
			ax3.scatter(x_31, y_31, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_32, y_32, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_33, y_33, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_34, y_34, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
			ax3.scatter(x_35, y_35, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)

		if Median != 'None':
			ax3.errorbar(x3_med, y3_med, xerr=[x3_err_min, x3_err_max], yerr=[y3_err_min, y3_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax3.scatter(x31_med, y31_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x32_med, y32_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x33_med, y33_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x34_med, y34_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x35_med, y35_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		ax3.set_xlim(xlim1, xlim2)
		ax3.set_ylim(ylim1, ylim2)
		ax3.set_yticklabels([])
		ax3.set_xticks(xticks)
		ax3.set_yticks(yticks)
		if Norm == 'None':
			secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(r' ')
			secax3 = ax3.secondary_yaxis('right', functions=(solar, ergs))
			secax3.set_ylabel(r'log L ('+y_var+r') [L$_{\odot}$]')
		elif Norm == 'Y':
			secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(r' ')
		ax3.set_title(f'{zlim3} < z < {zlim4}')
		ax3.grid()

		plt.tight_layout()
		plt.savefig(f'/Users/connor_auge/Desktop/New_plots4/{savestring}.pdf')
		plt.show()


	def ratio_plots(self, savestring, X, Y, Median, Nh, Lx, L, f1, f2, f3, f4, F1, field, spec_z, uv_slope, mir_slope1, mir_slope2, up_check, ulirg_Nh=None, ulirg_Lx=None, ulirg_Flux=None, ulirg_F1=None):
		plt.rcParams['font.size'] = 20
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = (uv_slope < -0.3) & (mir_slope1 >= -0.2)
		B2 = (uv_slope >= -0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2)
		B3 = (uv_slope > 0.2) & (mir_slope1 >= -0.2)
		B4 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0)
		B5 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0)

		zlim1 = 0.0
		zlim2 = 0.6
		zlim3 = 0.9
		zlim4 = 1.2

		zbins1 = (spec_z >= zlim1) & (spec_z <= zlim2)
		zbins2 = (spec_z >= zlim2) & (spec_z <= zlim3)
		zbins3 = (spec_z >= zlim3) & (spec_z <= zlim4)

		Nh[Nh <= 0] = np.nan

		if X == 'Nh':
			x = np.log10(Nh)
			if any(ulirg_Nh) != None:
				xlim1 = 19.8
				xlim2 = 24.8
			else:
				xlim1 = 19
				xlim2 = 24
			xlabel = r'log N$_{\mathrm{H}}$'

		elif X == 'Lx':
			x = Lx
			xlim1 = 42.5
			xlim2 = 46.5
			xlabel = r'log L$_{\mathrm{X}}$'


		elif X == 'Lbol':
			x = L
			xlim1 = 43
			xlim2 = 48
			xlabel = r'log L$_{\mathrm{bol}}$'
			
		if Y == 'UV':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f1/lx)
			y_var = r'0.25$\mu$m'
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2,-1,0,1,2]

		elif Y == 'MIR6':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f2/lx)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'MIR10':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f4/lx)
			ylabel = r'log L (10$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'FIR':
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f3/lx)
			ylabel = r'log L (100$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2,-1,0,1,2]

		elif Y == 'UV/MIR6':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_4*F1
			y = np.log10(f1/f2)
			ylabel = r'log L (0.25$\mu$m)/ L (10$\mu$m)'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'UV/MIR10':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f1/f4)
			ylabel = r'log L (0.25$\mu$m)/ L (10$\mu$m)'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2,-1,0,1,2]

		elif Y == 'UV/FIR':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f1/f3)
			ylabel = r'log L (0.25$\mu$m)/ L (100$\mu$m)'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'MIR6/FIR':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f2/f3)
			ylabel = r'log L (10$\mu$m)/ L (100$\mu$m)'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2,-1,0,1,2]

		elif Y == 'MIR10/FIR':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f4/f3)
			ylabel = r'log L (10$\mu$m)/ L (100$\mu$m)'
			ylim1 = -2.5
			ylim2 = 2.5
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'Lbol':
			l = np.asarray([10**i for i in L])
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(l/lx)
			ylabel = r'log L$_{\mathrm{bol}}$/ L$_{\mathrm{X}}$'
			ylim1 = 0 
			ylim2 = 4
			yticks = [0,1,2,3,4]

		elif Y == 'UV/Lbol':
			l = np.asarray([10**i for i in L])
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			y = np.log10(f1/l)
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1
			yticks = [-3,-2,-1,0,1]

		elif Y == 'MIR6/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			y = np.log10(f2/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1
			yticks = [-3,-2,-1,0,1]

		elif Y == 'MIR10/Lbol':
			l = np.asarray([10**i for i in L])
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f4/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1
			yticks = [-3,-2,-1,0,1]

		elif Y == 'FIR/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f3/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1
			yticks = [-3, -2, -1, 0, 1]

		else:
			print('Specify Y variable')
			return

		if any(ulirg_Flux) != None:
			u_f = np.asarray([10**i for i in ulirg_Flux])
			ulx = np.asarray([10**i for i in ulirg_Lx])
			ulirg_Flux = np.log10((u_f*ulirg_F1)/ulx)


		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		x_11 = x[zbins1][B1[zbins1]]
		y_11 = y[zbins1][B1[zbins1]]
		up_check_11 = up_check[zbins1][B1[zbins1]]
		x_12 = x[zbins1][B2[zbins1]]
		y_12 = y[zbins1][B2[zbins1]]
		up_check_12 = up_check[zbins1][B2[zbins1]]
		x_13 = x[zbins1][B3[zbins1]]
		y_13 = y[zbins1][B3[zbins1]]
		up_check_13 = up_check[zbins1][B3[zbins1]]
		x_14 = x[zbins1][B4[zbins1]]
		y_14 = y[zbins1][B4[zbins1]]
		up_check_14 = up_check[zbins1][B4[zbins1]]
		x_15 = x[zbins1][B5[zbins1]]
		y_15 = y[zbins1][B5[zbins1]]
		up_check_15 = up_check[zbins1][B5[zbins1]]

		x_21 = x[zbins2][B1[zbins2]]
		y_21 = y[zbins2][B1[zbins2]]
		up_check_21 = up_check[zbins2][B1[zbins2]]
		x_22 = x[zbins2][B2[zbins2]]
		y_22 = y[zbins2][B2[zbins2]]
		up_check_22 = up_check[zbins2][B2[zbins2]]
		x_23 = x[zbins2][B3[zbins2]]
		y_23 = y[zbins2][B3[zbins2]]
		up_check_23 = up_check[zbins2][B3[zbins2]]
		x_24 = x[zbins2][B4[zbins2]]
		y_24 = y[zbins2][B4[zbins2]]
		up_check_24 = up_check[zbins2][B4[zbins2]]
		x_25 = x[zbins2][B5[zbins2]]
		y_25 = y[zbins2][B5[zbins2]]
		up_check_25 = up_check[zbins2][B5[zbins2]]

		x_31 = x[zbins3][B1[zbins3]]
		y_31 = y[zbins3][B1[zbins3]]
		up_check_31 = up_check[zbins3][B1[zbins3]]
		x_32 = x[zbins3][B2[zbins3]]
		y_32 = y[zbins3][B2[zbins3]]
		up_check_32 = up_check[zbins3][B2[zbins3]]
		x_33 = x[zbins3][B3[zbins3]]
		y_33 = y[zbins3][B3[zbins3]]
		up_check_33 = up_check[zbins3][B3[zbins3]]
		x_34 = x[zbins3][B4[zbins3]]
		y_34 = y[zbins3][B4[zbins3]]
		up_check_34 = up_check[zbins3][B4[zbins3]]
		x_35 = x[zbins3][B5[zbins3]]
		y_35 = y[zbins3][B5[zbins3]]
		up_check_35 = up_check[zbins3][B5[zbins3]]

		if Median == 'Bins':
			c1m = c1
			c2m = c2
			c3m = c3
			c4m = c4
			c5m = c5

			# Median
			x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
			x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
			x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
			x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
			x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

			x21_med, y21_med = np.nanmedian(x_21), np.nanmedian(y_21)
			x22_med, y22_med = np.nanmedian(x_22), np.nanmedian(y_22)
			x23_med, y23_med = np.nanmedian(x_23), np.nanmedian(y_23)
			x24_med, y24_med = np.nanmedian(x_24), np.nanmedian(y_24)
			x25_med, y25_med = np.nanmedian(x_25), np.nanmedian(y_25)

			x31_med, y31_med = np.nanmedian(x_31), np.nanmedian(y_31)
			x32_med, y32_med = np.nanmedian(x_32), np.nanmedian(y_32)
			x33_med, y33_med = np.nanmedian(x_33), np.nanmedian(y_33)
			x34_med, y34_med = np.nanmedian(x_34), np.nanmedian(y_34)
			x35_med, y35_med = np.nanmedian(x_35), np.nanmedian(y_35)

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])
			x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med]), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med])
			x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med]), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med])


			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
			x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
			x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
			x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
			x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

			x21_25per, y21_25per = np.nanpercentile(x_21, 25), np.nanpercentile(y_21, 25)
			x22_25per, y22_25per = np.nanpercentile(x_22, 25), np.nanpercentile(y_22, 25)
			x23_25per, y23_25per = np.nanpercentile(x_23, 25), np.nanpercentile(y_23, 25)
			x24_25per, y24_25per = np.nanpercentile(x_24, 25), np.nanpercentile(y_24, 25)
			x25_25per, y25_25per = np.nanpercentile(x_25, 25), np.nanpercentile(y_25, 25)

			x31_25per, y31_25per = np.nanpercentile(x_31, 25), np.nanpercentile(y_31, 25)
			x32_25per, y32_25per = np.nanpercentile(x_32, 25), np.nanpercentile(y_32, 25)
			x33_25per, y33_25per = np.nanpercentile(x_33, 25), np.nanpercentile(y_33, 25)
			x34_25per, y34_25per = np.nanpercentile(x_34, 25), np.nanpercentile(y_34, 25)
			x35_25per, y35_25per = np.nanpercentile(x_35, 25), np.nanpercentile(y_35, 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
			x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per]), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per])
			x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per]), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per])

			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
			x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
			x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
			x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
			x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

			x21_75per, y21_75per = np.nanpercentile(x_21, 75), np.nanpercentile(y_21, 75)
			x22_75per, y22_75per = np.nanpercentile(x_22, 75), np.nanpercentile(y_22, 75)
			x23_75per, y23_75per = np.nanpercentile(x_23, 75), np.nanpercentile(y_23, 75)
			x24_75per, y24_75per = np.nanpercentile(x_24, 75), np.nanpercentile(y_24, 75)
			x25_75per, y25_75per = np.nanpercentile(x_25, 75), np.nanpercentile(y_25, 75)

			x31_75per, y31_75per = np.nanpercentile(x_31, 75), np.nanpercentile(y_31, 75)
			x32_75per, y32_75per = np.nanpercentile(x_32, 75), np.nanpercentile(y_32, 75)
			x33_75per, y33_75per = np.nanpercentile(x_33, 75), np.nanpercentile(y_33, 75)
			x34_75per, y34_75per = np.nanpercentile(x_34, 75), np.nanpercentile(y_34, 75)
			x35_75per, y35_75per = np.nanpercentile(x_35, 75), np.nanpercentile(y_35, 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
			x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per]) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y2_med
			x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per]) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y3_med

		elif Median == 'X-axis':

			c1m = 'gray'
			c2m = 'gray'
			c3m = 'gray'
			c4m = 'gray'
			c5m = 'gray'

			b1 = (Lx > 43)&(Lx < 43.5)
			b2 = (Lx > 43.5)&(Lx < 44)
			b3 = (Lx > 44)&(Lx < 44.5)
			b4 = (Lx > 44.5)&(Lx < 45)
			b5 = (Lx > 45)

			# Median
			x11_med, y11_med = np.nanmedian(x[zbins1][b1[zbins1]]), np.nanmedian(y[zbins1][b1[zbins1]])
			x12_med, y12_med = np.nanmedian(x[zbins1][b2[zbins1]]), np.nanmedian(y[zbins1][b2[zbins1]])
			x13_med, y13_med = np.nanmedian(x[zbins1][b3[zbins1]]), np.nanmedian(y[zbins1][b3[zbins1]])
			x14_med, y14_med = np.nanmedian(x[zbins1][b4[zbins1]]), np.nanmedian(y[zbins1][b4[zbins1]])
			x15_med, y15_med = np.nanmedian(x[zbins1][b5[zbins1]]), np.nanmedian(y[zbins1][b5[zbins1]])

			x21_med, y21_med = np.nanmedian(x[zbins2][b1[zbins2]]), np.nanmedian(y[zbins2][b1[zbins2]])
			x22_med, y22_med = np.nanmedian(x[zbins2][b2[zbins2]]), np.nanmedian(y[zbins2][b2[zbins2]])
			x23_med, y23_med = np.nanmedian(x[zbins2][b3[zbins2]]), np.nanmedian(y[zbins2][b3[zbins2]])
			x24_med, y24_med = np.nanmedian(x[zbins2][b4[zbins2]]), np.nanmedian(y[zbins2][b4[zbins2]])
			x25_med, y25_med = np.nanmedian(x[zbins2][b5[zbins2]]), np.nanmedian(y[zbins2][b5[zbins2]])

			x31_med, y31_med = np.nanmedian(x[zbins3][b1[zbins3]]), np.nanmedian(y[zbins3][b1[zbins3]])
			x32_med, y32_med = np.nanmedian(x[zbins3][b2[zbins3]]), np.nanmedian(y[zbins3][b2[zbins3]])
			x33_med, y33_med = np.nanmedian(x[zbins3][b3[zbins3]]), np.nanmedian(y[zbins3][b3[zbins3]])
			x34_med, y34_med = np.nanmedian(x[zbins3][b4[zbins3]]), np.nanmedian(y[zbins3][b4[zbins3]])
			x35_med, y35_med = np.nanmedian(x[zbins3][b5[zbins3]]), np.nanmedian(y[zbins3][b5[zbins3]])

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])
			x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med]), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med])
			x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med]), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med])

			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x[zbins1][b1[zbins1]], 25), np.nanpercentile(y[zbins1][b1[zbins1]], 25)
			x12_25per, y12_25per = np.nanpercentile(x[zbins1][b2[zbins1]], 25), np.nanpercentile(y[zbins1][b2[zbins1]], 25)
			x13_25per, y13_25per = np.nanpercentile(x[zbins1][b3[zbins1]], 25), np.nanpercentile(y[zbins1][b3[zbins1]], 25)
			x14_25per, y14_25per = np.nanpercentile(x[zbins1][b4[zbins1]], 25), np.nanpercentile(y[zbins1][b4[zbins1]], 25)
			x15_25per, y15_25per = np.nanpercentile(x[zbins1][b5[zbins1]], 25), np.nanpercentile(y[zbins1][b5[zbins1]], 25)

			x21_25per, y21_25per = np.nanpercentile(x[zbins2][b1[zbins2]], 25), np.nanpercentile(y[zbins2][b1[zbins2]], 25)
			x22_25per, y22_25per = np.nanpercentile(x[zbins2][b2[zbins2]], 25), np.nanpercentile(y[zbins2][b2[zbins2]], 25)
			x23_25per, y23_25per = np.nanpercentile(x[zbins2][b3[zbins2]], 25), np.nanpercentile(y[zbins2][b3[zbins2]], 25)
			x24_25per, y24_25per = np.nanpercentile(x[zbins2][b4[zbins2]], 25), np.nanpercentile(y[zbins2][b4[zbins2]], 25)
			x25_25per, y25_25per = np.nanpercentile(x[zbins2][b5[zbins2]], 25), np.nanpercentile(y[zbins2][b5[zbins2]], 25)

			x31_25per, y31_25per = np.nanpercentile(x[zbins3][b1[zbins3]], 25), np.nanpercentile(y[zbins3][b1[zbins3]], 25)
			x32_25per, y32_25per = np.nanpercentile(x[zbins3][b2[zbins3]], 25), np.nanpercentile(y[zbins3][b2[zbins3]], 25)
			x33_25per, y33_25per = np.nanpercentile(x[zbins3][b3[zbins3]], 25), np.nanpercentile(y[zbins3][b3[zbins3]], 25)
			x34_25per, y34_25per = np.nanpercentile(x[zbins3][b4[zbins3]], 25), np.nanpercentile(y[zbins3][b4[zbins3]], 25)
			x35_25per, y35_25per = np.nanpercentile(x[zbins3][b5[zbins3]], 25), np.nanpercentile(y[zbins3][b5[zbins3]], 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
			x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per]), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per])
			x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per]), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per])

			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x[zbins1][b1[zbins1]], 75), np.nanpercentile(y[zbins1][b1[zbins1]], 75)
			x12_75per, y12_75per = np.nanpercentile(x[zbins1][b2[zbins1]], 75), np.nanpercentile(y[zbins1][b2[zbins1]], 75)
			x13_75per, y13_75per = np.nanpercentile(x[zbins1][b3[zbins1]], 75), np.nanpercentile(y[zbins1][b3[zbins1]], 75)
			x14_75per, y14_75per = np.nanpercentile(x[zbins1][b4[zbins1]], 75), np.nanpercentile(y[zbins1][b4[zbins1]], 75)
			x15_75per, y15_75per = np.nanpercentile(x[zbins1][b5[zbins1]], 75), np.nanpercentile(y[zbins1][b5[zbins1]], 75)

			x21_75per, y21_75per = np.nanpercentile(x[zbins2][b1[zbins2]], 75), np.nanpercentile(y[zbins2][b1[zbins2]], 75)
			x22_75per, y22_75per = np.nanpercentile(x[zbins2][b2[zbins2]], 75), np.nanpercentile(y[zbins2][b2[zbins2]], 75)
			x23_75per, y23_75per = np.nanpercentile(x[zbins2][b3[zbins2]], 75), np.nanpercentile(y[zbins2][b3[zbins2]], 75)
			x24_75per, y24_75per = np.nanpercentile(x[zbins2][b4[zbins2]], 75), np.nanpercentile(y[zbins2][b4[zbins2]], 75)
			x25_75per, y25_75per = np.nanpercentile(x[zbins2][b5[zbins2]], 75), np.nanpercentile(y[zbins2][b5[zbins2]], 75)

			x31_75per, y31_75per = np.nanpercentile(x[zbins3][b1[zbins3]], 75), np.nanpercentile(y[zbins3][b1[zbins3]], 75)
			x32_75per, y32_75per = np.nanpercentile(x[zbins3][b2[zbins3]], 75), np.nanpercentile(y[zbins3][b2[zbins3]], 75)
			x33_75per, y33_75per = np.nanpercentile(x[zbins3][b3[zbins3]], 75), np.nanpercentile(y[zbins3][b3[zbins3]], 75)
			x34_75per, y34_75per = np.nanpercentile(x[zbins3][b4[zbins3]], 75), np.nanpercentile(y[zbins3][b4[zbins3]], 75)
			x35_75per, y35_75per = np.nanpercentile(x[zbins3][b5[zbins3]], 75), np.nanpercentile(y[zbins3][b5[zbins3]], 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
			x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per]) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y2_med
			x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per]) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y3_med



		fig = plt.figure(figsize=(18, 7))
		ax1 = plt.subplot(131, aspect='equal', adjustable='box')
		ax1.scatter(x_11, y_11, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
		ax1.scatter(x_12, y_12, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
		ax1.scatter(x_13, y_13, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
		ax1.scatter(x_14, y_14, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
		ax1.scatter(x_15, y_15, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		if any(ulirg_Nh) != None:
			print(ulirg_Flux)
			ax1.scatter(ulirg_Nh,ulirg_Flux,color='k',marker='s',s=75,label='ULIRGs')


		ax1.set_xlim(xlim1,xlim2)
		ax1.set_ylim(ylim1,ylim2)
		ax1.set_yticks(yticks)
		ax1.set_ylabel(ylabel)
		ax1.set_title(f'{zlim1} < z < {zlim2}')
		ax1.legend(fontsize=14)
		ax1.grid()

		ax2 = plt.subplot(132, aspect='equal', adjustable='box')
		ax2.scatter(x_21, y_21, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
		ax2.scatter(x_22, y_22, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
		ax2.scatter(x_23, y_23, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
		ax2.scatter(x_24, y_24, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
		ax2.scatter(x_25, y_25, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax2.errorbar(x2_med, y2_med, xerr=[x2_err_min, x2_err_max], yerr=[y2_err_min, y2_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax2.scatter(x21_med, y21_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x22_med, y22_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x23_med, y23_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x24_med, y24_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax2.scatter(x25_med, y25_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		ax2.set_xlim(xlim1, xlim2)
		ax2.set_ylim(ylim1, ylim2)
		ax2.set_yticks(yticks)
		ax2.set_xlabel(xlabel)
		ax2.set_yticklabels([])
		# ax2.legend(fontsize=14)
		ax2.set_title(f'{zlim2} < z < {zlim3}')
		ax2.grid()

		ax3 = plt.subplot(133, aspect='equal', adjustable='box')
		ax3.scatter(x_31, y_31, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
		ax3.scatter(x_32, y_32, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
		ax3.scatter(x_33, y_33, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
		ax3.scatter(x_34, y_34, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)
		ax3.scatter(x_35, y_35, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8,zorder=0)

		if Median != 'None':
			ax3.errorbar(x3_med, y3_med, xerr=[x3_err_min, x3_err_max], yerr=[y3_err_min, y3_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax3.scatter(x31_med, y31_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x32_med, y32_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x33_med, y33_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x34_med, y34_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax3.scatter(x35_med, y35_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		ax3.set_xlim(xlim1, xlim2)
		ax3.set_ylim(ylim1, ylim2)
		ax3.set_yticks(yticks)
		ax3.set_yticklabels([])
		ax3.set_title(f'{zlim3} < z < {zlim4}')
		ax3.grid()

		plt.tight_layout()
		plt.savefig(f'/Users/connor_auge/Desktop/New_plots3/{savestring}.pdf')
		plt.show()

	def Upanels_ratio_plots(self, savestring, X, Y, Median, Nh, Lx, L, L1, L2, L3, f1, f2, f3, f4, F1, field, spec_z, uv_slope, mir_slope1, mir_slope2, up_check, shape=None):
			plt.rcParams['font.size'] = 20
			plt.rcParams['axes.linewidth'] = 2
			plt.rcParams['xtick.major.size'] = 4
			plt.rcParams['xtick.major.width'] = 3
			plt.rcParams['ytick.major.size'] = 4
			plt.rcParams['ytick.major.width'] = 3

			# B1 = (uv_slope < -0.3) & (mir_slope1 >= -0.2)
			# B2 = (uv_slope >= -0.3) & (uv_slope <= 0.2) & (mir_slope1 >= -0.2)
			# B3 = (uv_slope > 0.2) & (mir_slope1 >= -0.2)
			# B4 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 > 0.0)
			# B5 = (uv_slope >= -0.3) & (mir_slope1 < -0.2) & (mir_slope2 <= 0.0)

			B1 = shape == 1
			B2 = shape == 2
			B3 = shape == 3
			B4 = shape == 4
			B5 = shape == 5


			Nh[Nh <= 0] = np.nan

			if X == 'Nh':
				x = np.log10(Nh)
				xlim1 = 19.5
				xlim2 = 24.5
				xticks = [20,21,22,23,24]
				xlabel = r'log N$_{\mathrm{H}}$ [cm$^{-2}$]'

			elif X == 'Lx':
				x = Lx
				xlim1 = 42.5
				xlim2 = 46.5
				xticks = [43,44,45,46]
				xlabel = r'log L$_{\mathrm{X}}$ [erg/s]'


			elif X == 'Lbol':
				x = L
				xlim1 = 43.5
				xlim2 = 47
				xticks = [44,45,46,47]
				xlabel = r'log L$_{\mathrm{bol}}$ [erg/s]'


			if Y == 'UV-MIR-FIR/Lbol':
				y1 = L1 - L
				y2 = L2 - L
				y3 = L3 - L
			
				ylabel1 = r'log L (UV)/ L$_{\mathrm{bol}}$'
				ylabel2 = r'log L (MIR)/ L$_{\mathrm{bol}}$'
				ylabel3 = r'log L (FIR)/ L$_{\mathrm{bol}}$'
				ylim1 = -3
				ylim2 = 0.5
				yticks = [-3,-2,-1,0]

			elif Y == 'UV/MIR-UV/Lx-MIR/Lx':
				lx = np.asarray([10**i for i in Lx])
				f_1 = np.asarray([10**i for i in f1])
				f_2 = np.asarray([10**i for i in f2])
				f_3 = np.asarray([10**i for i in f3])
				f1 = f_1*F1
				f2 = f_2*F1
				f3 = f_3*F1
				y1 = np.log10(f1/f2)
				y2 = np.log10(f1/lx)
				y3 = np.log10(f2/lx)

				ylabel1 = r'log L (UV)/ L(MIR)'
				ylabel2 = r'log L (UV)/ L$_{\mathrm{X}}$'
				ylabel3 = r'log L (MIR)/ L$_{\mathrm{X}}$'
				ylim1 = -2.5
				ylim2 = 2.5
				yticks = [-2,-1,0,1,2]

			elif Y == 'UV-MIR-FIR/Lx':
				y1 = L1 - Lx
				y2 = L2 - Lx
				y3 = L3 - Lx

				ylabel1 = r'log L (UV)/ L$_{\mathrm{X}}$'
				ylabel2 = r'log L (MIR)/ L$_{\mathrm{X}}$'
				ylabel3 = r'log L (FIR)/ L$_{\mathrm{X}}$'
				ylim1 = -3
				ylim2 = 1
				yticks = [-3,-2,-1,0,1]

			else:
				print('Specify all three Y axis variabls. Options are:  UV-MIR-FIR/Lbol;   UV/MIR-UV/Lx-MIR/Lx;   UV-MIR-FIR/Lx')
				return


			c1 = '#377eb8'
			c2 = '#984ea3'
			c3 = '#4daf4a'
			c4 = '#ff7f00'
			c5 = '#e41a1c'

			x_11 = x[B1]
			y_11 = y1[B1]
			up_check_11 = up_check[B1]
			x_12 = x[B2]
			y_12 = y1[B2]
			up_check_12 = up_check[B2]
			x_13 = x[B3]
			y_13 = y1[B3]
			up_check_13 = up_check[B3]
			x_14 = x[B4]
			y_14 = y1[B4]
			up_check_14 = up_check[B4]
			x_15 = x[B5]
			y_15 = y1[B5]
			up_check_15 = up_check[B5]

			x_21 = x[B1]
			y_21 = y2[B1]
			up_check_21 = up_check[B1]
			x_22 = x[B2]
			y_22 = y2[B2]
			up_check_22 = up_check[B2]
			x_23 = x[B3]
			y_23 = y2[B3]
			up_check_23 = up_check[B3]
			x_24 = x[B4]
			y_24 = y2[B4]
			up_check_24 = up_check[B4]
			x_25 = x[B5]
			y_25 = y2[B5]
			up_check_25 = up_check[B5]

			x_31 = x[B1]
			y_31 = y3[B1]
			up_check_31 = up_check[B1]
			x_32 = x[B2]
			y_32 = y3[B2]
			up_check_32 = up_check[B2]
			x_33 = x[B3]
			y_33 = y3[B3]
			up_check_33 = up_check[B3]
			x_34 = x[B4]
			y_34 = y3[B4]
			up_check_34 = up_check[B4]
			x_35 = x[B5]
			y_35 = y3[B5]
			up_check_35 = up_check[B5]

			if Median == 'Bins':
				c1m = c1
				c2m = c2
				c3m = c3
				c4m = c4
				c5m = c5

				# Median
				x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
				x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
				x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
				x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
				x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

				x21_med, y21_med = np.nanmedian(x_21), np.nanmedian(y_21)
				x22_med, y22_med = np.nanmedian(x_22), np.nanmedian(y_22)
				x23_med, y23_med = np.nanmedian(x_23), np.nanmedian(y_23)
				x24_med, y24_med = np.nanmedian(x_24), np.nanmedian(y_24)
				x25_med, y25_med = np.nanmedian(x_25), np.nanmedian(y_25)

				x31_med, y31_med = np.nanmedian(x_31), np.nanmedian(y_31)
				x32_med, y32_med = np.nanmedian(x_32), np.nanmedian(y_32)
				x33_med, y33_med = np.nanmedian(x_33), np.nanmedian(y_33)
				x34_med, y34_med = np.nanmedian(x_34), np.nanmedian(y_34)
				x35_med, y35_med = np.nanmedian(x_35), np.nanmedian(y_35)

				# if Y == 'UV-MIR-FIR/Lbol':
				# 	x31_med, y31_med = np.nanmedian(x_31[up_check_31 == 0]), np.nanmedian(y_31[up_check_31 == 0])
				# 	x32_med, y32_med = np.nanmedian(x_32[up_check_32 == 0]), np.nanmedian(y_32[up_check_32 == 0])
				# 	x33_med, y33_med = np.nanmedian(x_33[up_check_33 == 0]), np.nanmedian(y_33[up_check_33 == 0])
				# 	x34_med, y34_med = np.nanmedian(x_34[up_check_34 == 0]), np.nanmedian(y_34[up_check_34 == 0])
				# 	x35_med, y35_med = np.nanmedian(x_35[up_check_35 == 0]), np.nanmedian(y_35[up_check_35 == 0])

				x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])
				x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med]), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med])
				x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med]), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med])


				# 25 Percentile 
				x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
				x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
				x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
				x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
				x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

				x21_25per, y21_25per = np.nanpercentile(x_21, 25), np.nanpercentile(y_21, 25)
				x22_25per, y22_25per = np.nanpercentile(x_22, 25), np.nanpercentile(y_22, 25)
				x23_25per, y23_25per = np.nanpercentile(x_23, 25), np.nanpercentile(y_23, 25)
				x24_25per, y24_25per = np.nanpercentile(x_24, 25), np.nanpercentile(y_24, 25)
				x25_25per, y25_25per = np.nanpercentile(x_25, 25), np.nanpercentile(y_25, 25)

				x31_25per, y31_25per = np.nanpercentile(x_31, 25), np.nanpercentile(y_31, 25)
				x32_25per, y32_25per = np.nanpercentile(x_32, 25), np.nanpercentile(y_32, 25)
				x33_25per, y33_25per = np.nanpercentile(x_33, 25), np.nanpercentile(y_33, 25)
				x34_25per, y34_25per = np.nanpercentile(x_34, 25), np.nanpercentile(y_34, 25)
				x35_25per, y35_25per = np.nanpercentile(x_35, 25), np.nanpercentile(y_35, 25)

				# if Y == 'UV-MIR-FIR/Lbol':
				# 	x11_35per, y31_25per = np.nanpercentile(x_31[up_check_31 == 0], 25), np.nanpercentile(y_31[up_check_31 == 0], 25)
				# 	x12_35per, y32_25per = np.nanpercentile(x_32[up_check_32 == 0], 25), np.nanpercentile(y_32[up_check_32 == 0], 25)
				# 	x13_35per, y33_25per = np.nanpercentile(x_33[up_check_33 == 0], 25), np.nanpercentile(y_33[up_check_33 == 0], 25)
				# 	x14_35per, y34_25per = np.nanpercentile(x_34[up_check_34 == 0], 25), np.nanpercentile(y_34[up_check_34 == 0], 25)
				# 	x15_35per, y35_25per = np.nanpercentile(x_35[up_check_35 == 0], 25), np.nanpercentile(y_35[up_check_35 == 0], 25)

				x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
				x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per]), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per])
				x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per]), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per])

				# 75 Percentile 
				x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
				x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
				x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
				x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
				x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

				x21_75per, y21_75per = np.nanpercentile(x_21, 75), np.nanpercentile(y_21, 75)
				x22_75per, y22_75per = np.nanpercentile(x_22, 75), np.nanpercentile(y_22, 75)
				x23_75per, y23_75per = np.nanpercentile(x_23, 75), np.nanpercentile(y_23, 75)
				x24_75per, y24_75per = np.nanpercentile(x_24, 75), np.nanpercentile(y_24, 75)
				x25_75per, y25_75per = np.nanpercentile(x_25, 75), np.nanpercentile(y_25, 75)

				x31_75per, y31_75per = np.nanpercentile(x_31, 75), np.nanpercentile(y_31, 75)
				x32_75per, y32_75per = np.nanpercentile(x_32, 75), np.nanpercentile(y_32, 75)
				x33_75per, y33_75per = np.nanpercentile(x_33, 75), np.nanpercentile(y_33, 75)
				x34_75per, y34_75per = np.nanpercentile(x_34, 75), np.nanpercentile(y_34, 75)
				x35_75per, y35_75per = np.nanpercentile(x_35, 75), np.nanpercentile(y_35, 75)

				# if Y == 'UV-MIR-FIR/Lbol':
				# 	x31_75per, y31_75per = np.nanpercentile(x_31[up_check_11 == 0], 75), np.nanpercentile(y_31[up_check_11 == 0], 75)
				# 	x32_75per, y32_75per = np.nanpercentile(x_32[up_check_12 == 0], 75), np.nanpercentile(y_32[up_check_12 == 0], 75)
				# 	x33_75per, y33_75per = np.nanpercentile(x_33[up_check_13 == 0], 75), np.nanpercentile(y_33[up_check_13 == 0], 75)
				# 	x34_75per, y34_75per = np.nanpercentile(x_34[up_check_14 == 0], 75), np.nanpercentile(y_34[up_check_14 == 0], 75)
				# 	x35_75per, y35_75per = np.nanpercentile(x_35[up_check_15 == 0], 75), np.nanpercentile(y_35[up_check_15 == 0], 75)

				x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
				x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per]) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y2_med
				x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per]) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y3_med

			elif Median == 'X-axis':

				c1m = 'gray'
				c2m = 'gray'
				c3m = 'gray'
				c4m = 'gray'
				c5m = 'gray'

				b1 = (Lx > 43)&(Lx < 43.5)
				b2 = (Lx > 43.5)&(Lx < 44)
				b3 = (Lx > 44)&(Lx < 44.5)
				b4 = (Lx > 44.5)&(Lx < 45)
				b5 = (Lx > 45)

				# Median
				x11_med, y11_med = np.nanmedian(x[b1]), np.nanmedian(y1[b1])
				x12_med, y12_med = np.nanmedian(x[b2]), np.nanmedian(y1[b2])
				x13_med, y13_med = np.nanmedian(x[b3]), np.nanmedian(y1[b3])
				x14_med, y14_med = np.nanmedian(x[b4]), np.nanmedian(y1[b4])
				x15_med, y15_med = np.nanmedian(x[b5]), np.nanmedian(y1[b5])

				x21_med, y21_med = np.nanmedian(x[b1]), np.nanmedian(y2[b1])
				x22_med, y22_med = np.nanmedian(x[b2]), np.nanmedian(y2[b2])
				x23_med, y23_med = np.nanmedian(x[b3]), np.nanmedian(y2[b3])
				x24_med, y24_med = np.nanmedian(x[b4]), np.nanmedian(y2[b4])
				x25_med, y25_med = np.nanmedian(x[b5]), np.nanmedian(y2[b5])

				x31_med, y31_med = np.nanmedian(x[b1]), np.nanmedian(y3[b1])
				x32_med, y32_med = np.nanmedian(x[b2]), np.nanmedian(y3[b2])
				x33_med, y33_med = np.nanmedian(x[b3]), np.nanmedian(y3[b3])
				x34_med, y34_med = np.nanmedian(x[b4]), np.nanmedian(y3[b4])
				x35_med, y35_med = np.nanmedian(x[b5]), np.nanmedian(y3[b5])

				x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])
				x2_med, y2_med = np.asarray([x21_med,x22_med,x23_med,x24_med,x25_med]), np.asarray([y21_med,y22_med,y23_med,y24_med,y25_med])
				x3_med, y3_med = np.asarray([x31_med,x32_med,x33_med,x34_med,x35_med]), np.asarray([y31_med,y32_med,y33_med,y34_med,y35_med])

				# 25 Percentile 
				x11_25per, y11_25per = np.nanpercentile(x[b1], 25), np.nanpercentile(y1[b1], 25)
				x12_25per, y12_25per = np.nanpercentile(x[b2], 25), np.nanpercentile(y1[b2], 25)
				x13_25per, y13_25per = np.nanpercentile(x[b3], 25), np.nanpercentile(y1[b3], 25)
				x14_25per, y14_25per = np.nanpercentile(x[b4], 25), np.nanpercentile(y1[b4], 25)
				x15_25per, y15_25per = np.nanpercentile(x[b5], 25), np.nanpercentile(y1[b5], 25)

				x21_25per, y21_25per = np.nanpercentile(x[b1], 25), np.nanpercentile(y2[b1], 25)
				x22_25per, y22_25per = np.nanpercentile(x[b2], 25), np.nanpercentile(y2[b2], 25)
				x23_25per, y23_25per = np.nanpercentile(x[b3], 25), np.nanpercentile(y2[b3], 25)
				x24_25per, y24_25per = np.nanpercentile(x[b4], 25), np.nanpercentile(y2[b4], 25)
				x25_25per, y25_25per = np.nanpercentile(x[b5], 25), np.nanpercentile(y2[b5], 25)

				x31_25per, y31_25per = np.nanpercentile(x[b1], 25), np.nanpercentile(y3[b1], 25)
				x32_25per, y32_25per = np.nanpercentile(x[b2], 25), np.nanpercentile(y3[b2], 25)
				x33_25per, y33_25per = np.nanpercentile(x[b3], 25), np.nanpercentile(y3[b3], 25)
				x34_25per, y34_25per = np.nanpercentile(x[b4], 25), np.nanpercentile(y3[b4], 25)
				x35_25per, y35_25per = np.nanpercentile(x[b5], 25), np.nanpercentile(y3[b5], 25)

				x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
				x2_err_min, y2_err_min = x2_med - np.asarray([x21_25per,x22_25per,x23_25per,x24_25per,x25_25per]), y2_med - np.asarray([y21_25per,y22_25per,y23_25per,y24_25per,y25_25per])
				x3_err_min, y3_err_min = x3_med - np.asarray([x31_25per,x32_25per,x33_25per,x34_25per,x35_25per]), y3_med - np.asarray([y31_25per,y32_25per,y33_25per,y34_25per,y35_25per])

				# 75 Percentile 
				x11_75per, y11_75per = np.nanpercentile(x[b1], 75), np.nanpercentile(y1[b1], 75)
				x12_75per, y12_75per = np.nanpercentile(x[b2], 75), np.nanpercentile(y1[b2], 75)
				x13_75per, y13_75per = np.nanpercentile(x[b3], 75), np.nanpercentile(y1[b3], 75)
				x14_75per, y14_75per = np.nanpercentile(x[b4], 75), np.nanpercentile(y1[b4], 75)
				x15_75per, y15_75per = np.nanpercentile(x[b5], 75), np.nanpercentile(y1[b5], 75)

				x21_75per, y21_75per = np.nanpercentile(x[b1], 75), np.nanpercentile(y2[b1], 75)
				x22_75per, y22_75per = np.nanpercentile(x[b2], 75), np.nanpercentile(y2[b2], 75)
				x23_75per, y23_75per = np.nanpercentile(x[b3], 75), np.nanpercentile(y2[b3], 75)
				x24_75per, y24_75per = np.nanpercentile(x[b4], 75), np.nanpercentile(y2[b4], 75)
				x25_75per, y25_75per = np.nanpercentile(x[b5], 75), np.nanpercentile(y2[b5], 75)

				x31_75per, y31_75per = np.nanpercentile(x[b1], 75), np.nanpercentile(y3[b1], 75)
				x32_75per, y32_75per = np.nanpercentile(x[b2], 75), np.nanpercentile(y3[b2], 75)
				x33_75per, y33_75per = np.nanpercentile(x[b3], 75), np.nanpercentile(y3[b3], 75)
				x34_75per, y34_75per = np.nanpercentile(x[b4], 75), np.nanpercentile(y3[b4], 75)
				x35_75per, y35_75per = np.nanpercentile(x[b5], 75), np.nanpercentile(y3[b5], 75)

				x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
				x2_err_max, y2_err_max = np.asarray([x21_75per,x22_75per,x23_75per,x24_75per,x25_75per]) - x2_med, np.asarray([y21_75per,y22_75per,y23_75per,y24_75per,y25_75per]) - y2_med
				x3_err_max, y3_err_max = np.asarray([x31_75per,x32_75per,x33_75per,x34_75per,x35_75per]) - x3_med, np.asarray([y31_75per,y32_75per,y33_75per,y34_75per,y35_75per]) - y3_med

			def solar(x):
				return x - np.log10(3.8E33)

			def ergs(x):
				return x + np.log10(3.8E33)

			fig = plt.figure(figsize=(18, 7))
			ax1 = plt.subplot(131, aspect='equal', adjustable='box')

			ax1.scatter(x_11, y_11, color=c1, marker='+', rasterized=True, alpha=0.6, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='+', rasterized=True, alpha=0.6, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='+', rasterized=True, alpha=0.6, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='+', rasterized=True, alpha=0.6, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='+', rasterized=True, alpha=0.6, label='Panel 5',zorder=0)

			if Median != 'None':
				ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
				ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)


			ax1.set_xlim(xlim1,xlim2)
			ax1.set_ylim(ylim1,ylim2)
			ax1.set_xticks(xticks)
			ax1.set_yticks(yticks)
			ax1.set_ylabel(ylabel1)
			ax1.set_xlabel(' ')
			if X == 'Lbol':
				secax1 = ax1.secondary_xaxis('top', functions=(solar, ergs))
				secax1.set_xlabel(r' ')
			ax1.legend(fontsize=14)
			ax1.grid()

			ax2 = plt.subplot(132, aspect='equal', adjustable='box')
			ax2.scatter(x_21, y_21, color=c1, marker='+', rasterized=True, alpha=0.6, label='Panel 1',zorder=0)
			ax2.scatter(x_22, y_22, color=c2, marker='+', rasterized=True, alpha=0.6, label='Panel 2',zorder=0)
			ax2.scatter(x_23, y_23, color=c3, marker='+', rasterized=True, alpha=0.6, label='Panel 3',zorder=0)
			ax2.scatter(x_24, y_24, color=c4, marker='+', rasterized=True, alpha=0.6, label='Panel 4',zorder=0)
			ax2.scatter(x_25, y_25, color=c5, marker='+', rasterized=True, alpha=0.6, label='Panel 5',zorder=0)

			if Median != 'None':
				ax2.errorbar(x2_med, y2_med, xerr=[x2_err_min, x2_err_max], yerr=[y2_err_min, y2_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
				ax2.scatter(x21_med, y21_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax2.scatter(x22_med, y22_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax2.scatter(x23_med, y23_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax2.scatter(x24_med, y24_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax2.scatter(x25_med, y25_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

			ax2.set_xlim(xlim1, xlim2)
			ax2.set_ylim(ylim1, ylim2)
			ax2.set_xticks(xticks)
			ax2.set_yticks(yticks)
			ax2.set_xlabel(xlabel)
			ax2.set_ylabel(ylabel2)
			if X == 'Lbol':
				secax2 = ax2.secondary_xaxis('top', functions=(solar, ergs))
				secax2.set_xlabel(r'log L$_{\mathrm{bol}}$ [L$_{\odot}$]' )
			ax2.grid()

			ax3 = plt.subplot(133, aspect='equal', adjustable='box')
			if 'FIR' in Y:
				ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], marker=11, color=c1, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], marker=11, color=c2, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], marker=11, color=c3, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], marker=11, color=c4, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], marker=11, color=c5, rasterized=True, alpha=0.4,zorder=0)

				ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], marker=2, color=c1, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], marker=2, color=c2, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], marker=2, color=c3, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], marker=2, color=c4, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], marker=2, color=c5, rasterized=True, alpha=0.4,zorder=0)

				ax3.scatter(x_31[up_check_31 == 0], y_31[up_check_31 == 0], color=c1, marker='+', rasterized=True, alpha=0.6, label='Panel 1',zorder=0)
				ax3.scatter(x_32[up_check_32 == 0], y_32[up_check_32 == 0], color=c2, marker='+', rasterized=True, alpha=0.6, label='Panel 2',zorder=0)
				ax3.scatter(x_33[up_check_33 == 0], y_33[up_check_33 == 0], color=c3, marker='+', rasterized=True, alpha=0.6, label='Panel 3',zorder=0)
				ax3.scatter(x_34[up_check_34 == 0], y_34[up_check_34 == 0], color=c4, marker='+', rasterized=True, alpha=0.6, label='Panel 4',zorder=0)
				ax3.scatter(x_35[up_check_35 == 0], y_35[up_check_35 == 0], color=c5, marker='+', rasterized=True, alpha=0.6, label='Panel 5',zorder=0)

			else:
				ax3.scatter(x_31, y_31, color=c1, marker='P', rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32, y_32, color=c2, marker='P', rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33, y_33, color=c3, marker='P', rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34, y_34, color=c4, marker='P', rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35, y_35, color=c5, marker='P', rasterized=True, alpha=0.4,zorder=0)

			if Median != 'None':
				ax3.errorbar(x3_med, y3_med, xerr=[x3_err_min, x3_err_max], yerr=[y3_err_min, y3_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
				ax3.scatter(x31_med, y31_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax3.scatter(x32_med, y32_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax3.scatter(x33_med, y33_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax3.scatter(x34_med, y34_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
				ax3.scatter(x35_med, y35_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

			ax3.set_xlim(xlim1, xlim2)
			ax3.set_ylim(ylim1, ylim2)
			ax3.set_xticks(xticks)
			ax3.set_yticks(yticks)
			ax3.set_ylabel(ylabel3)
			ax3.set_xlabel(' ')
			if X == 'Lbol':
				secax3 = ax3.secondary_xaxis('top', functions=(solar, ergs))
				secax3.set_xlabel(r' ')
			ax3.grid()

			plt.tight_layout()
			plt.savefig(f'/Users/connor_auge/Desktop/Final_Plots/{savestring}.pdf')
			plt.show()

	def Box_3panel(self, savestring, var, x, z, uv_slope, mir_slope1, mir_slope2, ulirg_x=None):
		plt.rcParams['font.size'] = 27
		plt.rcParams['axes.linewidth'] = 3.5
		plt.rcParams['xtick.major.size'] = 5.5
		plt.rcParams['xtick.major.width'] = 4.5
		plt.rcParams['ytick.major.size'] = 5.5
		plt.rcParams['ytick.major.width'] = 4.5

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		if var == 'Nh':
			ylabel = r'log N$_{\mathrm{H}}$'
			units = r' [cm$^{-2}$]'
			ylim1 = 19.5
			ylim2 = 24.5

		elif var == 'Lx':
			x -= np.log10(3.8E33)

			ylabel = r'log L$_{\mathrm{X}}$'
			units = r' [L$_{\odot}$]'
			ylim1 = 8
			ylim2 = 13

		elif var == 'Lbol':
			x -= np.log10(3.8E33)
			if any(ulirg_x) != None:
				ulirg_x -= np.log10(3.8E33)

			ylabel = r'log L$_{\mathrm{bol}}$'
			units = r' [L$_{\odot}$]'
			ylim1 = 9.5
			ylim2 = 14.5

		elif var == 'Lbol/Lx':
			ylabel = r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$'
			units = ''
			ylim1 = -1
			ylim2 = 4

		xticklabels = ['1','2','3','4','5']

		if any(ulirg_x) != None:
			if var == 'Nh':
				ylim2 += 1
			else:
				ylim1 -= 0.5
				ylim2 += 0.5
			xticklabels = ['1','2','3','4','5','GOALS']

		x1 = x[B1]
		x2 = x[B2]
		x3 = x[B3]
		x4 = x[B4]
		x5 = x[B5]

		z1 = z[B1]
		z2 = z[B2]
		z3 = z[B3]
		z4 = z[B4]
		z5 = z[B5]

		zlim1 = 0
		zlim2 = 0.6
		zlim3 = 0.9
		zlim4 = 1.2

		x11 = x1[(z1 >= zlim1) & (z1 <= zlim2)]
		x12 = x2[(z2 >= zlim1) & (z2 <= zlim2)]
		x13 = x3[(z3 >= zlim1) & (z3 <= zlim2)]
		x14 = x4[(z4 >= zlim1) & (z4 <= zlim2)]
		x15 = x5[(z5 >= zlim1) & (z5 <= zlim2)]

		x21 = x1[(z1 > zlim2) & (z1 <= zlim3)]
		x22 = x2[(z2 > zlim2) & (z2 <= zlim3)]
		x23 = x3[(z3 > zlim2) & (z3 <= zlim3)]
		x24 = x4[(z4 > zlim2) & (z4 <= zlim3)]
		x25 = x5[(z5 > zlim2) & (z5 <= zlim3)]

		x31 = x1[(z1 > zlim3) & (z1 <= zlim4)]
		x32 = x2[(z2 > zlim3) & (z2 <= zlim4)]
		x33 = x3[(z3 > zlim3) & (z3 <= zlim4)]
		x34 = x4[(z4 > zlim3) & (z4 <= zlim4)]
		x35 = x5[(z5 > zlim3) & (z5 <= zlim4)]



	

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		def solar(x):
				return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		fig = plt.figure(figsize=(30, 10))
		ax1 = plt.subplot(131, aspect='equal', adjustable='box')
		ax1.plot([1,2,3,4,5],[np.nanmean(x11),np.nanmean(x12),np.nanmean(x13),np.nanmean(x14),np.nanmean(x15)],color='k')
		ax1.boxplot(x11,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c1,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax1.boxplot(x12,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c2,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax1.boxplot(x13,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c3,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax1.boxplot(x14,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c4,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax1.boxplot(x15,positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c5,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		
		ax1.set_ylim(ylim1, ylim2)
		plt.gca().invert_xaxis()		
		ax1.set_ylabel(ylabel+units)
		ax1.set_xlabel('Panel Number')
		ax1.set_xticklabels(xticklabels)
		if any(ulirg_x) != None:
			print(ulirg_x)
			ax1.boxplot(ulirg_x,positions=[6],patch_artist=True,boxprops=dict(facecolor='gray', color='k'),medianprops=dict(color='k',lw=6,alpha=1),meanline=True,showmeans=False,meanprops=dict(color='k',lw=6),whiskerprops=dict(color='gray',lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)

		ax1.grid()

		ax2 = plt.subplot(132, aspect='equal', adjustable='box')
		ax2.plot([1,2,3,4,5],[np.nanmean(x21),np.nanmean(x22),np.nanmean(x23),np.nanmean(x24),np.nanmean(x25)],color='k')
		ax2.boxplot(x21,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c1,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax2.boxplot(x22,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c2,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax2.boxplot(x23,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c3,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax2.boxplot(x24,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c4,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax2.boxplot(x25,positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c5,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		
		ax2.set_ylim(ylim1, ylim2)
		plt.gca().invert_xaxis()		
		ax2.set_xlabel('Panel Number')
		ax2.set_xticklabels(xticklabels)
		ax2.set_yticklabels([])
		ax2.grid()

		ax3 = plt.subplot(133, aspect='equal', adjustable='box')
		ax3.plot([1,2,3,4,5],[np.nanmean(x31),np.nanmean(x32),np.nanmean(x33),np.nanmean(x34),np.nanmean(x35)],color='k')
		ax3.boxplot(x31,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c1,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax3.boxplot(x32,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c2,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax3.boxplot(x33,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c3,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax3.boxplot(x34,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c4,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)
		ax3.boxplot(x35, positions=[5], patch_artist=True, boxprops=dict(facecolor=c5, color='k'), medianprops=dict(color='k', lw=3, alpha=0), meanline=True, showmeans=True, meanprops=dict(
			color='k', lw=6), whiskerprops=dict(color=c5, lw=3, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]), showcaps=False, showfliers=True)
		
		ax3.set_ylim(ylim1, ylim2)
		plt.gca().invert_xaxis()		
		ax3.set_xlabel('Panel Number')
		ax3.set_xticklabels(xticklabels)
		ax3.set_yticklabels([])

		if var == 'Lbol':
			secax3 = ax3.secondary_yaxis('right', functions=(ergs, solar))
			secax3.set_ylabel(ylabel+' [erg/s]')
		elif var == 'Lx':
			secax3 = ax3.secondary_yaxis('right', functions=(ergs, solar))
			secax3.set_ylabel(ylabel+' [erg/s]')
		ax3.grid()

		plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/New_plots4/'+savestring+'.pdf')
		plt.show()

	def Box_1panel(self, savestring, var, x, uv_slope, mir_slope1, mir_slope2, ulirg_x=None, shape=None, L2=None):
		plt.rcParams['font.size'] = 30
		plt.rcParams['axes.linewidth'] = 3.5
		plt.rcParams['xtick.major.size'] = 5.5
		plt.rcParams['xtick.major.width'] = 4.5
		plt.rcParams['ytick.major.size'] = 5.5
		plt.rcParams['ytick.major.width'] = 4.5

		# B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		# B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		# B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		# B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		# B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]
		shape = shape[np.isfinite(x)]

		x = x[np.isfinite(x)]

		B1 = shape == 1
		B2 = shape == 2
		B3 = shape == 3
		B4 = shape == 4
		B5 = shape == 5

		if var == 'Nh':
			ylabel = r'log N$_{\mathrm{H}}$'
			units = r' [cm$^{-2}$]'
			ylim1 = 19.5
			ylim2 = 24.5

		elif var == 'Lone':
			x -= np.log10(3.8E33)

			ylabel = r'log L (1$\mu$m)'
			units = r' [L$_{\odot}$]'
			ylim1 = 8
			ylim2 = 13

		elif var == 'Lx':
			x -= np.log10(3.8E33)

			ylabel = r'log L$_{\mathrm{X}}$'
			units = r' [L$_{\odot}$]'
			ylim1 = 8
			ylim2 = 13

		elif var == 'Lbol':
			x -= np.log10(3.8E33)
			if any(ulirg_x) != None:
				ulirg_x -= np.log10(3.8E33)

			ylabel = r'log L$_{\mathrm{bol}}$'
			units = r' [L$_{\odot}$]'
			ylim1 = 9.5
			ylim2 = 14.5

		elif var == 'Lbol/Lx':
			ylabel = r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$'
			units = ''
			ylim1 = -1
			ylim2 = 4

		xticklabels = ['1','2','3','4','5']

		if any(ulirg_x) != None:
			if var == 'Nh':
				ylim2 += 1
			else:
				ylim1 -= 0.5
				ylim2 += 0.5
			xticklabels = ['1','2','3','4','5','GOALS']

		x1 = x[B1]
		x2 = x[B2]
		x3 = x[B3]
		x4 = x[B4]
		x5 = x[B5]

		if any(L2) != None:
			L2_1 = np.nanmean(L2[B1])
			L2_2 = np.nanmean(L2[B2])
			L2_3 = np.nanmean(L2[B3])
			L2_4 = np.nanmean(L2[B4])
			L2_5 = np.nanmean(L2[B5])
			L2_array = np.array([L2_1,L2_2,L2_3,L2_4,L2_5])
		else:
			L2_array = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

		print(L2_array)


		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		def solar(x):
				return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		print(x1,x2,x3,x4,x5)

		fig = plt.figure(figsize=(11, 11))
		ax1 = plt.subplot(111, aspect='equal', adjustable='box')
		ax1.plot([1,2,3,4,5],[np.nanmean(x1),np.nanmean(x2),np.nanmean(x3),np.nanmean(x4),np.nanmean(x5)],color='k',zorder=6)
		parts = ax1.violinplot([x1,x2,x3,x4,x5], positions=[1,2,3,4,5], showmeans=False, showmedians=False, showextrema=False)
		for pc in parts['bodies']:
			pc.set_facecolor('#D43F3A')
			pc.set_edgecolor('black')
			pc.set_alpha(1)
		# violin1 = ax1.violinplot(x1,positions=[1])
		# violin2 = ax1.violinplot(x2,positions=[2])
		# violin3 = ax1.violinplot(x3,positions=[3])
		# violin4 = ax1.violinplot(x4,positions=[4])
		# violin5 = ax1.violinplot(x5,positions=[5])
		# for v1 in violin1['bodies']:
		# 	v1.set_color(c1)
		# 	# v1.set_edgecolor(c1)
		# for v2 in violin2['bodies']:
		# 	v2.set_color(c2)
		# 	# v2.set_edgecolor(c2)
		# for v3 in violin3['bodies']:
		# 	v3.set_color(c3)
		# 	# v3.set_edgecolor(c3)
		# for v4 in violin4['bodies']:
		# 	v4.set_color(c4)
		# 	# v4.set_edgecolor(c4)
		# for v5 in violin5['bodies']:
		# 	v5.set_color(c5)
		# 	# v5.set_edgecolor(c5)

		# ax1.boxplot(x1,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c1,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True,zorder=1)
		# ax1.boxplot(x2,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c2,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True,zorder=2)
		# ax1.boxplot(x3,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c3,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True,zorder=3)
		# ax1.boxplot(x4,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c4,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True,zorder=4)
		# ax1.boxplot(x5,positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=6),whiskerprops=dict(color=c5,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True,zorder=5)
		ax1.set_ylim(ylim1, ylim2)
		plt.gca().invert_xaxis()		
		ax1.set_ylabel(ylabel+units)
		ax1.set_xlabel('Panel Number')
		ax1.set_xticklabels(xticklabels)
		if any(ulirg_x) != None:
			print(ulirg_x)
			ax1.boxplot(ulirg_x,positions=[6],patch_artist=True,boxprops=dict(facecolor='gray', color='k'),medianprops=dict(color='k',lw=6,alpha=1),meanline=True,showmeans=False,meanprops=dict(color='k',lw=6),whiskerprops=dict(color='gray',lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=True)

		if var == 'Lbol':
			secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
			secax1.set_ylabel(ylabel+' [erg/s]')
		elif var == 'Lone':
			secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
			secax1.set_ylabel(ylabel+' [erg/s]')
		elif var == 'Lx':
			secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
			secax1.set_ylabel(ylabel+' [erg/s]')

		# ax1.plot([1,2,3,4,5],L2_array,'P',ms=23,color='r',markeredgecolor='k',markeredgewidth=3,label='Duras+2020',zorder=7)
		ax1.grid()
		# plt.legend(fontsize=15)
		plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/Final_Plots/'+savestring+'.pdf')
		plt.show()

	def ratios_1panel(self, savestring, X, Y, Median, Nh, Lx, L, F1, f1, f2, f3, f4, uv_slope, mir_slope1, mir_slope2, up_check, ulirg_Nh=None, ulirg_Lx=None, ulirg_Flux=None, ulirg_F1=None):
		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		if X == 'Nh':
			x = np.log10(Nh)
			xlim1 = 19
			xlim2 = 24
			xlabel = r'log N$_{\mathrm{H}}$'
			xunits = r' [cm$^{-2}$]'

		elif X == 'Lx':
			x = Lx
			xlim1 = 42.5
			xlim2 = 46.5
			xlabel = r'log L$_{\mathrm{X}}$'
			xunits = ' [erg/s]'

		elif X == 'Lbol':
			x = L
			xlim1 = 43
			xlim2 = 48
			xlabel = r'log L$_{\mathrm{bol}}$'
			xunits = '[erg/s]'

		if Y == 'UV':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f1/lx)
			y_var = r'0.25$\mu$m'
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2
			ylim2 = 3
			yticks = [-2,-1,0,1,2,3]

		elif Y == 'MIR6':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f2/lx)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2
			ylim2 = 3
			yticks = [-2,-1,0,1,2,3]

		elif Y == 'MIR10':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f4/lx)
			ylabel = r'log L (10$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2
			ylim2 = 3

		elif Y == 'FIR':
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f3/lx)
			ylabel = r'log L (100$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2
			ylim2 = 3

		elif Y == 'UV/MIR6':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_4*F1
			y = np.log10(f1/f2)
			ylabel = r'log L (0.25$\mu$m)/ L (10$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'UV/MIR10':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f1/f4)
			ylabel = r'log L (0.25$\mu$m)/ L (10$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'UV/FIR':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f1/f3)
			ylabel = r'log L (0.25$\mu$m)/ L (100$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'MIR6/FIR':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f2/f3)
			ylabel = r'log L (10$\mu$m)/ L (100$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'MIR10/FIR':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f4/f3)
			ylabel = r'log L (10$\mu$m)/ L (100$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'Lbol/Lx':
			l = np.asarray([10**i for i in L])
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(l/lx)
			ylabel = r'log L$_{\mathrm{bol}}$/ L$_{\mathrm{X}}$'
			ylim1 = 0 
			ylim2 = 4
			yticks = [0,1,2,3,4]

		elif Y == 'UV/Lbol':
			l = np.asarray([10**i for i in L])
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			y = np.log10(f1/l)
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'MIR6/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			y = np.log10(f2/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'MIR10/Lbol':
			l = np.asarray([10**i for i in L])
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f4/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'FIR/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f3/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		else:
			print('Specify Y variable')
			return

		if any(ulirg_Flux) != None:
			u_f = np.asarray([10**i for i in ulirg_Flux])
			ulx = np.asarray([10**i for i in ulirg_Lx])
			ulirg_Flux = np.log10((u_f*ulirg_F1)/ulx)

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		x_11 = x[B1]
		y_11 = y[B1]
		up_check_11 = up_check[B1]
		x_12 = x[B2]
		y_12 = y[B2]
		up_check_12 = up_check[B2]
		x_13 = x[B3]
		y_13 = y[B3]
		up_check_13 = up_check[B3]
		x_14 = x[B4]
		y_14 = y[B4]
		up_check_14 = up_check[B4]
		x_15 = x[B5]
		y_15 = y[B5]
		up_check_15 = up_check[B5]

		if Median == 'Bins':
			c1m = c1
			c2m = c2
			c3m = c3
			c4m = c4
			c5m = c5

			# Median
			x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
			x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
			x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
			x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
			x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])

			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
			x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
			x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
			x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
			x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
			x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
			x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
			x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
			x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
		
		elif Median == 'X-axis':

			c1m = 'gray'
			c2m = 'gray'
			c3m = 'gray'
			c4m = 'gray'
			c5m = 'gray'

			b1 = (Lx > 43)&(Lx < 43.5)
			b2 = (Lx > 43.5)&(Lx < 44)
			b3 = (Lx > 44)&(Lx < 44.5)
			b4 = (Lx > 44.5)&(Lx < 45)
			b5 = (Lx > 45)

			# Median
			x11_med, y11_med = np.nanmedian(x[b1]), np.nanmedian(y[b1])
			x12_med, y12_med = np.nanmedian(x[b2]), np.nanmedian(y[b2])
			x13_med, y13_med = np.nanmedian(x[b3]), np.nanmedian(y[b3])
			x14_med, y14_med = np.nanmedian(x[b4]), np.nanmedian(y[b4])
			x15_med, y15_med = np.nanmedian(x[b5]), np.nanmedian(y[b5])

			x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])

			# 25 Percentile 
			x11_25per, y11_25per = np.nanpercentile(x[b1], 25), np.nanpercentile(y[b1], 25)
			x12_25per, y12_25per = np.nanpercentile(x[b2], 25), np.nanpercentile(y[b2], 25)
			x13_25per, y13_25per = np.nanpercentile(x[b3], 25), np.nanpercentile(y[b3], 25)
			x14_25per, y14_25per = np.nanpercentile(x[b4], 25), np.nanpercentile(y[b4], 25)
			x15_25per, y15_25per = np.nanpercentile(x[b5], 25), np.nanpercentile(y[b5], 25)

			x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
			
			# 75 Percentile 
			x11_75per, y11_75per = np.nanpercentile(x[b1], 75), np.nanpercentile(y[b1], 75)
			x12_75per, y12_75per = np.nanpercentile(x[b2], 75), np.nanpercentile(y[b2], 75)
			x13_75per, y13_75per = np.nanpercentile(x[b3], 75), np.nanpercentile(y[b3], 75)
			x14_75per, y14_75per = np.nanpercentile(x[b4], 75), np.nanpercentile(y[b4], 75)
			x15_75per, y15_75per = np.nanpercentile(x[b5], 75), np.nanpercentile(y[b5], 75)

			x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)
		
		fig = plt.figure(figsize=(9,9))
		ax1 = plt.subplot(111, aspect='equal', adjustable='box')
		if 'Lbol' in Y:
			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=3, color=c1, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=3, color=c2, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=3, color=c3, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=3, color=c4, rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=3, color=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		else:
			ax1.scatter(x_11, y_11, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		if Median != 'None':
			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		ax1.set_xlim(xlim1, xlim2)
		if any(ulirg_Nh) != None:
			print(ulirg_Nh)
			print(ulirg_Flux)
			ax1.scatter(ulirg_Nh,ulirg_Flux,color='k',marker='s',s=75,label='GOALS AGN')
			ax1.set_xlim(19.9, 24.9)
		ax1.set_ylim(ylim1, ylim2)
		ax1.set_yticks(yticks)
		ax1.set_ylabel(ylabel)
		ax1.set_xlabel(xlabel+xunits)
		if X != 'Nh':
			secax3 = ax1.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(xlabel+r' [L$_{\odot}$]')
		ax1.legend(fontsize=14)
		ax1.grid()
		plt.savefig('/Users/connor_auge/Desktop/New_plots4/'+savestring+'.pdf')
		plt.show()
				
	def scatter_1panel(self, savestring, X, Y, Norm, Median, Nh, Lx, L, F1, f1, f2, f3, f4, uv_slope, mir_slope1, mir_slope2, up_check, shape=None, durras=False):
		plt.rcParams['font.size'] = 22
		plt.rcParams['axes.linewidth'] = 2
		plt.rcParams['xtick.major.size'] = 4
		plt.rcParams['xtick.major.width'] = 3
		plt.rcParams['ytick.major.size'] = 4
		plt.rcParams['ytick.major.width'] = 3

		B1 = shape == 1
		B2 = shape == 2
		B3 = shape == 3
		B4 = shape == 4
		B5 = shape == 5

		# B1 = np.where(np.logical_and(uv_slope < -0.3, mir_slope1 >= -0.1))[0]
		# B2 = np.where(np.logical_and(np.logical_and(uv_slope >= -0.3, uv_slope <=0.2),mir_slope1 >= -0.2))[0]	
		# B3 = np.where(np.logical_and(uv_slope > 0.2, mir_slope1 >= -0.2))[0]
		# B4 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 > 0.0)))[0]
		# B5 = np.where(np.logical_and(uv_slope > -0.05, np.logical_and(mir_slope1 < -0.2,mir_slope2 <= 0.0)))[0]

		if X == 'Nh':
			tax_check = False
		elif Norm == 'Both':
			tax_check = False
		elif Norm == 'X-axis':
			tax_check = False
		elif '/' in X:
			tax_check = False
		else:
			tax_check = True
		 

		if X == 'Nh':
			x = np.log10(Nh)
			xlim1 = 19
			xlim2 = 24
			xlabel = r'log N$_{\mathrm{H}}$'
			xunits = r' [cm$^{-2}$]'

		elif X == 'Lx':
			x = Lx
			xlim1 = 42.5
			xlim2 = 46
			xlabel = r'log L$_{\mathrm{X}}$'
			xunits = ' [erg/s]'
			xticks = [43,44,45,46]

		elif X == 'Lbol':
			x = L
			# xlim1 = 43.75
			# xlim2 = 46.75
			xlabel = r'log L$_{\mathrm{bol}}$'
			xunits = '[erg/s]'
			# xticks = [44.5,45.5,46.5]
			xlim1 = 43.75
			xlim2 = 46.75
			xticks = [44.5,45.5,46.5]

		elif X == 'UV':
			# f_1 = np.asarray([10**i for i in f1])
			# f1 = f_1*F1
			# x = np.log10(f1)
			x = f1
			xlabel = r'log L (0.25$\mu$m)'
			xunits = ' [erg/s]'
			xlim1 = 41.5
			xlim2 = 46.5
			xticks = [42,43,44,45,46]
			

		elif X == 'MIR6':
			# f_2 = np.asarray([10**i for i in f2])
			# f2 = f_2*F1
			# x = np.log10(f2)
			x = f2
			xlabel = r'log L (6$\mu$m)'
			xunits = ' [erg/s]'
			xlim1 = 42.5
			xlim2 = 46
			xticks = [43, 44, 45, 46]

		elif X == 'FIR':
			# f_3 = np.asarray([10**i for i in f3])
			# f3 = f_3*F1
			# x = np.log10(f3)
			x = f3
			xlabel = r'log L (100$\mu$m)'
			xlim1 = 42
			xlim2 = 46.5
			xticks = [43,44,45,46]
			xunits = ' [erg/s]'

		elif X == 'MIR6/Lx':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			lx = np.asarray([10**i for i in Lx])
			x = np.log10(f2/lx)
			xlabel = r'log L (6$\mu$m)/ L$_{\mathrm{X}}$'
			xlim1 = -2
			xlim2 = 2
			xticks = [-2,-1,0,1,2]
			xunits = ''

		elif X == 'FIR/Lx':
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			lx = np.asarray([10**i for i in Lx])
			x = np.log10(f3/lx)
			xlabel = r'log L (100$\mu$m)/ L$_{\mathrm{X}}$'
			xlim1 = -1.5
			xlim2 = 2.5
			xticks = [-2, -1, 0, 1, 2]
			xunits = ''

		if Y == 'UV':
			# f_1 = np.asarray([10**i for i in f1])
			# f1 = f_1*F1
			# y = np.log10(f1)
			y_var = r'0.25$\mu$m'
			ylabel = r'log L (0.25$\mu$m)'
			ylim1 = 42.5
			ylim2 = 46.5
			yticks = [43,44,45,46]

		elif Y == 'MIR6':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			y = np.log10(f2)
			ylabel = r'log L (6$\mu$m)'
			ylim1 = 42.5
			ylim2 = 46.5
			yticks = [43, 44, 45, 46]

		elif Y == 'MIR10':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f4)
			ylabel = r'log L (10$\mu$m)'
			ylim1 = 42.5
			ylim2 = 46.5
			yticks = [43, 44, 45, 46]

		elif Y == 'FIR':
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f3)
			ylabel = r'log L (100$\mu$m)'
			ylim1 = 42
			ylim2 = 46.5
			yticks = [42, 43, 44, 45, 46]

		elif Y == 'UV/MIR6':
			y = f1 - f2
			ylabel = r'log L (0.25$\mu$m)/ L (6$\mu$m)'
			ylim1 = -2.5
			ylim2 = 1.
			yticks = [-2,-1,-2,0,1]

		elif Y == 'UV/MIR10':
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f1/f4)
			ylabel = r'log L (0.25$\mu$m)/ L (10$\mu$m)'
			ylim1 = -3
			ylim2 = 2
			

		elif Y == 'UV/FIR':
			# f_1 = np.asarray([10**i for i in f1])
			# f1 = f_1*F1
			# f_3 = np.asarray([10**i for i in f3])
			# f3 = f_3*F1
			# y = np.log10(f1/f3)
			y = f1 - f3
			ylabel = r'log L (0.25$\mu$m)/ L (100$\mu$m)'
			ylim1 = -2.25
			ylim2 = 1.75
			yticks = [-2,-1,0,1,2]

		elif Y == 'FIR/UV':
			y = f3 - f1
			ylabel = r'log L (100$\mu$m)/ L (0.25$\mu$m)'
			ylim2 = 3
			ylim1 = -2
			yticks = [-2,-1,0,1,2,3]

		elif Y == 'UV/Lx':
			# f_1 = np.asarray([10**i for i in f1])
			# f1 = f_1*F1
			# lx = np.asarray([10**i for i in Lx])
			# y = np.log10(f1/lx)
			y = f1 - Lx
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -1.75
			ylim2 = 1.75
			yticks = [-2,-1,0,1,2]

		elif Y == 'MIR6/Lx':
			# f_2 = np.asarray([10**i for i in f2])
			# f2 = f_2*F1
			# lx = np.asarray([10**i for i in Lx])
			# y = np.log10(f2/lx)
			y = f2 - Lx
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -1.5
			ylim2 = 2
			yticks = [-1,0,1,2]

		elif Y == 'FIR/Lx':
			# f_3 = np.asarray([10**i for i in f3])
			# f3 = f_3*F1
			# lx = np.asarray([10**i for i in Lx])
			# y = np.log10(f3/lx)
			y = f3 - Lx
			ylabel = r'log L (100$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -1.75
			ylim2 = 2.25
			yticks = [-1,0,1,2]

		elif Y == 'MIR6/FIR':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f2/f3)
			ylabel = r'log L (6$\mu$m)/ L (100$\mu$m)'
			ylim1 = -3
			ylim2 = 2
			yticks = [-2, -1, 0, 1, 2]

		elif Y == 'FIR/MIR6':
			y = f3 - f2
			ylabel = r'log L (100$\mu$m)/ L (6$\mu$m)'
			ylim1 = -2
			ylim2 = 2
			yticks = [-2,-1,0,1,2]

		elif Y == 'MIR10/FIR':
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			f_3 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f4/f3)
			ylabel = r'log L (10$\mu$m)/ L (100$\mu$m)'
			ylim1 = -3
			ylim2 = 2

		elif Y == 'Lbol/Lx':

			# l = np.asarray([10**i for i in L])
			# lx = np.asarray([10**i for i in Lx])
			# y = l/lx
			y = L - Lx
			# y = np.asarray([10**i for i in yi])
			# y = np.log10(l/lx)
			ylabel = r'log L$_{\mathrm{bol}}$/ L$_{\mathrm{X}}$'
			ylim1 = 0 
			ylim2 = 3
			yticks = [0,1,2,3]
			# ylim1 = 1
			# ylim2 = 3000
			# yticks = [1,10,100,1000]

		elif Y == 'Lx/Lbol':
			l = np.asarray([10**i for i in L])
			lx = np.asarray([10**i for i in Lx])
			# y = l/lx
			yi = Lx - L
			# y = np.asarray([10**i for i in yi])
			y = np.log10(lx/l)
			ylabel = r'log L$_{\mathrm{X}}$/ L$_{\mathrm{bol}}$'
			# ylim1 = -3
			# ylim2 = 0.5
			# yticks = [-3,-2,-1,0]
			ylim1 = -3
			ylim2 = 0.5
			yticks = [-3,-2,-1,0]
			# ylim1 = 1
			# ylim2 = 3000
			# yticks = [1,10,100,1000]

		elif Y == 'UV/Lbol':
			l = np.asarray([10**i for i in L])
			f_1 = np.asarray([10**i for i in f1])
			f1 = f_1*F1
			y = np.log10(f1/l)
			ylabel = r'log L (0.25$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'MIR6/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			y = np.log10(f2/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'MIR10/Lbol':
			l = np.asarray([10**i for i in L])
			f_4 = np.asarray([10**i for i in f4])
			f4 = f_4*F1
			y = np.log10(f4/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		elif Y == 'FIR/Lbol':
			l = np.asarray([10**i for i in L])
			f_2 = np.asarray([10**i for i in f3])
			f3 = f_3*F1
			y = np.log10(f3/l)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{bol}}$'
			ylim1 = -3
			ylim2 = 1

		else:
			print('Specify Y variable')
			return

		if Norm == 'Both':
			if '/' in Y:
				print('Cannot Normalize Y-axis')
				return
			else:
				x_s = np.asarray([10**i for i in x])
				x = np.log10(x_s/F1)
				y_s = np.asarray([10**i for i in y])
				y = np.log10(y_s/F1)
				xlabel = xlabel+r'/L (1$\mu$m)'
				ylabel = ylabel+r'/L (1$\mu$m)'
				ylim1 = -2.5
				ylim2 = 1.5
				xlim1 = -2
				xlim2 = 2
				yticks = [-2, -1, 0, 1]

		elif Norm == 'X-axis':
			x_s = np.asarray([10**i for i in x])
			x = np.log10(x_s/F1)
			xlabel = xlabel+r'/L (1$\mu$m)'
			xlim1 = -2
			xlim2 = 2

		elif Norm == 'Y-axis':
			if '/' in Y:
				print('Cannot Normalize Y-axis')
				return
			else:
				y_s = np.asarray([10**i for i in y])
				y = np.log10(y_s/F1)
				ylabel = ylabel+r'/L (1$\mu$m)'
				ylim1 = -2.5
				ylim2 = 1.5
				yticks = [-2, -1, 0, 1]

		print('x: ',x)
		print('y: ',y)


		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		x_11 = x[B1]
		y_11 = y[B1]
		up_check_11 = up_check[B1]
		x_12 = x[B2]
		y_12 = y[B2]
		up_check_12 = up_check[B2]
		x_13 = x[B3]
		y_13 = y[B3]
		up_check_13 = up_check[B3]
		x_14 = x[B4]
		y_14 = y[B4]
		up_check_14 = up_check[B4]
		x_15 = x[B5]
		y_15 = y[B5]
		up_check_15 = up_check[B5]

		# if Median == 'Bins':
		c1m = c1
		c2m = c2
		c3m = c3
		c4m = c4
		c5m = c5

		# Median
		x11_med, y11_med = np.nanmedian(x_11), np.nanmedian(y_11)
		x12_med, y12_med = np.nanmedian(x_12), np.nanmedian(y_12)
		x13_med, y13_med = np.nanmedian(x_13), np.nanmedian(y_13)
		x14_med, y14_med = np.nanmedian(x_14), np.nanmedian(y_14)
		x15_med, y15_med = np.nanmedian(x_15), np.nanmedian(y_15)

		# print('here: ')
		# print('med: ',y11_med)
		# print(y_11)
		# print(up_check_11)
		# print(y_11[up_check_11 == 0])
		# print('med: ',np.nanmedian(y_11[up_check_11 == 0]))

		# if X == 'FIR' or Y in 'FIR':
		# if 'Lbol' in Y:
			# print('check yes1')
		# x11_med, y11_med = np.nanmedian(x_11[up_check_11 == 0]), np.nanmedian(y_11[up_check_11 == 0])
		# x12_med, y12_med = np.nanmedian(x_12[up_check_12 == 0]), np.nanmedian(y_12[up_check_12 == 0])
		# x13_med, y13_med = np.nanmedian(x_13[up_check_13 == 0]), np.nanmedian(y_13[up_check_13 == 0])
		# x14_med, y14_med = np.nanmedian(x_14[up_check_14 == 0]), np.nanmedian(y_14[up_check_14 == 0])
		# x15_med, y15_med = np.nanmedian(x_15[up_check_15 == 0]), np.nanmedian(y_15[up_check_15 == 0])
			# print(y11_med)

		x1_med, y1_med = np.asarray([x11_med,x12_med,x13_med,x14_med,x15_med]), np.asarray([y11_med,y12_med,y13_med,y14_med,y15_med])

		# # 25 Percentile 
		x11_25per, y11_25per = np.nanpercentile(x_11, 25), np.nanpercentile(y_11, 25)
		x12_25per, y12_25per = np.nanpercentile(x_12, 25), np.nanpercentile(y_12, 25)
		x13_25per, y13_25per = np.nanpercentile(x_13, 25), np.nanpercentile(y_13, 25)
		x14_25per, y14_25per = np.nanpercentile(x_14, 25), np.nanpercentile(y_14, 25)
		x15_25per, y15_25per = np.nanpercentile(x_15, 25), np.nanpercentile(y_15, 25)

		# if X == 'FIR' or Y in 'FIR':
		# if 'Lbol' in Y:
		# x11_25per, y11_25per = np.nanpercentile(x_11[up_check_11 == 0], 25), np.nanpercentile(y_11[up_check_11 == 0], 25)
		# x12_25per, y12_25per = np.nanpercentile(x_12[up_check_12 == 0], 25), np.nanpercentile(y_12[up_check_12 == 0], 25)
		# x13_25per, y13_25per = np.nanpercentile(x_13[up_check_13 == 0], 25), np.nanpercentile(y_13[up_check_13 == 0], 25)
		# x14_25per, y14_25per = np.nanpercentile(x_14[up_check_14 == 0], 25), np.nanpercentile(y_14[up_check_14 == 0], 25)
		# x15_25per, y15_25per = np.nanpercentile(x_15[up_check_15 == 0], 25), np.nanpercentile(y_15[up_check_15 == 0], 25)

		x1_err_min, y1_err_min = x1_med - np.asarray([x11_25per,x12_25per,x13_25per,x14_25per,x15_25per]), y1_med - np.asarray([y11_25per,y12_25per,y13_25per,y14_25per,y15_25per])
		# 75 Percentile 
		x11_75per, y11_75per = np.nanpercentile(x_11, 75), np.nanpercentile(y_11, 75)
		x12_75per, y12_75per = np.nanpercentile(x_12, 75), np.nanpercentile(y_12, 75)
		x13_75per, y13_75per = np.nanpercentile(x_13, 75), np.nanpercentile(y_13, 75)
		x14_75per, y14_75per = np.nanpercentile(x_14, 75), np.nanpercentile(y_14, 75)
		x15_75per, y15_75per = np.nanpercentile(x_15, 75), np.nanpercentile(y_15, 75)

		# if X == 'FIR' or Y in 'FIR':
		# if 'Lbol' in Y:
		# x11_75per, y11_75per = np.nanpercentile(x_11[up_check_11 == 0], 75), np.nanpercentile(y_11[up_check_11 == 0], 75)
		# x12_75per, y12_75per = np.nanpercentile(x_12[up_check_12 == 0], 75), np.nanpercentile(y_12[up_check_12 == 0], 75)
		# x13_75per, y13_75per = np.nanpercentile(x_13[up_check_13 == 0], 75), np.nanpercentile(y_13[up_check_13 == 0], 75)
		# x14_75per, y14_75per = np.nanpercentile(x_14[up_check_14 == 0], 75), np.nanpercentile(y_14[up_check_14 == 0], 75)
		# x15_75per, y15_75per = np.nanpercentile(x_15[up_check_15 == 0], 75), np.nanpercentile(y_15[up_check_15 == 0], 75)

		x1_err_max, y1_err_max = np.asarray([x11_75per,x12_75per,x13_75per,x14_75per,x15_75per]) - x1_med, np.asarray([y11_75per,y12_75per,y13_75per,y14_75per,y15_75per]) - y1_med
		
		# elif Median == 'X-axis':

		c1mx = 'k'
		c2mx = 'k'
		c3mx = 'k'
		c4mx = 'k'
		c5mx = 'k'

		# b1 = (Lx > 43)&(Lx < 43.5)
		# b2 = (Lx > 43.5)&(Lx < 44)
		# b3 = (Lx > 44)&(Lx < 44.5)
		# b4 = (Lx > 44.5)&(Lx < 45)
		# b5 = (Lx > 45)

		if Norm == 'Both':
			b1 = (x < -0.5)
			b2 = (x > -0.5)&(x < 0)
			b3 = (x > 0)&(x < 0.5)
			b4 = (x > 0.5)&(x < 1)
			b5 = (x > 1.5)

		elif Norm == 'X-axis':
			b1 = (x < -0.5)
			b2 = (x > -0.5)&(x < 0)
			b3 = (x > 0)&(x < 0.5)
			b4 = (x > 0.5)&(x < 1)
			b5 = (x > 1.5)

		elif (X == 'Lbol') & (Median == 'Both'):
			b1 = (x > 44.5) & (x < 45)
			b2 = (x > 45) & (x < 45.5)
			b3 = (x > 45.5) & (x < 46)
			b4 = (x > 46) & (x < 46.5)
			b5 = (x > 46.5)

		else:
			b1 = (x > 43) & (x < 43.5)
			b2 = (x > 43.5) & (x < 44)
			b3 = (x > 44) & (x < 44.5)
			b4 = (x > 44.5) & (x < 45)
			b5 = (x > 45)

		Lx_duras = Lit_functions.Durras_Lbol(np.arange(42,48,0.25),typ='Lbol')
		Lx_Hopkins = Lit_functions.Hopkins_Lbol(np.arange(42,48,0.25),band='Lx')

		Median
		x11_medx, y11_medx = np.nanmedian(x[b1]), np.nanmedian(y[b1])
		x12_medx, y12_medx = np.nanmedian(x[b2]), np.nanmedian(y[b2])
		x13_medx, y13_medx = np.nanmedian(x[b3]), np.nanmedian(y[b3])
		x14_medx, y14_medx = np.nanmedian(x[b4]), np.nanmedian(y[b4])
		x15_medx, y15_medx = np.nanmedian(x[b5]), np.nanmedian(y[b5])

		# if X == 'FIR' or 'FIR' in Y:
		# # # if 'Lbol' in Y:
		# 	print('check yes1')
		# 	x11_medx, y11_medx = np.nanmedian(x[b1][up_check[b1] == 0]), np.nanmedian(y[b1][up_check[b1] == 0])
		# 	x12_medx, y12_medx = np.nanmedian(x[b2][up_check[b2] == 0]), np.nanmedian(y[b2][up_check[b2] == 0])
		# 	x13_medx, y13_medx = np.nanmedian(x[b3][up_check[b3] == 0]), np.nanmedian(y[b3][up_check[b3] == 0])
		# 	x14_medx, y14_medx = np.nanmedian(x[b4][up_check[b4] == 0]), np.nanmedian(y[b4][up_check[b4] == 0])
		# 	x15_medx, y15_medx = np.nanmedian(x[b5][up_check[b5] == 0]), np.nanmedian(y[b5][up_check[b5] == 0])
				

		x1_medx, y1_medx = np.asarray([x11_medx,x12_medx,x13_medx,x14_medx,x15_medx]), np.asarray([y11_medx,y12_medx,y13_medx,y14_medx,y15_medx])

		# 25 Percentile 
		x11_25perx, y11_25perx = np.nanpercentile(x[b1], 25), np.nanpercentile(y[b1], 25)
		x12_25perx, y12_25perx = np.nanpercentile(x[b2], 25), np.nanpercentile(y[b2], 25)
		x13_25perx, y13_25perx = np.nanpercentile(x[b3], 25), np.nanpercentile(y[b3], 25)
		x14_25perx, y14_25perx = np.nanpercentile(x[b4], 25), np.nanpercentile(y[b4], 25)
		x15_25perx, y15_25perx = np.nanpercentile(x[b5], 25), np.nanpercentile(y[b5], 25)

		# if X == 'FIR' or 'FIR' in Y:
		# # # if 'Lbol' in Y:
		# 	x11_25perx, y11_25perx = np.nanpercentile(x[b1][up_check[b1] == 0], 25), np.nanpercentile(y[b1][up_check[b1] == 0], 25)
		# 	x12_25perx, y12_25perx = np.nanpercentile(x[b2][up_check[b2] == 0], 25), np.nanpercentile(y[b2][up_check[b2] == 0], 25)
		# 	x13_25perx, y13_25perx = np.nanpercentile(x[b3][up_check[b3] == 0], 25), np.nanpercentile(y[b3][up_check[b3] == 0], 25)
		# 	x14_25perx, y14_25perx = np.nanpercentile(x[b4][up_check[b4] == 0], 25), np.nanpercentile(y[b4][up_check[b4] == 0], 25)
		# 	x15_25perx, y15_25perx = np.nanpercentile(x[b5][up_check[b5] == 0], 25), np.nanpercentile(y[b5][up_check[b5] == 0], 25)

		x1_err_minx, y1_err_minx = x1_medx - np.asarray([x11_25perx,x12_25perx,x13_25perx,x14_25perx,x15_25perx]), y1_medx - np.asarray([y11_25perx,y12_25perx,y13_25perx,y14_25perx,y15_25perx])
			
		# 75 Percentile 
		x11_75perx, y11_75perx = np.nanpercentile(x[b1], 75), np.nanpercentile(y[b1], 75)
		x12_75perx, y12_75perx = np.nanpercentile(x[b2], 75), np.nanpercentile(y[b2], 75)
		x13_75perx, y13_75perx = np.nanpercentile(x[b3], 75), np.nanpercentile(y[b3], 75)
		x14_75perx, y14_75perx = np.nanpercentile(x[b4], 75), np.nanpercentile(y[b4], 75)
		x15_75perx, y15_75perx = np.nanpercentile(x[b5], 75), np.nanpercentile(y[b5], 75)

		# if X == 'FIR' or 'FIR' in Y:
		# # # if 'Lbol' in Y:
		# 	x11_75perx, y11_75perx = np.nanpercentile(x[b1][up_check[b1] == 0], 75), np.nanpercentile(y[b1][up_check[b1] == 0], 75)
		# 	x12_75perx, y12_75perx = np.nanpercentile(x[b2][up_check[b2] == 0], 75), np.nanpercentile(y[b2][up_check[b2] == 0], 75)
		# 	x13_75perx, y13_75perx = np.nanpercentile(x[b3][up_check[b3] == 0], 75), np.nanpercentile(y[b3][up_check[b3] == 0], 75)
		# 	x14_75perx, y14_75perx = np.nanpercentile(x[b4][up_check[b4] == 0], 75), np.nanpercentile(y[b4][up_check[b4] == 0], 75)
		# 	x15_75perx, y15_75perx = np.nanpercentile(x[b5][up_check[b5] == 0], 75), np.nanpercentile(y[b5][up_check[b5] == 0], 75)

		x1_err_maxx, y1_err_maxx = np.asarray([x11_75perx,x12_75perx,x13_75perx,x14_75perx,x15_75perx]) - x1_medx, np.asarray([y11_75perx,y12_75perx,y13_75perx,y14_75perx,y15_75perx]) - y1_medx

		def solar(x):
			return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)

		# fig = plt.figure(figsize=(9,9))
		fig = plt.figure(figsize=(9,9))
		ax1 = plt.subplot(111)#, aspect='equal', adjustable='box')
		if 'FIR' in X:
			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors='gray', rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=0, color='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=0, color='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=0, color='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=0, color='gray', rasterized=True, alpha=0.8,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=0, color='gray', rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)		

		# elif 'Lbol' in Y:
		# 	ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

		# 	ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=3, color=c1, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=3, color=c2, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=3, color=c3, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=3, color=c4, rasterized=True, alpha=0.8,zorder=0)
		# 	ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=3, color=c5, rasterized=True, alpha=0.8,zorder=0)

		# 	ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
		# 	ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
		# 	ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
		# 	ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
		# 	ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		elif 'FIR' in Y:
			# ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.8,zorder=0)
			# ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.8,zorder=0)
			# ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.8,zorder=0)
			# ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.8,zorder=0)
			# ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.8,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=11, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=11, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=11, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=11, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=11, color='gray', rasterized=True, alpha=0.5,zorder=0)

			ax1.scatter(x_11[up_check_11 == 1], y_11[up_check_11 == 1], marker=2, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_12[up_check_12 == 1], y_12[up_check_12 == 1], marker=2, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_13[up_check_13 == 1], y_13[up_check_13 == 1], marker=2, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_14[up_check_14 == 1], y_14[up_check_14 == 1], marker=2, color='gray', rasterized=True, alpha=0.5,zorder=0)
			ax1.scatter(x_15[up_check_15 == 1], y_15[up_check_15 == 1], marker=2, color='gray', rasterized=True, alpha=0.5,zorder=0)

			ax1.scatter(x_11[up_check_11 == 0], y_11[up_check_11 == 0], color=c1, marker='P', lw=0, rasterized=True, s=55, alpha=0.85, label='Panel 1',zorder=0)
			ax1.scatter(x_12[up_check_12 == 0], y_12[up_check_12 == 0], color=c2, marker='P', lw=0, rasterized=True, s=55, alpha=0.85, label='Panel 2',zorder=0)
			ax1.scatter(x_13[up_check_13 == 0], y_13[up_check_13 == 0], color=c3, marker='P', lw=0, rasterized=True, s=55, alpha=0.85, label='Panel 3',zorder=0)
			ax1.scatter(x_14[up_check_14 == 0], y_14[up_check_14 == 0], color=c4, marker='P', lw=0, rasterized=True, s=55, alpha=0.85, label='Panel 4',zorder=0)
			ax1.scatter(x_15[up_check_15 == 0], y_15[up_check_15 == 0], color=c5, marker='P', lw=0, rasterized=True, s=55, alpha=0.85, label='Panel 5',zorder=0)

		else:
			ax1.scatter(x_11, y_11, color=c1, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='P', lw=0, rasterized=True, alpha=0.8, label='Panel 5',zorder=0)

		# if Median != 'None':
		if Median == 'Bins':
			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True)

		elif Median == 'X-axis':
			ax1.errorbar(x1_medx, y1_medx, xerr=[x1_err_minx, x1_err_maxx], yerr=[y1_err_minx, y1_err_maxx],mfc=c1mx, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_medx, y11_medx, color=c1mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x12_medx, y12_medx, color=c2mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x13_medx, y13_medx, color=c3mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x14_medx, y14_medx, color=c4mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x15_medx, y15_medx, color=c5mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
		
		elif Median == 'Both':
			ax1.errorbar(x1_medx, y1_medx, xerr=[x1_err_minx, x1_err_maxx], yerr=[y1_err_minx, y1_err_maxx], mfc=c1mx, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_medx, y11_medx, color=c1mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x12_medx, y12_medx, color=c2mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x13_medx, y13_medx, color=c3mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x14_medx, y14_medx, color=c4mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)
			ax1.scatter(x15_medx, y15_medx, color=c5mx, marker='s', s=150, edgecolor='k', linewidth=2, alpha=0.75, rasterized=True)

			ax1.errorbar(x1_med, y1_med, xerr=[x1_err_min, x1_err_max], yerr=[y1_err_min, y1_err_max], mfc=c1m, ecolor='k', capsize=5, fmt='none', rasterized=True,zorder=1)
			ax1.scatter(x11_med, y11_med, color=c1m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True,alpha=0.75)
			ax1.scatter(x12_med, y12_med, color=c2m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True,alpha=0.75)
			ax1.scatter(x13_med, y13_med, color=c3m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True,alpha=0.75)
			ax1.scatter(x14_med, y14_med, color=c4m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True,alpha=0.75)
			ax1.scatter(x15_med, y15_med, color=c5m, marker='o', s=150, edgecolor='k', linewidth=2, rasterized=True,alpha=0.75)

		elif Median == 'None':
			ax1.plot([np.nan],[np.nan])

		# if durras:
			# ax1.plot(np.linspace(40,50,100),self.Durras(np.linspace(40,50,100),typ='Lbol'))
		# ax1.plot(np.arange(42,48,0.25),np.log10(Lx_duras),color='r',label='Duras+2020')
		# ax1.plot(np.arange(42,48,0.25),np.log10(Lx_Hopkins),color='b',label='Hopkins+2007')

		# ax1.set_yscale('log')
		ax1.set_xlim(xlim1,xlim2)
		ax1.set_ylim(ylim1,ylim2)
		ax1.set_xticks(xticks)
		ax1.set_yticks(yticks)
		ax1.set_ylabel(ylabel)
		if Norm == 'X-axis':
			ax1.set_xlabel(xlabel)
		elif Norm == 'Both':
			ax1.set_xlabel(xlabel)	
		else:
			ax1.set_xlabel(xlabel+xunits)
		if tax_check:
			secax3 = ax1.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(xlabel+r' [L$_{\odot}$]')
		ax1.legend(fontsize=14)
		ax1.grid()
		# plt.ylim(1,3000)
		# plt.xlim(42,49)
		plt.savefig('/Users/connor_auge/Desktop/Final_Plots/'+savestring+'.pdf')
		plt.show()


	def Durras(self,l,typ):
		
		if typ == 'Lx':
			a = 15.33
			b = 11.48
			c = 15.20
			# kx = a*(1+(np.log10(lx)/b)**c)
		elif typ == 'Lbol':
			a = 10.96
			b = 11.93
			c = 17.79
		kx = a*(1+((l - np.log10(3.8E33))/b)**c)
		return kx