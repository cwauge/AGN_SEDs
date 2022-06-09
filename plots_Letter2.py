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
from SED_v7 import Flux_to_Lum
import matplotlib.patheffects as pe

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
		plt.savefig(f'/Users/connor_auge/Desktop/New_plots3/{savestring}.pdf')
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

	def Upanels_ratio_plots(self, savestring, X, Y, Median, Nh, Lx, L, L1, L2, L3, f1, f2, f3, f4, F1, field, spec_z, uv_slope, mir_slope1, mir_slope2, up_check):
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

			ax1.scatter(x_11, y_11, color=c1, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 1',zorder=0)
			ax1.scatter(x_12, y_12, color=c2, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 2',zorder=0)
			ax1.scatter(x_13, y_13, color=c3, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 3',zorder=0)
			ax1.scatter(x_14, y_14, color=c4, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 4',zorder=0)
			ax1.scatter(x_15, y_15, color=c5, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 5',zorder=0)

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
			ax2.scatter(x_21, y_21, color=c1, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 1',zorder=0)
			ax2.scatter(x_22, y_22, color=c2, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 2',zorder=0)
			ax2.scatter(x_23, y_23, color=c3, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 3',zorder=0)
			ax2.scatter(x_24, y_24, color=c4, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 4',zorder=0)
			ax2.scatter(x_25, y_25, color=c5, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 5',zorder=0)

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
				print('yes')
				ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], facecolor='none', edgecolors=c1, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], facecolor='none', edgecolors=c2, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], facecolor='none', edgecolors=c3, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], facecolor='none', edgecolors=c4, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], facecolor='none', edgecolors=c5, rasterized=True, alpha=0.4,zorder=0)

				ax3.scatter(x_31[up_check_31 == 1], y_31[up_check_31 == 1], marker=3, color=c1, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32[up_check_32 == 1], y_32[up_check_32 == 1], marker=3, color=c2, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33[up_check_33 == 1], y_33[up_check_33 == 1], marker=3, color=c3, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34[up_check_34 == 1], y_34[up_check_34 == 1], marker=3, color=c4, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35[up_check_35 == 1], y_35[up_check_35 == 1], marker=3, color=c5, rasterized=True, alpha=0.4,zorder=0)

				ax3.scatter(x_31[up_check_31 == 0], y_31[up_check_31 == 0], color=c1, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 1',zorder=0)
				ax3.scatter(x_32[up_check_32 == 0], y_32[up_check_32 == 0], color=c2, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 2',zorder=0)
				ax3.scatter(x_33[up_check_33 == 0], y_33[up_check_33 == 0], color=c3, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 3',zorder=0)
				ax3.scatter(x_34[up_check_34 == 0], y_34[up_check_34 == 0], color=c4, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 4',zorder=0)
				ax3.scatter(x_35[up_check_35 == 0], y_35[up_check_35 == 0], color=c5, marker='P', lw=0, rasterized=True, alpha=0.4, label='Panel 5',zorder=0)

			else:
				ax3.scatter(x_31, y_31, color=c1, marker='P', lw=0, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_32, y_32, color=c2, marker='P', lw=0, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_33, y_33, color=c3, marker='P', lw=0, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_34, y_34, color=c4, marker='P', lw=0, rasterized=True, alpha=0.4,zorder=0)
				ax3.scatter(x_35, y_35, color=c5, marker='P', lw=0, rasterized=True, alpha=0.4,zorder=0)

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
			plt.savefig(f'/Users/connor_auge/Desktop/New_plots3/{savestring}.pdf')
			plt.show()

	def Box_1panel(self, savestring, var, x, uv_slope, mir_slope1, mir_slope2, ulirg_x=None):
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

		c1 = '#377eb8'
		c2 = '#984ea3'
		c3 = '#4daf4a'
		c4 = '#ff7f00'
		c5 = '#e41a1c'

		def solar(x):
				return x - np.log10(3.8E33)

		def ergs(x):
			return x + np.log10(3.8E33)


		fig = plt.figure(figsize=(9, 9))
		ax1 = plt.subplot(111, aspect='equal', adjustable='box')
		ax1.plot([1,2,3,4,5],[np.nanmean(x1),np.nanmean(x2),np.nanmean(x3),np.nanmean(x4),np.nanmean(x5)],color='k')
		ax1.boxplot(x1,positions=[1],patch_artist=True,boxprops=dict(facecolor=c1, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4),whiskerprops=dict(color=c1,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=False)
		ax1.boxplot(x2,positions=[2],patch_artist=True,boxprops=dict(facecolor=c2, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4),whiskerprops=dict(color=c2,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=False)
		ax1.boxplot(x3,positions=[3],patch_artist=True,boxprops=dict(facecolor=c3, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4),whiskerprops=dict(color=c3,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=False)
		ax1.boxplot(x4,positions=[4],patch_artist=True,boxprops=dict(facecolor=c4, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4),whiskerprops=dict(color=c4,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=False)
		ax1.boxplot(x5,positions=[5],patch_artist=True,boxprops=dict(facecolor=c5, color='k'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4),whiskerprops=dict(color=c5,lw=3,path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]),showcaps=False,showfliers=False)
		
		ax1.set_ylim(ylim1, ylim2)
		plt.gca().invert_xaxis()		
		ax1.set_ylabel(ylabel+units)
		ax1.set_xlabel('Panel Number')
		ax1.set_xticklabels(xticklabels)
		if any(ulirg_x) != None:
			ax1.boxplot(ulirg_x,positions=[6],patch_artist=True,boxprops=dict(facecolor='gray', color='gray'),medianprops=dict(color='k',lw=3,alpha=0),meanline=True,showmeans=True,meanprops=dict(color='k',lw=4))

		if var == 'Lbol':
			secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
			secax1.set_ylabel(ylabel+' [erg/s]')
		elif var == 'Lx':
			secax1 = ax1.secondary_yaxis('right', functions=(ergs, solar))
			secax1.set_ylabel(ylabel+' [erg/s]')
		ax1.grid()
		plt.tight_layout()
		plt.savefig('/Users/connor_auge/Desktop/New_plots3/'+savestring+'.pdf')
		plt.show()

	def ratios_1panel(self, savestring, X, Y, Median, Nh, Lx, L, F1, f1, f2, f3, f4, uv_slope, mir_slope1, mir_slope2, up_check):
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

		elif Y == 'MIR6':
			f_2 = np.asarray([10**i for i in f2])
			f2 = f_2*F1
			lx = np.asarray([10**i for i in Lx])
			y = np.log10(f2/lx)
			ylabel = r'log L (6$\mu$m)/ L$_{\mathrm{X}}$'
			ylim1 = -2
			ylim2 = 3

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


		ax1.set_xlim(xlim1,xlim2)
		ax1.set_ylim(ylim1,ylim2)
		ax1.set_yticks(yticks)
		ax1.set_ylabel(ylabel)
		ax1.set_xlabel(xlabel+xunits)
		if X != 'Nh':
			secax3 = ax1.secondary_xaxis('top', functions=(solar, ergs))
			secax3.set_xlabel(xlabel+r' [L$_{\odot}$]')
		ax1.legend(fontsize=14)
		ax1.grid()
		plt.savefig('/Users/connor_auge/Desktop/New_plots3/'+savestring+'.pdf')
		plt.show()
				
