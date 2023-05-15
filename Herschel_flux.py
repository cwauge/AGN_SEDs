import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from matplotlib.colors import LogNorm
from bootstrap_err import BootStrap


class Herschel():

    def __init__(self, fname, ra, dec):
        self.fname = fname
        self.ra = ra
        self.dec = dec

        with fits.open(fname) as hdul:
            self.head = hdul[1].header
            self.data = hdul[1].data
            self.err = hdul[2].data
            self.exp = hdul[3].data
            self.mask = hdul[4].data
            self.wcs_data = WCS(hdul[1].header)
            self.wcs_err = WCS(hdul[2].header)

    def sum_pixels(self,dat_array,wcs,im_size):
        position = SkyCoord(self.ra*u.degree, self.dec*u.degree)
        size = im_size*u.pixel
        cutout = Cutout2D(dat_array,position,size,wcs=wcs)

        return cutout.data[0][0]

    def random(self, data, data_err):
        rand_data = np.random.normal(data, data_err)

        return rand_data

    def MC(self,numb):
        detection = self.sum_pixels(self.data,self.wcs_data,3)
        error = self.sum_pixels(self.err,self.wcs_err,3)

        out_detection_boot = []
        
        for _ in range(numb):
            detect_boot = self.random(detection,error)
            out_detection_boot.append(detect_boot)

        return np.asarray(out_detection_boot)
