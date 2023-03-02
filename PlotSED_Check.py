import numpy as np 
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import TextBox, CheckButtons, Slider

plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.minor.size'] = 3.
plt.rcParams['xtick.minor.width'] = 2.
plt.rcParams['ytick.minor.size'] = 3.
plt.rcParams['ytick.minor.width'] = 2.


def main(fname, ID, z, wavelength, Lum, L):
    plot = IntPlot(fname, ID, z, wavelength, Lum, L)

class IntPlot():
    """
    interactive plotting class to check SEDs and indicate extrapolations

    """

    def __init__(self, fname, ID, z, wavelength, Lum, L, norm, up_check):
        self.fname = fname
        self.ID = np.asarray(ID)
        self.z = np.asarray(z)
        self.wavelength = np.asarray(wavelength)
        self.Lum = np.asarray(Lum)
        self.L = np.asarray(L)
        self.norm = np.asarray(norm)
        self.up_check = np.asarray(up_check)
        self.out_array = np.array([0, 0, 0, 0, 0, 0, 0])

    def Plot(self,point_x,point_y,save=False):
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.plot(self.wavelength,self.Lum)
        ax.plot(self.wavelength,self.Lum,'x',c='k')
        ax.plot(point_x,point_y,'x',ms=10,c='r')

        ax.set_xlabel(r'Rest Wavelength [$\mu$ m]')
        ax.set_ylabel(r'$\lambda$L$_\lambda$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(self.ID)
        ax.text(0.05,0.8,f'Lx = {np.log10(self.L)}',transform=ax.transAxes,fontsize=18)
        ax.text(0.05,0.7,f'z = {self.z}',transform=ax.transAxes,fontsize=18)
        ax.text(0.05,0.6,f'FIR upper: {self.up_check}',transform=ax.transAxes,fontsize=18)
        plt.xlim(5E-5, 7E2)
        plt.ylim(1E-4, 1E2)
        plt.grid()

        rax = plt.axes([0.75, 0.0, 0.25, 0.35])
        labels = ['Bad SED', 'UV extrap', 'F1 extrap', 'MIR extrap', 'FIR extrap', 'Bad FIR', 'Manual Check']
        visibility = [False, False, False, False, False, False, False]
        check = CheckButtons(rax, labels, visibility)
        # check.label.set_fontsize(14)
        check.on_clicked(self.check_box)

        rax2 = plt.axes([0.0,0.0,0.2,0.1])
        labels2 = ['Save Output']
        visibility2 = [False]
        save_check = CheckButtons(rax2, labels2, visibility2)
        # save_check.label.set_fontsize(14)
        save_check.on_clicked(self.button_press)

        if save:
            plt.savefig(f'/Users/connor_auge/Desktop/sed_check_output/{self.ID}_SED.pdf')
        plt.show()

    def button_press(self, button):
        if button == 'Save Output': self.save(self.fname, self.out_array)

    def check_box(self, check_mark):
        if check_mark == 'Bad SED': self.out_array[0] = 1
        if check_mark == 'UV extrap': self.out_array[1] = 1 
        if check_mark == 'F1 extrap': self.out_array[2] = 1
        if check_mark == 'MIR extrap': self.out_array[3] = 1
        if check_mark == 'FIR extrap': self.out_array[4] = 1
        if check_mark == 'Bad FIR': self.out_array[5] = 1
        if check_mark == 'Manual Check': self.out_array[6] = 1


    def save(self, fname, out_array):
        try:
            with open(f'/Users/connor_auge/Desktop/sed_check_output/{fname}.txt','a') as my_file:
               my_file.write('%s,%f,%f,%f,%f,%f,%f,%f\n' % (self.ID, out_array[0], out_array[1], out_array[2], out_array[3], out_array[4], out_array[5], out_array[6]))
        except FileNotFoundError:
            self.write_file(fname)
    
    def write_file(self,fname):
        outf = open(f'/Users/connor_auge/Desktop/sed_check_output/{fname}.txt','w')
        outf.writelines('ID,Bad_SED,UV_extrap,F1_extrap,MIR_extrap,FIR_extrap,Bad_FIR,Manual_Check\n')
        outf.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Script for plotting a single normalized SED and manually checking the quality of different features. Quality check can be save to an external file.')
    parser.add_argument('fname',help='Name of output file to be written',type=str)
    parser.add_argument('--ID','-id',help='Source ID')
    parser.add_argument('--z','-z',help='redshift')
    parser.add_argument('--wavelength','-x',help='SED wavelength (x-axis) array')
    parser.add_argument('--Lum','-y',help='SED luminosity (y-axis) array')
    parser.add_argument('--L','-L',help='additional Luminosity to examine (default is X-ray luminosity)')
    parser.add_argument('--norm','-norm',help='value the SED has already been normalized by')
    parser.add_argument('--up_check','-upper',help='flag indiciating if the upper limit is used to determine the FIR lum')