import numpy as np
import re
import pylab as plt
import math
import random as rnd
import time
import subprocess
import sys

from termcolor import colored
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interpolate
from pylab import *
from numpy import linspace
from os import listdir
from os.path import isfile, join

from trgpy.config import cosmo_params
from trgpy.dictionary_constants import cnsts
from trgpy.dictionary_bandpass import bndpass

#--- Magic values
UnDefVal = -999
UpLimVal = 99
LoLimVal = -99

#--- High frequency cut-off for NED data
#frq_high = 2.E15   # [Hz]


""" This module is meant to facilitate the use of CIGALE. It can read NED
.bsv files with photometric data and convert them into CIGALE .mag files.
It can also plot SED data and CIGALE best fits.

Started:    26.11.2018
Modified:   22.06.2019
"""


#
#
class GalaxySED:

    def __init__(self, object_id, frq_range=None, remove_data_indices=None, crop_data=False, verbose=False, tbl=False):
        """
            Status:
                Needs fixing: shouldn't have to depend on emg.csv. Should be
                more general. exctract_emg is only used to get z.
        """

        # Read in NED data for object_id and get redshift
        band, frq_o, wvl_o, flx, eflx, z = read_ned(object_id, frq_range=frq_range, remove_data_indices=remove_data_indices, crop_data=crop_data, verbose=verbose, tbl=tbl)

        self.object_id = object_id
        self.z = z
        self.frq_o = frq_o
        self.wvl_o = wvl_o
        self.band = band
        self.flx = flx
        self.eflx = eflx


    def kill_datapoints(self, indices_discard=[]):
        """
           Remove specified photometry data points.
        """

        if indices_discard != []:
            # Check that indices are valid, otherwise exit with error
            assert len(indices_discard) < len(self.band), \
                'Invalid number of indices given!'
            assert min(indices_discard) >= -1.0 * len(self.band), \
                'Minimum indices too small!'
            assert max(indices_discard) < len(self.band), \
                'Maximum indices too large!'

            foo = self.frq_o
            foo[indices_discard] = UnDefVal
            indices_retain = [i for i, item in enumerate(foo) if foo[
                i] != UnDefVal]
            self.band = [self.band[i] for i in indices_retain]
            self.wvl_o = self.wvl_o[indices_retain]
            self.frq_o = self.frq_o[indices_retain]
            self.flx = self.flx[indices_retain]
            self.eflx = self.eflx[indices_retain]


    def list(self):
        """
           List the photometry (band, frq, flx, eflx).
        """
        # List photometry to terminal
        print(
            '             Band                          frq_o[Hz]   flx[Jy]     eflx[Jy]')
        j = 0
        for i in self.frq_o:
            if self.eflx[j] != UpLimVal and self.eflx[j] != 999:
                output_string = (str(j).ljust(10) + '   ' + self.band[j].ljust(30)
                                 + '{:.2e}'.format(float(self.frq_o[j])) + '    '
                                 + '{:.2e}'.format(float(self.flx[j])) + '    '
                                 + '{:.2e}'.format(float(self.eflx[j])))
            else:
                output_string = (str(j).ljust(10) + '   ' + self.band[j].ljust(30)
                                 + '{:.2e}'.format(float(self.frq_o[j])) + '    '
                                 + '{:.2e}'.format(float(self.flx[j])) + '    '
                                 + repr(int(self.eflx[j])).rjust(5))

            print(output_string)
            j = j + 1


    def pcigale(self, create_mag=True, path=None, verbose=False):
        """
           Truncate self to only contain the band entries which
           are found in the pcigale filter database.

           Input:
                 create_mag : Create a [object_id].mag file? Default (=True).

                 path : path to folder where .mag file is to be
                        stored. Default (=None) is cigale cigale_mag/.

                verbose : output known and unknown passbands to terminal.
                          Default (=False).

           STATUS:
                18.12.16: started
                06.11.18: python 3.6.5 compatible. Changed default path.
        """
        # Format band ids to pcigale bandpass names
        band_original = [item for item in self.band]
        band_pcigale = _map_bands_to_pcigale(self.band, verbose=verbose)
        self.band = band_original
        # Identify passbands entries unknown to pcigale
        indices_unknown = [i for i, item in enumerate(band_pcigale) if
                           item == 'REMOVE']
        unknown_passbands = [band_original[i] for i in indices_unknown]
        print('--------------')
        print('Uknown passbands:')
        print(unknown_passbands)
        print('--------------')
        # Removing passbands entries unknown to pcigale
        indices_known = [i for i, item in enumerate(band_pcigale) if
                         item != 'REMOVE']
        if indices_known != []:
            self.band = [band_original[i] for i in indices_known]
            self.flx = [self.flx[i] for i in indices_known]    # [Jy]
            self.eflx = [self.eflx[i] for i in indices_known]    # [Jy]
            self.frq_o = [self.frq_o[i] for i in indices_known]    # [Hz]
            self.wvl_o = [self.wvl_o[i] for i in indices_known]    # [um]
            # create .mag file
            if create_mag:
                band_out = [band_pcigale[i] for i in indices_known]
                flx_pcigale = np.array(self.flx) * 1.E3    # [mJy]
                wvl_o_pcigale = np.array(self.wvl_o) * 1.E-6 / 1.E-10    # [AA]
                # Convert flux errors to mJy
                eflx_pcigale = np.array(self.eflx)
                indices_errors = [i for i, x in enumerate(eflx_pcigale)
                                  if x > 0 and x < UpLimVal]
                eflx_pcigale[indices_errors] = eflx_pcigale[
                    indices_errors] * 1.E3    # [mJy]
                # UnDefVal is magic value for undefined flux error in pcigale
                indices_undefined = [i for i, x in enumerate(eflx_pcigale)
                                     if x == UnDefVal]
                eflx_pcigale[indices_undefined] = eflx_pcigale[
                    indices_undefined] - 10098
                # UpLimVal is magic value for upper limit in pcigale
                indices_upper_limits = [
                    i for i, x in enumerate(eflx_pcigale) if x == UpLimVal]
                eflx_pcigale[indices_upper_limits] = eflx_pcigale[
                    indices_upper_limits] - 1098

                create_pcigale_mag(self.object_id, self.z, band_out,
                                   wvl_o_pcigale, flx_pcigale, eflx_pcigale,
                                   path=path)

            self.flx = np.array(self.flx)
            self.eflx = np.array(self.eflx)
            self.frq_o = np.array(self.frq_o)
            self.wvl_o = np.array(self.wvl_o)

        else:
            print('No bands were recognised by the pcigale filter database')



#
#
def read_ned(object_id, frq_range=None, remove_data_indices=None,
        tbl=False, crop_data=False, verbose=False):
    """
       Reads in NED file for object_id.

       NED file format
       - if the uncertainty is blank/undefined in the NED-file: eflx=999
       - if the flux is an upper limit in the NED file then infer the 1sigma
         and: flx=3sigma and eflx=99
       - if the flux is negative (but eflx is defined) in the NED file:
         flx=3*eflx and eflx=99

       low_frq_cut,high_frq_cut    :    defines frequency range outside
                                        which data are colored in red on output
       remove_data_indices         :    indices of data points that should be
                                        higlighted in red and possibly removed
       crop_data = False           :    data are not cropped before return
       verbose = True              :    data are displayed in terminal

       STATUS:
           15.09.17: started
           06.11.18: python 3.6.5 compatible
           07.11.18: reads redshift from .bsv file. Automatically finds line
                     where data starts.
           26.11.18: can now read tbl files from new NED.
    """


    # Read in NED file first to get redshift and line number to read from
    if tbl:
        try:
            # Get redshift and its line_number
            print(' ')
            print('reading in NED/' + object_id + '.tbl')

            f = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/NED/' + object_id + '.tbl', 'r')
            lines = f.readlines()[0:]
            f.close()
            line_number_z = 0
            for line in lines:
                p = line.strip()
                foo = p.split('z=')
                if len(foo) == 2:
                    z = foo[1]
                    break
                line_number_z = line_number_z+1

            # Get line number where '|No.' starts
            line_number = 0
            for line in lines:
                p = line.strip()
                foo = p.split('|No')
                if len(foo) == 2:
                    break
                line_number = line_number+1

            # Get all positions of '|' in p
            index_bar = [pos for pos, char in enumerate(p) if char == '|']
            # Remove first instance
            index_bar = index_bar[1:]
            index_bar_truncated = index_bar[:len(index_bar)-1]

            # Add '|' at index_bar positions
            i=0
            out = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/NED/' + object_id + '.bsv', 'w')
            for line in lines:
                if i >= line_number+4:
                    foo = line[0:index_bar[0]]
                    jj=0
                    for ii in index_bar_truncated:
                        foo = foo+'|'+line[index_bar[jj]+1:index_bar[jj+1]]
                        jj=jj+1
                    foo = foo + '\n'
                else:
                    foo = line
                out.write(foo)
                i=i+1
            out.close()

        except IOError:
            print >> sys.stderr, "Exception: %s" % str(e)
            sys.exit(1)
    else:
        try:
            print(' ')
            print('reading in NED/' + object_id + '.bsv')
            f = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/NED/' + object_id + '.bsv', 'r')
            lines = f.readlines()[1:]
            f.close()
            line_number = 0
            for line in lines:
                p = line.strip()

                # Get redshift
                foo = p.split('z=')
                if len(foo) == 2:
                    z = foo[1]
                if '1|' in p:
                    break
                line_number = line_number+1
        except IOError:
            print >> sys.stderr, "Exception: %s" % str(e)
            sys.exit(1)


    # Read in NED file photometry data
    if tbl:
        try:
            f = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/NED/' + object_id + '.bsv', 'r')
            lines = f.readlines()[line_number+4:]
            f.close()

            band = []
            frq_o = []
            flx = []
            eflx = []
            snr = []

            for line in lines:
                p = line.strip()

                # split columns into v-variables
                p = p.split('|')
                if p != []:
                    v1 = p[1]
                    v3 = p[3]
                    v4 = p[4]
                    v5 = p[5]
                    v6 = p[6]
                    v7 = p[7]
                    v9 = p[9]
                    v10 = p[10]
                    v11 = p[11]
                    v14 = p[14]

                    # find strings to get rid off
                    index_1 = v1.find('CO')
                    index_1a = v1.find('H{alpha}')
                    index_11 = v11.find('+/-')
                    index_14 = v14.find('no uncertainty')
                    #index_7 = v7.find('+/-')
                    index_3a = v3.find('<')
                    index_3b = v3.find('>')
                    #index_4a = v4.find('mag')

                    snr_dummy = 1.0
                    # Remove CO, Halpha and > entries entries
                    if index_1 == -1 and index_1a == -1 and index_3b == -1:
                        # Get flux error; remove +/-
                        if index_11 != -1:
                            v11 = v11[3:len(v11)]
                        # If no flux uncertainty set eflx to UnDefVal
                        if index_14 != -1:
                            v11 = UnDefVal
                        # If upper limit set flx to upper limit and eflx=99
                        if index_3a != -1:
                            v6 = v9
                            v11 = UpLimVal

                        band.append(v1)
                        frq_o.append(v5)
                        flx.append(v6)
                        eflx.append(v11)
                        snr.append(snr_dummy)
        except IOError:
            print >> sys.stderr, "Exception: %s" % str(e)
            sys.exit(1)

    else:
        try:
            f = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/NED/' + object_id + '.bsv', 'r')
            lines = f.readlines()[line_number+1:]
            f.close()

            band = []
            frq_o = []
            flx = []
            eflx = []
            snr = []

            for line in lines:
                p = line.strip()

                # split columns into v-variables
                p = p.split('|')
                if p != []:
                    v1 = p[1]
                    v3 = p[3]
                    v4 = p[4]
                    v5 = p[5]
                    v6 = p[6]
                    v7 = p[7]
                    v10 = p[10]

                    # find strings to get rid off
                    index_1 = v1.find('CO')
                    index_1a = v1.find('H{alpha}')
                    index_7 = v7.find('+/-')
                    index_3a = v3.find('<')
                    index_3b = v3.find('>')
                    index_4a = v4.find('mag')

                    snr_dummy = 1.0
                    if index_1 == -1 and index_1a == -1:  # We do not consider CO lines
                        if index_7 != -1:
                            # flux error, after +/- has been removed
                            v7 = v7[3:len(v7)]
                        if index_3a != -1:
                            if v10[0:1] == 'n':
                                snr_dummy = -99  # no sigma defined, so we discard the upper limit
                            else:
                                snr_dummy = float(v10[0:1])
                            v6 = v7
                            v7 = str(UpLimVal)
                        if index_3b != -1 and index_4a != -1:
                            snr_dummy = float(v10[0:1])
                            v6 = v7
                            v7 = str(UpLimVal)
                        if v7 == ' ' or v7 == '':
                            v7 = str(UnDefVal)

                        band.append(v1)
                        frq_o.append(v5)
                        flx.append(v6)
                        eflx.append(v7)
                        snr.append(snr_dummy)
        except IOError:
            print >> sys.stderr, "Exception: %s" % str(e)
            sys.exit(1)

    # Convert data from string to float
    frq_o = [float(x) for x in frq_o]
    flx = [float(x) for x in flx]
    eflx = [float(x) for x in eflx]
    snr = [float(x) for x in snr]

    # Only keep data points which have well define SNR
    indices_retain = [i for i, x in enumerate(snr) if x != -99]
    band = [band[i] for i in indices_retain]
    frq_o = [frq_o[i] for i in indices_retain]
    flx = [flx[i] for i in indices_retain]
    eflx = [eflx[i] for i in indices_retain]
    snr = [snr[i] for i in indices_retain]

    j = 0
    frac = 0.9
    for i in frq_o:
        # Make sure we are getting 3sigma upper limits
        if eflx[j] == UpLimVal:
            flx[j] = 3. * flx[j] / snr[j]
        if flx[j] < 0:
            flx[j] = 3. * eflx[j]
            eflx[j] = UpLimVal
        # If eflx/flx > frac discard data point
        if eflx[j] != UpLimVal and eflx[j] != UnDefVal and eflx[j] / flx[j] >= frac:
            flx[j] = UnDefVal
        j = j + 1

    # remove data points with eflx/flx > frac
    indices = [i for i, x in enumerate(flx) if x != UnDefVal]
    band = [band[i] for i in indices]
    frq_o = [frq_o[i] for i in indices]
    flx = [flx[i] for i in indices]
    eflx = [eflx[i] for i in indices]

    band = np.array(band)
    frq_o = np.array(frq_o)
    flx = np.array(flx)
    eflx = np.array(eflx)

    # Set eflx = 0.4 * flx for data points with undefined eflx
    indices = np.where(eflx == UnDefVal)
    eflx[indices] = flx[indices] * 0.4

    # Sort data according to frequency
    indices = frq_o.argsort()
    band = (band[indices])[::-1]
    flx = (flx[indices])[::-1]
    eflx = (eflx[indices])[::-1]
    frq_o = (frq_o[indices])[::-1]

    # Determine which data points to exclude and color code red
    indices = arange(0, len(frq_o))
    if frq_range is not None:
        indices_discard = np.reshape(np.where((frq_o < frq_range[0]) | (frq_o > frq_range[1])), -1)
        if remove_data_indices is not None:
            indices_discard = np.concatenate((np.array(indices_discard), np.array(remove_data_indices)), axis=0)
    else:
        indices_discard = None #indices + UnDefVal

    # Output to terminal
    if verbose:
        print('             Band                          frq_o[Hz]   flx[Jy]     eflx[Jy]')
        k = 0
        for i in frq_o:
            if eflx[k] != UpLimVal:
                output_string = (str(k).ljust(10) + '   ' + band[k].ljust(30)
                                 + '{:.2e}'.format(float(frq_o[k])) + '    '
                                 + '{:.2e}'.format(float(flx[k])) + '    '
                                 + '{:.2e}'.format(float(eflx[k])))
            else:
                output_string = (str(k).ljust(10) + '   ' + band[k].ljust(30)
                                 + '{:.2e}'.format(float(frq_o[k])) + '    '
                                 + '{:.2e}'.format(float(flx[k])) + '    '
                                 + repr(int(eflx[k])).rjust(5))

            if k in np.array(indices_discard):
                print(colored(output_string, 'red'))
            else:
                print(output_string)
            k = k + 1

    # Extract data not in indices_discard (highlighted in red in the output)
    if crop_data:
        foo = frq_o
        foo[indices_discard] = UnDefVal
        indices_retain = np.where(foo != UnDefVal)
        frq_o = np.array(frq_o[indices_retain])
        flx = np.array(flx[indices_retain])
        eflx = np.array(eflx[indices_retain])

    # Calculate observed wavelength [um]
    wvl_o = cnsts['c'] / frq_o    # [Hz]
    wvl_o = wvl_o * 1.E6

    return band, frq_o, wvl_o, flx, eflx, z    # [], [Hz], [um], [Jy]


#
#
def plot_sed(id, z, x, flx, eflx, xaxis='GHz', markerfacecolor='LightPink',
             markeredgecolor='DarkRed', new_plot=True, show_plot=True):
    """
       Plot SED data in the observers frame, i.e., observed
       frequency ([GHz]) versus observed flux density ([mJy)].
    """

    indices_up = [i for i, j in enumerate(eflx) if j == UpLimVal]
    indices_not_up = [i for i, j in enumerate(eflx) if j != UpLimVal]
    indices_err = [i for i, j in enumerate(eflx) if j != 999 and j != UpLimVal]
    indices_err_ud = [i for i, j in enumerate(eflx) if j == 999]

    if xaxis == 'micron':
        xlabel = 'Observed wavelength [micron]'
        xlim1 = 3.E-2
        xlim2 = 1.E6
    else:
        x = x / 1.E9  # [GHz]
        xlabel = 'Observed frequency [GHz]'
        xlim1 = 0.5
        xlim2 = 1.E6
    flx = flx * 1.E3  # [mJy]
    eflx = eflx * 1.E3  # [mJy]

    if new_plot:
        plt.ioff()
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(xlim1, xlim2)
        plt.ylim(1.E-5, 5.E6)
        plt.xlabel(xlabel)
        plt.ylabel('Flux density [mJy]')
        plt.title(id + ' (z=' + str(z) + ')')

    plt.plot(x[indices_not_up], flx[indices_not_up], linestyle='', marker='o',
             markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
    plt.errorbar(x[indices_err], flx[indices_err], yerr=eflx[indices_err],
                 fmt='none', ecolor=markeredgecolor)
    if len(indices_up) != 0:
        plt.errorbar(x[indices_up], flx[indices_up], uplims=True, fmt='none',
                     yerr=.5 * flx[indices_up], ecolor=markeredgecolor)
    if show_plot:
        plt.show()


#
#
def obs_to_restframe_sed(z, x_o, flx, wvl_domain=False):
    """
       Transform an observed SED, i.e. flx vs. frq or wvl, to the
       restframe, i.e., specific luminosity vs. frq or wvl.

       The observed flux (flx) must be in units of Jy. The output
       specific luminosity is in units of W Hz^-1.

       Input:
           z            :    redshift
           x_o          :    observed frq (Hz) or wvl (m)
           flx          :    observed flux
           wvl_domain   :    if False: frq domain; if True: wvl domain

        Output:
           x_r          :    restframe frq or wvl
           L_r          :    specific luminosity (W Hz^-1 or W m^-1)

       06.12.16
    """


    # Luminosity distance [m]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    dl = cosmo.luminosity_distance(z)
    dl = dl.value * cnsts['pc2m'] * 1.E6

    if wvl_domain:
        wvl_r = x_o / (1. + z)    # [m]
        frq_r = cnsts['c'] / (wvl_r)    # [Hz]
        L_frq_r = dl * (flx * 1.E-26) * (4. * math.pi *
                                         dl / (1. + z))    # [W Hz^-1]
        L_wvl_r = L_frq_r * cnsts['c'] / (wvl_r * wvl_r)    # [W m^-1]
        x_r = wvl_r
        L_r = L_wvl_r
    else:
        frq_r = x_o * (1. + z)    # [Hz]
        L_frq_r = dl * (flx * 1.E-26) * (4. * math.pi *
                                         dl / (1. + z))    # [W Hz^-1]
        x_r = frq_r
        L_r = L_frq_r

    return x_r, L_r


#
#
def _integrand_get_lum(x, x_r, L_r):
    """
       Returns specific luminosity at a given frq or wvl.

       Helping function for get_lum.

       06.12.16
    """

    foo = UnivariateSpline(x_r, L_r, k=5, s=0)(x)

    return foo


#
#
def get_lum(x_r, L_r, x_1, x_2, wvl_domain=False):
    """
       Returns integrated luminosity of given SED from x_1 to x_2.
       SED is in restframe.

       Input:
           x_r          :    rest-frame frq (Hz) or wvl (m)
           L_r          :    specific luminosity (W Hz^-1 or W m^-1)
           x_a          :    start integration limit (in Hz or m)
           x_b          :    end integration limit (in Hz or m)
           wvl_domain   :    if False: frq domain; if True: wvl domain

       Output:
           lum          :    luminosity (Lsolar)

       06.12.16
    """

    if wvl_domain:
        x_r = x_r    # [m]
        L_r = L_r / cnsts['Lsolar']    # [Lsolar m^-1]
        x1 = x_r
        y1 = L_r
        scale = 1.
    else:
        x_1 = x_1 / 1.E9    # [GHz]
        x_2 = x_2 / 1.E9    # [GHz]
        x_r = x_r / 1.E9    # [GHz]
        L_r = L_r / cnsts['Lsolar']    # [Lsolar Hz^-1]
        x1 = x_r[::-1]
        y1 = L_r[::-1]
        scale = 1.E9

    lum = quad(_integrand_get_lum, x_1, x_2,
               args=(x1, y1), limit=40, epsrel=0.1)
    lum = lum[0] * scale  # [Lsolar]

    return lum


#
#
def create_pcigale_mag(object_id, z, band, wvl_o, flx, eflx, path=None,
                       append_magfile=False):
    """
       Creates .mag pcigale input file.

       Input:
           object_id    :   string containing object_id
           z            :   redshift
           band         :   list of passbands
           wvl_o        :   array of observed wavelengths [AA]
           flx          :   array of observed fluxes [mJy]
           eflx         :   array of observed flux errors [mJy]
           path         :   path to where .mag file should be stored
                            Default (=None) is pcigale_mag/

       Keywords:
            append_magfile  :   if True append to existing .mag file.

       STATUS:
            14.12.16: started
            06.11.18: python 3.6.5 compatible
    """

    # Create new file or append to existing?
    if path:
        target = open(path + object_id + '.mag', 'w')
        header_str = '# id                                         \
                      redshift       '
    else:
        if append_magfile:
            target = open('cigale_mag/' + object_id + '.mag', 'a')
            header_str = ' '
        else:
            target = open('cigale_mag/' + object_id + '.mag', 'w')
            header_str = '# id                                         \
                          redshift       '
    data_str = '  ' + '{0:30}'.format(object_id) + '             ' + \
               '{0:12}'.format(z) + '   '

    # Complete header and data strings
    i = 0
    for arg in band:
        header_str = header_str + '{0:20}  {1:20}'.format(arg, arg + '_err')
        flx_str = '{0:2.3e}'.format(flx[i])
        eflx_str = '{0:2.3e}'.format(eflx[i])
        data_str = data_str + '{0:20}  {1:20}'.format(flx_str, eflx_str)
        i += 1
    header_str = header_str + '\n'
    data_str = data_str + '\n'

    # Write to file and close
    target.write(header_str)
    target.write(data_str)
    target.close()



#
#
def _map_bands_to_pcigale(band, verbose=False):
    """
       Internal helping function that maps bandpass identifiers in .bsv files
       from NED to their valid bandpass identifiers in pcigale (>v12).

       Input:
           band          :    list of bandpass ids.

        Output:
           band          :    list of bandpass identifiers valid in
                              pcigale.

       STATUS:
            29.11.18: started
    """

    # Format band ids to CIGALE names
    unknown_band = [band[i] for i in arange(0, len(band))]
    for i in arange(0, len(band), 1):
        band_str = band[i]
        if 'CFHT' in band_str:
            if 'u' in band_str:
                band[i] = 'cfht.megacam.u'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'GALEX' in band_str:
            if 'NUV' in band_str:
                band[i] = 'galex.NUV'
            elif 'FUV' in band_str:
                band[i] = 'galex.FUV'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SDSS' in band_str:
            if 'u' in band_str:
                band[i] = 'sdss.up'
            elif 'g' in band_str:
                band[i] = 'SDSS_g'
            elif 'r' in band_str:
                band[i] = 'SDSS_r'
            elif 'i' in band_str:
                band[i] = 'SDSS_i'
            elif 'z' in band_str:
                band[i] = 'SDSS_z'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('ACS' in band_str):
            if ('F435W' in band_str) or ('B (HST' in band_str):
                band[i] = 'ACS_F435W'
            elif 'F475W' in band_str:
                band[i] = 'ACS_F475W'
            elif 'F502N' in band_str:
                band[i] = 'ACS_F502N'
            elif 'F550M' in band_str:
                band[i] = 'ACS_F550M'
            elif 'F555W' in band_str:
                band[i] = 'ACS_F555W'
            elif 'F606W' in band_str:
                band[i] = 'ACS_F606W'
            elif 'F625W' in band_str:
                band[i] = 'ACS_F625W'
            elif 'F658N' in band_str:
                band[i] = 'ACS_F658N'
            elif 'F660N' in band_str:
                band[i] = 'ACS_F660N'
            elif ('F775W' in band_str) or ('i (HST' in band_str):
                band[i] = 'ACS_F775W'
            elif 'F814W' in band_str:
                band[i] = 'ACS_F814W'
            elif ('F850LP' in band_str) or ('z (HST' in band_str):
                band[i] = 'ACS_F850LP'
            elif 'F892N' in band_str:
                band[i] = 'ACS_F892N'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('WFP' in band_str):
            if 'F555W' in band_str:
                band[i] = 'WFPC2_F555W'
            elif 'F702W' in band_str:
                band[i] = 'WFPC2_F702W'
            elif 'F814W' in band_str:
                band[i] = 'WFPC2_F814W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('WFC' in band_str):
            if 'F110W' in band_str:
                band[i] = 'HST_WFC3_F110W'
            elif 'F125W' in band_str:
                band[i] = 'HST_WFC3_F125W'
            elif 'F130N' in band_str:
                band[i] = 'HST_WFC3_F130N'
            elif 'F160W' in band_str:
                band[i] = 'HST_WFC3_F160W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and (('NIC2' in band_str) or ('NICMOS' in band_str)):
            if 'F160W' in band_str:
                band[i] = 'NICMOS_F160W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('Subaru' in band_str) or ('SUBARU' in band_str):
            band_str = band_str.replace('AB', '')
            band_str = band_str.replace('SUBARU', '')
            band_str = band_str.replace('Subaru', '')
            if 'B ' in band_str:
                band[i] = 'SUBARU_B'
            elif 'V' in band_str:
                band[i] = 'SUBARU_V'
            elif 'R' in band_str:
                band[i] = 'SUBARU_r'
            elif 'K_s' in band_str:
                band[i] = 'SUBARU_MOIRCS_Ks'
            elif ('i' in band_str) or ('I' in band_str):
                band[i] = 'SUBARU_i'
            elif 'z' in band_str:
                band[i] = 'SUBARU_z'
            elif 'y' in band_str:
                band[i] = 'SUBARU_HSC_y'
            elif 'NB816' in band_str:
                band[i] = 'SUBARU_HSC_NB816'
            elif 'NB921' in band_str:
                band[i] = 'SUBARU_HSC_NB921'
            elif 'NB973' in band_str:
                band[i] = 'SUBARU_HSC_NB973'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'WHT' in band_str:
            if 'U' in band_str:
                band[i] = 'WHT_U'
            elif 'R' in band_str:
                band[i] = 'WHT_R'
            elif 'g' in band_str:
                band[i] = 'WHT_g'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Keck II' in band_str:
            if 'R' in band_str:
                band[i] = 'KECKDEIMOS_R'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('(KECK)' in band_str) or ('Keck' in band_str):
            if 'G ' in band_str or 'g ' in band_str:
                band[i] = 'KECKLRIS_G'
            elif 'R ' in band_str or 'r ' in band_str:
                band[i] = 'KECKLRIS_R'
            elif ('I ' in band_str) or ('i ' in band_str):
                band[i] = 'KECKLRIS_I'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'KPNO' in band_str:
            if 'U' in band_str:
                band[i] = 'KPNOU'
            elif 'R' in band_str:
                band[i] = 'KPNOR'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Cousins' in band_str:
            band_str = band_str.replace('Cousins', '')
            if 'R' in band_str:
                band[i] = 'RC'
            elif 'I' in band_str:
                band[i] = 'IC'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('UKIDSS' in band_str) or ('UKIRT' in band_str):
            if 'K ' in band_str:
                band[i] = 'UKIRT_UKIDSS_K'
            elif 'H ' in band_str:
                band[i] = 'UKIRT_UKIDSS_H'
            elif 'J ' in band_str:
                band[i] = 'UKIRT_UKIDSS_J'
            elif 'Y ' in band_str:
                band[i] = 'UKIRT_UKIDSS_Y'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'PAN-STARRS' in band_str:
            if 'g' in band_str:
                band[i] = 'PanStarrsPS1_g'
            elif 'i' in band_str:
                band[i] = 'PanStarrsPS1_i'
            elif 'r' in band_str:
                band[i] = 'PanStarrsPS1_r'
            elif 'w' in band_str:
                band[i] = 'PanStarrsPS1_w'
            elif 'y' in band_str:
                band[i] = 'PanStarrsPS1_y'
            elif 'z' in band_str:
                band[i] = 'PanStarrsPS1_z'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Bessel' in band_str:
            if 'J' in band_str:
                band[i] = 'J'
            elif 'K' in band_str:
                band[i] = 'K'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'P200' in band_str or 'WIRC' in band_str:
            if 'J' in band_str:
                band[i] = 'P200WIRC_J'
            elif 'K' in band_str:
                band[i] = 'P200WIRC_K'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif '2MASS' in band_str:
            if 'J' in band_str:
                band[i] = '2mass.J'
            elif 'H' in band_str:
                band[i] = '2mass.H'
            elif 'K' in band_str:
                band[i] = '2mass.Ks'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'WISE' in band_str:
            if '3.4' in band_str:
                band[i] = 'WISE1'
            elif '4.6' in band_str:
                band[i] = 'WISE2'
            elif '12' in band_str:
                band[i] = 'WISE3'
            elif ('23' in band_str) or ('22' in band_str):
                band[i] = 'WISE4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRAC' in band_str:
            if '3.6' in band_str:
                band[i] = 'spitzer.irac.ch1'
            elif '4.5' in band_str:
                band[i] = 'spitzer.irac.ch2'
            elif ('5.6' in band_str) or ('5.8' in band_str):
                band[i] = 'spitzer.irac.ch3'
            elif '8.0' in band_str:
                band[i] = 'spitzer.irac.ch4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRS' in band_str:
            if '16' in band_str:
                band[i] = 'IRS16'
            elif '22' in band_str:
                band[i] = 'IRS22'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'ISOCAM' in band_str:
            if '15' in band_str:
                band[i] = 'ISOCAM15'
            elif ('LW3' in band_str):
                band[i] = 'ISOCAM15'
            elif ('LW2' in band_str):
                band[i] = 'ISOCAM7'
            elif ('7' in band_str):
                band[i] = 'ISOCAM7'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'monochromatic' in band_str:
            if '29.3microns' in band_str:
                band[i] = 'mono_29.3micron'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRAS' in band_str:
            if '12' in band_str:
                band[i] = 'IRAS12'
            elif '25' in band_str:
                band[i] = 'IRAS2'
            elif '60' in band_str:
                band[i] = 'IRAS3'
            elif '100' in band_str:
                band[i] = 'IRAS4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('MIPS' in band_str) or ('Spitzer' in band_str):
            if '24' in band_str:
                band[i] = 'MIPS1'
            elif '70' in band_str:
                band[i] = 'MIPS2'
            elif '160' in band_str:
                band[i] = 'MIPS3'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'AKARI' in band_str:
            if '11' in band_str:
                band[i] = 'AKARI_11micron'
            elif '15' in band_str:
                band[i] = 'AKARI_15micron'
            elif '18' in band_str:
                band[i] = 'AKARI_18micron'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'PACS' in band_str:
            if '70' in band_str:
                band[i] = 'PACS_blue'
            elif '100' in band_str:
                band[i] = 'PACS_green'
            elif '160' in band_str:
                band[i] = 'PACS_red'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('BLAST' in band_str):
            if '250' in band_str:
                band[i] = 'BLAST250'
            elif '350' in band_str:
                band[i] = 'BLAST350'
            elif '500' in band_str:
                band[i] = 'BLAST500'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('SPIRE' in band_str) or ('Herschel' in band_str):
            if '250' in band_str:
                band[i] = 'SPIRE250'
            elif '350' in band_str:
                band[i] = 'SPIRE350'
            elif '500' in band_str:
                band[i] = 'SPIRE500'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('SHARC' in band_str) or ('CSO' in band_str):
            if '350' in band_str:
                band[i] = 'SHARCII350'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SCUBA' in band_str:
            if '350' in band_str:
                band[i] = 'SCUBA350'
            elif '450' in band_str:
                band[i] = 'SCUBA450'
            elif '750' in band_str:
                band[i] = 'SCUBA750'
            elif '850' in band_str:
                band[i] = 'SCUBA850'
            elif ('1350' in band_str) or ('1300' in band_str):
                band[i] = 'SCUBA1350'
            elif '2000' in band_str:
                band[i] = 'SCUBA2000'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SMA' in band_str:
            if '434' in band_str:
                band[i] = 'SMA434'
            elif '870' in band_str:
                band[i] = 'SMA880'
            elif '880' in band_str:
                band[i] = 'SMA880'
            elif '1.2 mm' in band_str:
                band[i] = 'SMA1200'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('PLANCK' in band_str):
            if '350' in band_str:
                band[i] = 'Planck_857GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('LABOCA' in band_str) or ('APEX' in band_str) or ('SABOCA' in band_str):
            if '870' in band_str:
                band[i] = 'LABOCA870'
            elif ('352' in band_str) or ('350' in band_str):
                band[i] = 'SABOCA350'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'MAMBO' in band_str:
            if ('1200' in band_str) or (
                    '250' in band_str) or ('1.2' in band_str):
                band[i] = 'MAMBO1200'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('AZTEC' in band_str) or ('AzTEC' in band_str):
            if ('1100' in band_str) or ('1.1' in band_str):
                band[i] = 'AZTEC1100'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('PdBI' in band_str) or ('PDBI' in band_str) or ('EMIR' in band_str) or ('NOEMA' in band_str):
            if ('236.797' in band_str) or ('1270' in band_str) or ('239' in band_str) or ('234' in band_str) or ('1.2 mm' in band_str):
                band[i] = 'PDBI_236GHz'
            elif '1.3 mm' in band_str:
                band[i] = 'PDBI_230GHz'
            elif '225.12' in band_str:
                band[i] = 'PDBI_225.12GHz'
            elif '212.537' in band_str:
                band[i] = 'PDBI_212.537GHz'
            elif ('153.132' in band_str) or ('149.2' in band_str) or ('2 mm' in band_str) or ('2.0 mm' in band_str) or ('142' in band_str):
                band[i] = 'PDBI_149.2GHz'
            elif '129.244' in band_str:
                band[i] = 'PDBI_129.244GHz'
            elif '117' in band_str:
                band[i] = 'PDBI_117GHz'
            elif '2.8 mm' in band_str:
                band[i] = 'PDBI_107GHz'
            elif '103.8' in band_str:
                band[i] = 'PDBI_103.8GHz'
            elif ('100.8' in band_str) or ('3 mm' in band_str) or ('3.0mm' in band_str):
                band[i] = 'PDBI_100.8GHz'
            elif '96.5' in band_str:
                band[i] = 'PDBI_96.5GHz'
            elif '92.9' in band_str:
                band[i] = 'PDBI_92.9GHz'
            elif ('90.8' in band_str) or ('91.7' in band_str) or ('3.3mm' in band_str):
                band[i] = 'PDBI_91.7GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('OVRO' in band_str) or ('CARMA' in band_str):
            if ('3mm' in band_str) or ('105.4' in band_str):
                band[i] = 'OVRO_3mm'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SPT' in band_str:
            if '1.4' in band_str:
                band[i] = 'SPT1400'
            elif '2.0' in band_str:
                band[i] = 'SPT2000'
            elif ('3.3' in band_str) or ('3.0' in band_str):
                band[i] = 'SPT3300'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'VLA' in band_str:
            if ('1.4' in band_str) or ('1.49' in band_str) or (
                    '1400 MHz' in band_str):
                band[i] = 'VLA_L'
            elif ('4.89' in band_str) or ('4.49 cm' in band_str) or ('5.0' in band_str) or ('4830' in band_str):
                band[i] = 'VLA_C'
            elif ('8.48' in band_str) or ('8.7' in band_str) or ('3.55 cm' in band_str):
                band[i] = 'VLA_X_8.48GHz'
            elif '14.9' in band_str:
                band[i] = 'VLA_Ku_14.9GHz'
            elif 'K-band' in band_str and '22.0' in band_str:
                band[i] = 'VLA_K_22.0GHz'
            elif 'K-band' in band_str and '21.0' in band_str:
                band[i] = 'VLA_K_21.0GHz'
            elif 'K-band' in band_str and '23.2' in band_str:
                band[i] = 'VLA_K_23.2GHz'
            elif '23.9' in band_str:
                band[i] = 'VLA_23.9GHz'
            elif 'Ka' in band_str and '31.06' in band_str:
                band[i] = 'VLA_Ka_31.5GHz'
            elif ('Ka' in band_str) and ('31.5' in band_str):
                band[i] = 'VLA_Ka_31.5GHz'
            elif '32.3' in band_str:
                band[i] = 'VLA_32.3GHz'
            elif '32.0' in band_str:
                band[i] = 'VLA_32.0GHz'
            elif ('Ka' in band_str) and ('33.8' in band_str):
                band[i] = 'VLA_Ka_33.8GHz'
            elif 'Ka' in band_str and '34.4' in band_str:
                band[i] = 'VLA_Ka_34.4GHz'
            elif '46.6' in band_str:
                band[i] = 'VLA_46.6GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'GBT' in band_str:
            if '4830' in band_str:
                band[i] = 'GBT_4830MHz'
            elif '8.6 mm' in band_str:
                band[i] = 'GBT_8.6mm'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('Molonglo' in band_str) or ('SUMSS' in band_str):
            if '408' in band_str:
                band[i] = 'MOL_408MHz'
            elif '843' in band_str:
                band[i] = 'MOL_843MHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('Parkes' in band_str):
            if '4.85' in band_str:
                band[i] = 'Parkes_4.85GHz'
            elif '5010' in band_str:
                band[i] = 'Parkes_5GHz'
            elif '2700' in band_str:
                band[i] = 'Parkes_2.7GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('Effelsberg' in band_str):
            if '4.8' in band_str:
                band[i] = 'Effels_4.8GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        else:
            if verbose:
                print(band_str + ' is not a known bandpass and is ignored.')
            band[i] = 'REMOVE'

    return band


#
#
def _map_bands_to_pcigale_old(band, verbose=False):
    """
       Internal helping function that maps bandpass identifiers in .bsv files
       from NED to their valid bandpass identifiers in pcigale.

       Input:
           band          :    list of bandpass ids.

        Output:
           band          :    list of bandpass identifiers valid in
                              pcigale.

       STATUS:
            14.12.16: started
            02.02.17: minor update
            06.11.18: python 3.6.5 compatible
    """

    # Format band ids to CIGALE names
    unknown_band = [band[i] for i in arange(0, len(band))]
    for i in arange(0, len(band), 1):
        band_str = band[i]
        if 'CFHT' in band_str:
            if 'u' in band_str:
                band[i] = 'CFHT_u'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'GALEX' in band_str:
            if 'NUV' in band_str:
                band[i] = 'GALEX_NUV'
            elif 'FUV' in band_str:
                band[i] = 'GALEX_FUV'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SDSS' in band_str:
            if 'u' in band_str:
                band[i] = 'SDSS_u'
            elif 'g' in band_str:
                band[i] = 'SDSS_g'
            elif 'r' in band_str:
                band[i] = 'SDSS_r'
            elif 'i' in band_str:
                band[i] = 'SDSS_i'
            elif 'z' in band_str:
                band[i] = 'SDSS_z'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('ACS' in band_str):
            if ('F435W' in band_str) or ('B (HST' in band_str):
                band[i] = 'ACS_F435W'
            elif 'F475W' in band_str:
                band[i] = 'ACS_F475W'
            elif 'F502N' in band_str:
                band[i] = 'ACS_F502N'
            elif 'F550M' in band_str:
                band[i] = 'ACS_F550M'
            elif 'F555W' in band_str:
                band[i] = 'ACS_F555W'
            elif 'F606W' in band_str:
                band[i] = 'ACS_F606W'
            elif 'F625W' in band_str:
                band[i] = 'ACS_F625W'
            elif 'F658N' in band_str:
                band[i] = 'ACS_F658N'
            elif 'F660N' in band_str:
                band[i] = 'ACS_F660N'
            elif ('F775W' in band_str) or ('i (HST' in band_str):
                band[i] = 'ACS_F775W'
            elif 'F814W' in band_str:
                band[i] = 'ACS_F814W'
            elif ('F850LP' in band_str) or ('z (HST' in band_str):
                band[i] = 'ACS_F850LP'
            elif 'F892N' in band_str:
                band[i] = 'ACS_F892N'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('WFP' in band_str):
            if 'F555W' in band_str:
                band[i] = 'WFPC2_F555W'
            elif 'F702W' in band_str:
                band[i] = 'WFPC2_F702W'
            elif 'F814W' in band_str:
                band[i] = 'WFPC2_F814W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and ('WFC' in band_str):
            if 'F110W' in band_str:
                band[i] = 'HST_WFC3_F110W'
            elif 'F125W' in band_str:
                band[i] = 'HST_WFC3_F125W'
            elif 'F130N' in band_str:
                band[i] = 'HST_WFC3_F130N'
            elif 'F160W' in band_str:
                band[i] = 'HST_WFC3_F160W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('HST' in band_str) and (('NIC2' in band_str) or ('NICMOS' in band_str)):
            if 'F160W' in band_str:
                band[i] = 'NICMOS_F160W'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('Subaru' in band_str) or ('SUBARU' in band_str):
            band_str = band_str.replace('AB', '')
            band_str = band_str.replace('SUBARU', '')
            band_str = band_str.replace('Subaru', '')
            if 'B ' in band_str:
                band[i] = 'SUBARU_B'
            elif 'V' in band_str:
                band[i] = 'SUBARU_V'
            elif 'R' in band_str:
                band[i] = 'SUBARU_r'
            elif 'K_s' in band_str:
                band[i] = 'SUBARU_MOIRCS_Ks'
            elif ('i' in band_str) or ('I' in band_str):
                band[i] = 'SUBARU_i'
            elif 'z' in band_str:
                band[i] = 'SUBARU_z'
            elif 'y' in band_str:
                band[i] = 'SUBARU_HSC_y'
            elif 'NB816' in band_str:
                band[i] = 'SUBARU_HSC_NB816'
            elif 'NB921' in band_str:
                band[i] = 'SUBARU_HSC_NB921'
            elif 'NB973' in band_str:
                band[i] = 'SUBARU_HSC_NB973'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'WHT' in band_str:
            if 'U' in band_str:
                band[i] = 'WHT_U'
            elif 'R' in band_str:
                band[i] = 'WHT_R'
            elif 'g' in band_str:
                band[i] = 'WHT_g'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Keck II' in band_str:
            if 'R' in band_str:
                band[i] = 'KECKDEIMOS_R'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('(KECK)' in band_str) or ('Keck' in band_str):
            if 'G ' in band_str or 'g ' in band_str:
                band[i] = 'KECKLRIS_G'
            elif 'R ' in band_str or 'r ' in band_str:
                band[i] = 'KECKLRIS_R'
            elif ('I ' in band_str) or ('i ' in band_str):
                band[i] = 'KECKLRIS_I'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'KPNO' in band_str:
            if 'U' in band_str:
                band[i] = 'KPNOU'
            elif 'R' in band_str:
                band[i] = 'KPNOR'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Cousins' in band_str:
            band_str = band_str.replace('Cousins', '')
            if 'R' in band_str:
                band[i] = 'RC'
            elif 'I' in band_str:
                band[i] = 'IC'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('UKIDSS' in band_str) or ('UKIRT' in band_str):
            if 'K ' in band_str:
                band[i] = 'UKIRT_UKIDSS_K'
            elif 'H ' in band_str:
                band[i] = 'UKIRT_UKIDSS_H'
            elif 'J ' in band_str:
                band[i] = 'UKIRT_UKIDSS_J'
            elif 'Y ' in band_str:
                band[i] = 'UKIRT_UKIDSS_Y'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'PAN-STARRS' in band_str:
            if 'g' in band_str:
                band[i] = 'PanStarrsPS1_g'
            elif 'i' in band_str:
                band[i] = 'PanStarrsPS1_i'
            elif 'r' in band_str:
                band[i] = 'PanStarrsPS1_r'
            elif 'w' in band_str:
                band[i] = 'PanStarrsPS1_w'
            elif 'y' in band_str:
                band[i] = 'PanStarrsPS1_y'
            elif 'z' in band_str:
                band[i] = 'PanStarrsPS1_z'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'Bessel' in band_str:
            if 'J' in band_str:
                band[i] = 'J'
            elif 'K' in band_str:
                band[i] = 'K'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'P200' in band_str or 'WIRC' in band_str:
            if 'J' in band_str:
                band[i] = 'P200WIRC_J'
            elif 'K' in band_str:
                band[i] = 'P200WIRC_K'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif '2MASS' in band_str:
            if 'J' in band_str:
                band[i] = 'J_2mass'
            elif 'H' in band_str:
                band[i] = 'H_2mass'
            elif 'K' in band_str:
                band[i] = 'Ks_2mass'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'WISE' in band_str:
            if '3.4' in band_str:
                band[i] = 'WISE1'
            elif '4.6' in band_str:
                band[i] = 'WISE2'
            elif '12' in band_str:
                band[i] = 'WISE3'
            elif ('23' in band_str) or ('22' in band_str):
                band[i] = 'WISE4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRAC' in band_str:
            if '3.6' in band_str:
                band[i] = 'IRAC1'
            elif '4.5' in band_str:
                band[i] = 'IRAC2'
            elif ('5.6' in band_str) or ('5.8' in band_str):
                band[i] = 'IRAC3'
            elif '8.0' in band_str:
                band[i] = 'IRAC4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRS' in band_str:
            if '16' in band_str:
                band[i] = 'IRS16'
            elif '22' in band_str:
                band[i] = 'IRS22'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'ISOCAM' in band_str:
            if '15' in band_str:
                band[i] = 'ISOCAM15'
            if '7' in band_str:
                band[i] = 'ISOCAM7'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'monochromatic' in band_str:
            if '29.3microns' in band_str:
                band[i] = 'mono_29.3micron'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'IRAS' in band_str:
            if '12' in band_str:
                band[i] = 'IRAS1'
            elif '25' in band_str:
                band[i] = 'IRAS2'
            elif '60' in band_str:
                band[i] = 'IRAS3'
            elif '100' in band_str:
                band[i] = 'IRAS4'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('MIPS' in band_str) or ('Spitzer' in band_str):
            if '24' in band_str:
                band[i] = 'MIPS1'
            elif '70' in band_str:
                band[i] = 'MIPS2'
            elif '160' in band_str:
                band[i] = 'MIPS3'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'AKARI' in band_str:
            if '11' in band_str:
                band[i] = 'AKARI_11micron'
            elif '15' in band_str:
                band[i] = 'AKARI_15micron'
            elif '18' in band_str:
                band[i] = 'AKARI_18micron'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'PACS' in band_str:
            if '70' in band_str:
                band[i] = 'PACS_blue'
            elif '100' in band_str:
                band[i] = 'PACS_green'
            elif '160' in band_str:
                band[i] = 'PACS_red'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('SPIRE' in band_str) or ('Herschel' in band_str):
            if '250' in band_str:
                band[i] = 'SPIRE250'
            elif '350' in band_str:
                band[i] = 'SPIRE350'
            elif '500' in band_str:
                band[i] = 'SPIRE500'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('SHARC' in band_str) or ('CSO' in band_str):
            if '350' in band_str:
                band[i] = 'SHARCII350'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SCUBA' in band_str:
            if '350' in band_str:
                band[i] = 'SCUBA350'
            elif '450' in band_str:
                band[i] = 'SCUBA450'
            elif '750' in band_str:
                band[i] = 'SCUBA750'
            elif '850' in band_str:
                band[i] = 'SCUBA850'
            elif ('1350' in band_str) or ('1300' in band_str):
                band[i] = 'SCUBA1350'
            elif '2000' in band_str:
                band[i] = 'SCUBA2000'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SMA' in band_str:
            if '434' in band_str:
                band[i] = 'SMA434'
            elif '870' in band_str:
                band[i] = 'SMA880'
            elif '880' in band_str:
                band[i] = 'SMA880'
            elif '1.2 mm' in band_str:
                band[i] = 'SMA1200'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('PLANCK' in band_str):
            if '350' in band_str:
                band[i] = 'Planck_857GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('LABOCA' in band_str) or ('APEX' in band_str) or ('SABOCA' in band_str):
            if '870' in band_str:
                band[i] = 'LABOCA870'
            elif ('352' in band_str) or ('350' in band_str):
                band[i] = 'SABOCA350'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'MAMBO' in band_str:
            if ('1200' in band_str) or (
                    '250' in band_str) or ('1.2' in band_str):
                band[i] = 'MAMBO1200'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('AZTEC' in band_str) or ('AzTEC' in band_str):
            if ('1100' in band_str) or ('1.1' in band_str):
                band[i] = 'AZTEC1100'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('PdBI' in band_str) or ('PDBI' in band_str) or ('EMIR' in band_str) or ('NOEMA' in band_str):
            if ('236.797' in band_str) or ('1270' in band_str) or ('239' in band_str) or ('234' in band_str) or ('1.2 mm' in band_str):
                band[i] = 'PDBI_236GHz'
            elif '1.3 mm' in band_str:
                band[i] = 'PDBI_230GHz'
            elif '225.12' in band_str:
                band[i] = 'PDBI_225.12GHz'
            elif '212.537' in band_str:
                band[i] = 'PDBI_212.537GHz'
            elif ('153.132' in band_str) or ('149.2' in band_str) or ('2 mm' in band_str) or ('2.0 mm' in band_str) or ('142' in band_str):
                band[i] = 'PDBI_149.2GHz'
            elif '129.244' in band_str:
                band[i] = 'PDBI_129.244GHz'
            elif '117' in band_str:
                band[i] = 'PDBI_117GHz'
            elif '2.8 mm' in band_str:
                band[i] = 'PDBI_107GHz'
            elif '103.8' in band_str:
                band[i] = 'PDBI_103.8GHz'
            elif ('100.8' in band_str) or ('3 mm' in band_str) or ('3.0mm' in band_str):
                band[i] = 'PDBI_100.8GHz'
            elif '96.5' in band_str:
                band[i] = 'PDBI_96.5GHz'
            elif '92.9' in band_str:
                band[i] = 'PDBI_92.9GHz'
            elif ('90.8' in band_str) or ('91.7' in band_str) or ('3.3mm' in band_str):
                band[i] = 'PDBI_91.7GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif ('OVRO' in band_str) or ('CARMA' in band_str):
            if ('3mm' in band_str) or ('105.4' in band_str):
                band[i] = 'OVRO_3mm'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'SPT' in band_str:
            if '1.4' in band_str:
                band[i] = 'SPT1400'
            elif '2.0' in band_str:
                band[i] = 'SPT2000'
            elif ('3.3' in band_str) or ('3.0' in band_str):
                band[i] = 'SPT3300'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'VLA' in band_str:
            if ('1.4' in band_str) or ('1.49' in band_str) or (
                    '1400 MHz' in band_str):
                band[i] = 'VLA_L'
            elif ('4.89' in band_str) or ('4.49 cm' in band_str) or ('5.0' in band_str) or ('4830' in band_str):
                band[i] = 'VLA_C'
            elif ('8.48' in band_str) or ('8.7' in band_str) or ('3.55 cm' in band_str):
                band[i] = 'VLA_X_8.48GHz'
            elif '14.9' in band_str:
                band[i] = 'VLA_Ku_14.9GHz'
            elif 'K-band' in band_str and '22.0' in band_str:
                band[i] = 'VLA_K_22.0GHz'
            elif 'K-band' in band_str and '21.0' in band_str:
                band[i] = 'VLA_K_21.0GHz'
            elif 'K-band' in band_str and '23.2' in band_str:
                band[i] = 'VLA_K_23.2GHz'
            elif '23.9' in band_str:
                band[i] = 'VLA_23.9GHz'
            elif 'Ka' in band_str and '31.06' in band_str:
                band[i] = 'VLA_Ka_31.5GHz'
            elif ('Ka' in band_str) and ('31.5' in band_str):
                band[i] = 'VLA_Ka_31.5GHz'
            elif '32.3' in band_str:
                band[i] = 'VLA_32.3GHz'
            elif '32.0' in band_str:
                band[i] = 'VLA_32.0GHz'
            elif ('Ka' in band_str) and ('33.8' in band_str):
                band[i] = 'VLA_Ka_33.8GHz'
            elif 'Ka' in band_str and '34.4' in band_str:
                band[i] = 'VLA_Ka_34.4GHz'
            elif '46.6' in band_str:
                band[i] = 'VLA_46.6GHz'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        elif 'GBT' in band_str:
            if '4830' in band_str:
                band[i] = 'GBT_4830MHz'
            elif '8.6 mm' in band_str:
                band[i] = 'GBT_8.6mm'
            else:
                if verbose:
                    print(band_str + ' is not a known bandpass and is ignored.')
                band[i] = 'REMOVE'
        else:
            if verbose:
                print(band_str + ' is not a known bandpass and is ignored.')
            band[i] = 'REMOVE'

    return band


#
#
def read_pcigale_best_model(cwd, object_id):
    """
       Reads in the best pcigale model and returns z, observed frq, wvl,
       flx, and dust (IR) luminosity.

       Input:
           cwd          :    current working directory
           object_id    :    object id

       Output:
           z                :    redshift
           frq_o            :    observed frq [Hz]
           wvl_o            :    observed wvl [m]
           flx              :    flx [Jy]
           dust_luminosity  :    dust luminosity [Jy]

       STATUS:
            06.12.16: started.
            15.09.17: minor update.
            06.11.18: python 3.6.5 compatible. Extended path with 'cigale_fits'
    """

    try:
        # Open *_best_mode.fits
        hdulist = fits.open(cwd + '/' + object_id + '_best_model.fits')
        cols = hdulist[1].columns
        hdulist_data = hdulist[1].data
        hdulist_hdr = hdulist[1].header
        # Redshift
        z = hdulist_hdr['universe.redshift']
        z = float(z)
        # Luminosity distance [m]
        cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
                Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
        dl = cosmo.luminosity_distance(z)
        dl = dl.value * 3.0857E16 * 1.E6

        # Extract total SED model [nm], [mJy]
        wvl_o = np.array([row[0] for row in hdulist_data])
        flx = np.array([row[1] for row in hdulist_data])
        # Convert to [m], [Hz], [Jy]
        wvl_o = wvl_o * 1.E-9
        flx = flx * 1.E-3
        frq_o = cnsts['c'] / wvl_o

        # Extract dust SED model [W nm^-1] --> [Jy]
        index_1 = [i for i, item in enumerate(cols.names)
                   if 'dust.Umin_Umax' in item]
        index_1 = index_1[0]
        index_2 = [i for i, item in enumerate(cols.names)
                   if 'dust.Umin_Umin' in item]
        index_2 = index_2[0]
        lum_dust = (np.array([row[index_1] for row in hdulist_data])
                    + np.array([row[index_2] for row in hdulist_data]))
        # Convert specific luminosity to [W m^-1]
        lum_dust = lum_dust * 1.E9
        # Convert specific luminosity to flux [Jy] (note: no (1+z) correction)
        flx_dust = ((lum_dust / (4. * pi * dl)) / dl)
        flx_dust = flx_dust * (wvl_o * wvl_o) / cnsts['c']
        flx_dust = flx_dust / 1.E-26

        # Extract un/atteanuated stellar SED model [W nm^-1] --> [Jy]
        # Unattenuated
        index_1 = [i for i, item in enumerate(cols.names)
                   if item == 'stellar.old']
        index_1 = index_1[0]
        index_2 = [i for i, item in enumerate(cols.names)
                   if item == 'stellar.young']
        index_2 = index_2[0]
        lum_stellar_unatt = (np.array([row[index_1] for row in hdulist_data])
                             + np.array([row[index_2] for row in hdulist_data]))
        # Attenuated
        index_3 = [i for i, item in enumerate(cols.names)
                   if item == 'attenuation.stellar.old']
        index_3 = index_3[0]
        index_4 = [i for i, item in enumerate(cols.names)
                   if item == 'attenuation.stellar.young']
        index_4 = index_4[0]
        lum_stellar_att = (np.array([row[index_1] for row in hdulist_data])
                           + np.array([row[index_3] for row in hdulist_data])
                           + np.array([row[index_2] for row in hdulist_data])
                           + np.array([row[index_4] for row in hdulist_data]))
        # Convert specific luminosities to [W m^-1]
        lum_stellar_unatt = lum_stellar_unatt * 1.E9
        lum_stellar_att = lum_stellar_att * 1.E9
        # Convert specific luminosity to flux [Jy] (note: no (1+z) correction)
        # Unattenuated
        flx_stellar_unatt = ((lum_stellar_unatt / (4. * pi * dl)) / dl)
        flx_stellar_unatt = flx_stellar_unatt * (wvl_o * wvl_o) / cnsts['c']
        flx_stellar_unatt = flx_stellar_unatt / 1.E-26
        # Attenuated
        flx_stellar_att = ((lum_stellar_att / (4. * pi * dl)) / dl)
        flx_stellar_att = flx_stellar_att * (wvl_o * wvl_o) / cnsts['c']
        flx_stellar_att = flx_stellar_att / 1.E-26

        # Extract Fritz2006 AGN SED model [W nm^-1] --> [Jy]
        index = [i for i, item in enumerate(cols.names) if 'agn.fritz2006' in item]
        index = index[0]
        lum_agn = np.array([row[index] for row in hdulist_data])
        # Convert specific luminosity to [W m^-1]
        lum_agn = lum_agn * 1.E9
        # Convert specific luminosity to flux [Jy] (note: no (1+z) correction)
        flx_agn = ((lum_agn / (4. * pi * dl)) / dl)
        flx_agn = flx_agn * (wvl_o * wvl_o) / cnsts['c']
        flx_agn = flx_agn / 1.E-26

        # Extract non-thermal radio SED model [W nm^-1] --> [Jy]
        index = [i for i, item in enumerate(cols.names)
                 if item == 'radio_nonthermal']
        if len(index) > 0:
            index = index[0]
            L_radio = np.array([row[index] for row in hdulist_data])
            # Convert specific luminosity to [W m^-1]
            L_radio = L_radio * 1.E9
            # Convert specific luminosity to flux [Jy] (note: no (1+z) correction)
            flx_radio = ((L_radio / (4. * pi * dl)) / dl)
            flx_radio = flx_radio * (wvl_o * wvl_o) / cnsts['c']
            flx_radio = flx_radio / 1.E-26
        else:
            flx_radio = frq_o*0.

        # Close opened fits files
        hdulist.close()

        return {'z': z, 'frq_o': frq_o, 'wvl_o': wvl_o, 'flx': flx,
                'flx_dust': flx_dust, 'flx_agn': flx_agn, 'flx_radio': flx_radio,
                'flx_stellar_unatt': flx_stellar_unatt,
                'flx_stellar_att': flx_stellar_att}
    except IOError:
        print >> sys.stderr, "Exception: %s" % str(e)
        sys.exit(1)



#
#
def convert_bandpass_pcigale_bandpass(bandname):
    """
        Converts a bandpass in src/bandpasses to a pcigale formatted
        bandpass and stores it in ../database_builder/filters/

       Started: 22.06.19
       Updated: 23.07.19. Made sure every filter is in wavelength increasing
       order, and also set first and last entry to trans=0.
    """

    # [GHz], [um], []
    x,wvl,trans = np.loadtxt("/Users/tgreve/Dropbox/Work/local/python/trgpy/src/bandpasses/"+bandname+".dat", unpack=True, skiprows=3)


    # Convert [um] --> [AA]
    wvl[:] = [x * 1.E-6/1.E-10 for x in wvl]

    # Test whether wvl is increasing, if not sort wavelength ascending order
    if wvl[1] > wvl[2]:
        indices_sorted = np.argsort(wvl)
        wvl = np.array(wvl)[indices_sorted]
        trans = np.array(trans)[indices_sorted]

    # Sets first and last entry of trans to 0, as required by pcigale
    trans[0] = 0
    trans[len(np.array(trans))-1] = 0

    # Save to pcigale format
    #wvl = wvl[::-1]
    #trans = trans[::-1]
    target = open('/Users/tgreve/Dropbox/Work/local/python/trgpy/src/bandpasses/pcigale-bandpass/' + bandname+'.dat','w')
    target.write('# '+bandname+'\n')
    target.write('#photon'+'\n')
    target.write('# '+bandname+'\n')
    for i in np.arange(0,len(wvl),1):
        output_string = '{0:2.5f}  {1:2.5f}'.format(wvl[i],trans[i])
        print(output_string)
        output_string = output_string+'\n'
        target.write(output_string)
    target.close()

     #wvl_min=min(wvl)   # [AA]
     #wvl_max=max(wvl)   # [AA]
     #plt.ioff()
     #plt.figure()
     #plt.xlim(wvl_min,wvl_max)
     #plt.ylim(-0.1,1.1)
     #plt.xlabel('Wavelength [AA]')
     #plt.ylabel('Transmission [%]')
     #plt.title('Transmission curve')
     #plt.plot(wvl, trans,'b-')
     #plt.show()


##
##
#def extract_pcigale_points(band, frq, wvl, flx, eflx):
#    """
#       Extracts and returns (band, frq, wvl, flx, eflx) entries where band
#       is a recognised pcigale band.
#
#       Input:
#           band     :   list of bands
#           frq_o    :   central frequencies
#           wvl_o    :   central wavelengths
#           flx      :   flux
#           eflx     :   flux error
#
#
#       Output:
#           band     :   list of bands
#           frq_o    :   central frequencies
#           wvl_o    :   central wavelengths
#           flx      :   flux
#           eflx     :   flux error
#
#       Started: 19.01.17
#       Updated: 10.02.17
#    """
#    # Format band ids to pcigale bandpass names
#    band_original = [item for item in band]
#    band_pcigale = _map_bands_to_pcigale(band)
#    band = band_original
#    # Identify passbands entries unknown to pcigale
#    indices_unknown = [i for i, item in enumerate(band_pcigale) if
#                       item == 'REMOVE']
#    unknown_passbands = [band_original[i] for i in indices_unknown]
#    print('--------------')
#    print('Uknown passbands:')
#    print(unknown_passbands)
#    print('--------------')
#    # Removing passbands entries unknown to pcigale
#    indices_known = [i for i, item in enumerate(band_pcigale) if
#                     item != 'REMOVE']
#    if indices_known != []:
#        band = [band_original[i] for i in indices_known]
#        flx = [flx[i] for i in indices_known]    # [Jy]
#        eflx = [eflx[i] for i in indices_known]    # [Jy]
#        frq_o = [frq_o[i] for i in indices_known]    # [Hz]
#        wvl_o = [wvl_o[i] for i in indices_known]    # [um]
#    #    # create .mag file
#    #    if create_mag:
#    #        band_out=[band_pcigale[i] for i in indices_known]
#    #        flx_pcigale=np.array(self.flx)*1.E3    # [mJy]
#    #        wvl_o_pcigale = np.array(self.wvl_o)*1.E-6/1.E-10    # [AA]
#    #        # Convert flux errors to mJy
#    #        eflx_pcigale = np.array(self.eflx)
#    #        indices_errors = [i for i,x in enumerate(eflx_pcigale) if x > 0 and x < 99]
#    #        eflx_pcigale[indices_errors] = eflx_pcigale[indices_errors]*1.E3    # [mJy]
#    #        # -9999 is magic value for undefined flux error in pcigale
#    #        indices_undefined = [i for i,x in enumerate(eflx_pcigale) if x == -999]
#    #        eflx_pcigale[indices_undefined] = eflx_pcigale[indices_undefined] - 10098
#    #        # -999 is magic value for upper limit in pcigale
#    #        indices_upper_limits = [i for i,x in enumerate(eflx_pcigale) if x == 99]
#    #        eflx_pcigale[indices_upper_limits] = eflx_pcigale[indices_upper_limits] - 1098
#    #
#    #        create_pcigale_mag(self.object_id, self.z, band_out, wvl_o_pcigale, flx_pcigale, eflx_pcigale, path=path)
#    #
#    #    self.flx = np.array(self.flx)
#    #    self.eflx = np.array(self.eflx)
#    #    self.frq_o = np.array(self.frq_o)
#    #    self.wvl_o = np.array(self.wvl_o)
#    #
#    else:
#        print('No bands were recognised by the pcigale filter database')
#
#    return band, frq_o, wvl_o, flx, eflx
