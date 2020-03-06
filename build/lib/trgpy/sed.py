"""
"""
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import c, L_sun

from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interpolate

from trgpy.config import cosmo_params



#
#
def get_mBB_thick_radio(z=2., SFR=100., Td=45., beta=1.5, frq_c=2000.0,
                        fnth=1.0, alpha_nt=0.7, alpha_t=0.1, frq_obs_1=1.0,
                        frq_obs_2 = 1.E5):

    """Returns the SED of the raio-IR SED from eq. 15 in Yun & Carilli (2002)"""


    # Get luminosity distance, [Mpc]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
             Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    DL = cosmo.luminosity_distance(z)

    # Redshift frequencies
    frq_obs = np.linspace(frq_obs_1, frq_obs_2)*u.GHz
    frq_rest = frq_obs*(1+z)

    # Observed radio spectrum, [Jy]
    flux_radio = (25*fnth*frq_rest.value**(-alpha_nt) + 0.71*frq_rest.value**(-alpha_t))*u.Jy

    # Observed far-IR/mm spectrum, [Jy]
    flux_mm = 1.3E-3*(frq_rest.value**3.)*(1.-np.exp(-1.0*((frq_rest.value/frq_c))**beta))/(np.exp((0.048*frq_rest.value/Td))-1.0)
    flux_mm = ((flux_mm*(1.+z)*SFR/DL.value)/DL.value)*u.Jy

    flux = flux_mm+flux_radio

    return frq_obs, frq_rest, flux_radio, flux_mm, flux



#
#
def get_specific_radio_luminosity(frq_o, flux_frq_o, frq_r, z, alpha=-0.7):
    """ PURPOSE:
            Calculates the rest-frame specific radio luminosity at frequency,
            frq_r, based on an observed radio flux, flux_frq_o, at observed
            frequency, frq_o, and a input redshift.

            A synchroton radio spectral index of alpha=-0.7
            is the default assumption but can take any value.

        INPUT:
            frq_o           :   observed frequency, [GHz]
            frq_r           :   rest-frame frequency at which luminosity should
                                be calculated, [GHz]
            z               :   source redshift
            alpha           :   synchroton radio spectral index

        OUTPUT:
            lum_frq_r       :   radio luminosities at frq_r, [W Hz^-1]

        STATUS:
            28.01.2020: started

        t.greve@ucl.ac.uk, 28 Jan, 2020
    """

    # Get luminosity distance, [Mpc]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(z)
    d_L = d_L*u.pc.to(u.m)


    radio_luminosity = 4.*np.pi*d_L*d_L*flux_frq_o

    return radio_luminosity


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
           flx          :    observed flux (Jy)
           wvl_domain   :    if False: frq domain; if True: wvl domain

        Output:
           x_r          :    restframe frq or wvl
           L_r          :    specific luminosity (W Hz^-1 or W m^-1)

       06.12.16: Started
       31.01.20: Implemented into new sed.py module
    """

    # Get luminosity distance, [Mpc]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(z)
    d_L = d_L*u.Mpc.to(u.m)
    dl = d_L.value

    if wvl_domain:
        wvl_r = x_o / (1. + z)    # [m]
        frq_r = c / (wvl_r)    # [Hz]
        L_frq_r = dl * (flx * 1.E-26) * (4. * np.pi *
                                         dl / (1. + z))    # [W Hz^-1]
        L_wvl_r = L_frq_r * c / (wvl_r * wvl_r)    # [W m^-1]
        x_r = wvl_r
        L_r = L_wvl_r
    else:
        frq_r = x_o * (1. + z)    # [Hz]
        L_frq_r = dl * (flx * 1.E-26) * (4. * np.pi *
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

       06.12.16: Started
       31.01.20: Implemented into new sed.py module
    """

    foo = UnivariateSpline(x_r, L_r, k=5, s=0)(x)

    return foo



#
#
def get_lum_from_sed(x_r, L_r, x_1, x_2, wvl_domain=False):
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

       06.12.16: Started
       31.01.20: Implemented into new sed.py module
    """

    # Is x-axis frequency or wavelength domain?
    if wvl_domain:
        x_r = x_r    # [m]
        L_r = L_r / L_sun.value    # [Lsolar m^-1]
        x1 = x_r
        y1 = L_r
        scale = 1.
    else:
        x_1 = x_1 / 1.E9    # [GHz]
        x_2 = x_2 / 1.E9    # [GHz]
        x_r = x_r / 1.E9    # [GHz]
        L_r = L_r / L_sun.value    # [Lsolar Hz^-1]
        scale = 1.E9

    # Sorting array so that x1 is increasing (for Univariate spline)
    indices = np.argsort(x_r)
    x1 = x_r[indices]
    y1 = L_r[indices]

    # Integrate SED
    lum = quad(_integrand_get_lum, x_1, x_2,
               args=(x1, y1), limit=40, epsrel=0.1)
    lum = lum[0] * scale  # [Lsolar]

    return lum


#
#
def get_flux_from_sed(x_o, flx, x_val):
    """
       Returns flux at a given frequency for a given
       observed SED. Does interpolation of provided SED.

       Input:
           x_o          :    observer-frame frequency (Hz or GHz)
           flx          :    observed SED (mJy or Jy)
           x_val        :    frequency at which to get flux (Hz or GHz)

       Output:
           flx_val      :    flux (Jy)

       31.01.20: Started
    """

    # Converting Hz to GHz
    if x_o.unit == "Hz":
        x_o = x_o*u.Hz.to(u.GHz)
    # Converting Hz to GHz
    if x_val.unit == "Hz":
        x_val = x_val*u.Hz.to(u.GHz)
    # Converting mJy to Jy
    if flx.unit == "mJy":
        flx = flx*u.mJy.to(u.Jy)

    # Sorting array so that x1 is increasing (for Univariate spline)
    indices = np.argsort(x_o)
    x1 = x_o[indices]
    y1 = flx[indices]

    flx_val = UnivariateSpline(x1, y1, k=5, s=0)(x_val)

    return flx_val
