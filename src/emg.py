import numpy as np
import csv
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

import pandas as pd

from trgpy.config import cosmo_params_standard_1
from trgpy.dictionary_transitions import freq
cosmo_params = cosmo_params_standard_1


# === Routines related to line fluxes and luminosities =======================
#
#
def line_lum(z, sdv, transition, units='prime'):
    """ PURPOSE:
            Return line luminosity for a given redshift, transition, and line
            flux.

        INPUT:
            z               :   source redshift
            sdv             :   line flux [Jy km/s]
            transition      :   line transition
            lum_line_unit   :   'prime': [lum_line] = K km/s pc^2 or
                                'solar': = [lum_line] = Lsolar
        OUTPUT:
                            :   line luminosity ([K km/s pc^2] or [Lsolar])

        STATUS:
            02.02.2015: started
            28.10.18: python 3.6.5 compatible.
            29.10.18: can take single value, array, or list as input.
            21.05.19: Fixed: can take single value, array, or list as input.
    """

    # If input not a list or an array
    if np.shape(z) == ():
        z = [z]
    if np.shape(sdv) == ():
        sdv = [sdv]
    if np.shape(transition) == ():
        transition = [transition]

    z = np.array(z)
    sdv = np.array(sdv)
    transition = np.array(transition)

    # Get luminosity distance, [Mpc]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(z)
    d_L = d_L.value

    # Get observed frequency, [GHz]
    while True:
        try:
            frq_o = []
            for i in np.arange(0,len(z)):
                frq_o.append((np.array(freq[transition[i]]) / (1. + z[i])))
            break
        except ValueError:
            print("Not a valid transition! Please try again.")

    # Remember upper limits and undefined values
    indices_undefined = [i for i, item in enumerate(
        sdv) if item == -999]
    indices_upper_limits = [i for i, item in enumerate(
        sdv) if item == 99]

    # Calculate line luminosity - remember upper limits and undefined values
    frq_o = np.array(frq_o)
    if units == 'solar':
        lum_line = (1.04E-3 * sdv * frq_o * d_L * d_L)
        if indices_undefined != []:
            for j in indices_undefined:
                lum_line[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                lum_line[j] = 99
    else:
        lum_line = (3.25E7 * sdv * (1. / (frq_o * frq_o)) * d_L * d_L
                / ((1.0 + z)**3.))
        if indices_undefined != []:
            for j in indices_undefined:
                lum_line[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                lum_line[j] = 99

    return lum_line


#
#
def lum_line_conversion(transition, line_luminosity, conversion='lprime2lsun'):
    """ PURPOSE:
       Convert line luminosities from Lprime [K km/s pc^2] to Lsolar [Lsun] or
       vice versa.

       The default conversion is Lprime to Lsun. Transition has to be a string.
       Input line_luminosity can be of any type (float, array, type) but is
       outputted as array. Even single-value inputs. Undefined line_luminosities
       and upper limits (i.e., -999 and 99) are passed on in the output.

        STATUS:
            19.08.15: Started
            21.05.19: Can take single value, array, or list as input.
    """

    # If input not a list or an array
    if np.shape(line_luminosity) == ():
        line_luminosity = [line_luminosity]
    if np.shape(transition) == ():
        transition = [transition]

    line_luminosity = np.array(line_luminosity)
    transition = np.array(transition)

    # Get rest-frame frequency ([GH]) for transition
    while True:
        try:
            line_frq_rest = []
            for i in np.arange(0,len(transition)):
                line_frq_rest.append((freq[transition[i]]))
            break
        except ValueError:
            print("Not a valid transition! Please try again.")

    # Remember upper limits and undefined values
    indices_undefined = [i for i, item in enumerate(
        line_luminosity) if item == -999]
    indices_upper_limits = [i for i, item in enumerate(
        line_luminosity) if item == 99]

    line_frq_rest = np.array(line_frq_rest)
    line_luminosity = np.array(line_luminosity)


    # Make conversion  - remember upper limits and undefined values
    if conversion == 'lsun2lprime':
        line_luminosity = ((line_luminosity * 1.E9) / 3.18E4) * \
            (100. / (line_frq_rest))**3.
        if indices_undefined != []:
            for j in indices_undefined:
                line_luminosity[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                line_luminosity[j] = 99
    else:
        line_luminosity = 3.18E4 * \
            ((line_frq_rest / 100.)**3.) * (line_luminosity / 1.E9)
        if indices_undefined != []:
            for j in indices_undefined:
                line_luminosity[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                line_luminosity[j] = 99

    return line_luminosity


#
#
def line_flux(z, transition, line_lum, line_lum_unit='prime'):
    """ PURPOSE:
            Returns the velocity-integrated line flux for a given source
            redshift, rest-frame line frequency, and line luminosity.

        INPUT:
            z               :   source redshift
            transition      :   line transition
            line_lum        :   line luminosity
            line_lum_unit   :   'prime': [lum_line] = K km/s pc^2 or
                                'solar': = [lum_line] = Lsolar
        OUTPUT:
                            :   line flux [Jy km/s]

        STATUS:
            01.02.15: started
            10.02.17: minor update
            07.11.18: python 3.6.6 compatible. Can take single values or lists.
            21.05.19: Fixed: can take single value, array, or list as input.
    """

    # If input not a list or an array
    if np.shape(z) == ():
        z = [z]
    if np.shape(line_lum) == ():
        line_lum = [line_lum]
    if np.shape(transition) == ():
        transition = [transition]

    z = np.array(z)
    line_lum = np.array(line_lum)
    transition = np.array(transition)

    # Get luminosity distance, [Mpc]
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(z)
    d_L = d_L.value

    # Get observed frequency, [GHz]
    while True:
        try:
            frq_o = []
            for i in np.arange(0,len(z)):
                frq_o.append((np.array(freq[transition[i]]) / (1. + z[i])))
            break
        except ValueError:
            print("Not a valid transition! Please try again.")

    # Remember upper limits and undefined values
    indices_undefined = [i for i, item in enumerate(
        line_lum) if item == -999]
    indices_upper_limits = [i for i, item in enumerate(
        line_lum) if item == 99]

    # Calculate line flux - remember upper limits and undefined values
    frq_o = np.array(frq_o)
    if line_lum_unit == 'solar':
        sdv =  (((line_lum / 1.04E-3) * (1. + z)) / (d_L * d_L)) / (frq_o*(1+z))
        if indices_undefined != []:
            for j in indices_undefined:
                sdv[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                sdv[j] = 99
    else:
        sdv = (line_lum / 3.25E7) * (frq_o**2) * ((1.0 + z)**3) / (d_L**2)
        if indices_undefined != []:
            for j in indices_undefined:
                sdv[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                sdv[j] = 99

    return sdv

#
#
def line_flux_conversion(frq, sdv, conversion='jansky2si'):
    """ PURPOSE:
            Converts line fluxes given in units of Jy km/s to
            units of W m^-2, or vice versa. This is particularly
            useful when dealing with FIR fine-structure lines, where
            integrated line fluxes are often times given in units of W m^-2.

        INPUT:
            frq:            :   approximate line center frequency [GHz].
            sdv             :   line flux [Jy km/s] or [W m^-2].
            conversion      :   Determines which way the conversion goes.

        OUTPUT:
                            :   integrated line flux [Jy km/s] or [W m^-2].

        STATUS:
            02.02.15: started
            10.02.17: updated
            07.11.18: python 3.6.6 compatible. Can take single values or lists.
            21.05.19: Fixed: can take single value, array, or list as input.
    """

    # If input not a list or an array
    if np.shape(sdv) == ():
        sdv = [sdv]
    if np.shape(frq) == ():
        frq = [frq]

    # Remember upper limits and undefined values
    indices_undefined = [i for i, item in enumerate(
        sdv) if item == -999]
    indices_upper_limits = [i for i, item in enumerate(
        sdv) if item == 99]

    sdv = np.array(sdv)
    frq = np.array(frq)

    if conversion == 'si2jansky':
        # [Jy km/s]
        result = (sdv / 1.E-26) * 3.E5 * (1. / (frq * 1.E9))
        if indices_undefined != []:
            for j in indices_undefined:
                result[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                result[j] = 99
    else:
        # [W m^-2]
        result = sdv * (1.E-26) * (frq * 1.E9) * (1. / 3.E5)
        if indices_undefined != []:
            for j in indices_undefined:
                result[j] = -999
        if indices_upper_limits != []:
            for j in indices_upper_limits:
                result[j] = 99

    return result




# === Routines related to emg data base ======================================


#
#
def extract_digame_csv(object_id='all', object_type='all', transition='all',
                    reference='all', verbose=False):
    """ PURPOSE:
            Extract data entries from digame.csv file according to the object_id,
            object_type and transition criteria.

        INPUT:
                object_id   :   specific object id to extract.
                                Default 'all'.
                object_type :   specific object type to extract.
                                Default 'all'
                transition  :   transition to extract.
                                Default 'all'
    """

    # Initialize data.
    data = np.zeros((10000,), dtype=[('ID', list), ('type', list),
                                    ('year', float),
                                    ('transition', list), ('z', float),
                                    ('ez', float), ('FWHM', float),
                                    ('eFWHM', float), ('SdV', float),
                                    ('eSdV', float), ('L_transition', float),
                                    ('eL_transition', float),
                                    ('magnification', float),
                                    ('reference', list),
                                    ('reference_url', list),
                                    ('NED_url', list),
                                    ('err_magnification', list),
                                    ('lir_8_1000_literature', float),
                                    ('elir_8_1000_literature', float),
                                    ('lir_8_1000_cigale', float),
                                    ('elir_8_1000_cigale', float),
                                    ('lir_40_120_cigale', float),
                                    ('elir_40_120_cigale', float)])

    #['CGCG052-037', 'ULIRG', 'HNC(1-0)', '0.02', '-999.0', '-999.0', '-999.0'  , '3.9', '0.53'    , '0.0'                  ,'0.0'                         ,'1.0', '1'    , 'Privon et al. (2015)', 'https://arxiv.org/pdf/1509.07512.pdf', ''      , '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
    #ID            , Type   , Transition, z     , error z ,  FWHM   , error FWHM, SdV  , error  SdV, Luminosity not selected,error  luminosity not selected,u    , error u, reference             , reference link                        , NED link,      ,      ,      ,      ,      ,

    # Open csv file.
    i = 0
    cr = csv.reader(open("/Users/tgreve/Dropbox/Work/local/python/trgpy/src/digame_export.csv"))
    next(cr)
    for row in cr:
        data['ID'][i] = row[0].strip()
        data['type'][i] = row[1].strip()
        data['transition'][i] = row[2].strip()
        data['z'][i] = row[3]   # Set z equal to Z_LINE
        data['ez'][i] = row[4]  # Set ez equal to ERR_Z_LINE
        data['FWHM'][i] = row[5]
        data['eFWHM'][i] = row[6]
        data['SdV'][i] = row[7]
        data['eSdV'][i] = row[8]
        data['magnification'][i] = row[11]
        data['err_magnification'][i] = row[12]
        data['reference'][i] = row[13]
        i = i + 1
    data = data[0:i]

    #    #if row[2] != "NAN":
    #        data['ID'][i] = row[0].strip()
    #        data['include'][i] = row[2].strip()
    #        data['year'][i] = row[3].strip()
    #        # If Z_LINE is -999 set z equal to Z_OPT and ez equal to ERR_Z_OPT
    #        if row[8] == '-999':
    #            data['z'][i] = row[6]
    #            data['ez'][i] = row[7]
    #        # If Z_OPT is -999 set z equal to Z_LINE and ez equal to ERR_Z_LINE
    #        if row[6] == '-999':
    #            data['z'][i] = row[8]
    #            data['ez'][i] = row[9]
    #        data['reference_url'][i] = row[24]
    #        data['NED_url'][i] = row[25]
    #        if row[18] != '':
    #            data['lir_8_1000_literature'][i] = float(row[18])
    #        if row[19] != '':
    #            data['elir_8_1000_literature'][i] = float(row[19])
    #        if row[22] != '':
    #            data['lir_8_1000_cigale'][i] = float(row[22])
    #            data['elir_8_1000_cigale'][i] = 0.2*float(row[22])
    #        if row[23] != '':
    #            data['lir_40_120_cigale'][i] = float(row[23])
    #            data['elir_40_120_cigale'][i] = 0.2*float(row[23])

    # Extract according to input criteria.
    if object_id != 'all':
        data = data[data['ID'] == object_id]
    if object_type != 'all':
        data = data[data['type'] == object_type]
    if transition != 'all':
        data = data[data['transition'] == transition]
    if reference != 'all':
        data = data[data['reference'] == reference]

    # Remove empty ID strings
    data = data[data['ID'] != '']

    if verbose:
        list_emg(data)

    return data

#
#
def extract_emg_csv(object_id='all', object_type='all', transition='all',
                    reference='all', flag_include=False, verbose=False):
    """ PURPOSE:
            Extract data entries from emg.csv file according to the object_id,
            object_type and transition criteria. If flag_include=True, only
            objects that have include=1 in the csv file are extracted.

        INPUT:
                object_id   :   specific object id to extract.
                                Default 'all'.
                object_type :   specific object type to extract.
                                Default 'all'
                transition  :   transition to extract.
                                Default 'all'

        STATUS:
            07.11.18: python 3.6.5 compatible.
            21.07.19: reads in reference url.
            29.07.19: added reference keyword.
    """

    # Initialize data.
    data = np.zeros((10000,), dtype=[('ID', list), ('type', list),
                                    ('year', float),
                                    ('transition', list), ('z', float),
                                    ('ez', float), ('FWHM', float),
                                    ('eFWHM', float), ('SdV', float),
                                    ('eSdV', float), ('L_transition', float),
                                    ('eL_transition', float),
                                    ('magnification', float),
                                    ('include', list), ('reference', list),
                                    ('reference_url', list),
                                    ('NED_url', list),
                                    ('err_magnification', list),
                                    ('lir_8_1000_literature', float),
                                    ('elir_8_1000_literature', float),
                                    ('lir_8_1000_cigale', float),
                                    ('elir_8_1000_cigale', float),
                                    ('lir_40_120_cigale', float),
                                    ('elir_40_120_cigale', float)])



    # Open csv file.
    i = 0
    cr = csv.reader(open("/Users/tgreve/Dropbox/Work/local/python/trgpy/src/EMGs-v1.3.csv"))
    #cr = csv.reader(open("/Users/tgreve/Dropbox/Work/local/python/trgpy/src/digame_export.csv"))
    for row in cr:
        if row[2] == '0' or row[2] == '1':
        #if row[2] != "NAN":
            data['ID'][i] = row[0].strip()
            data['include'][i] = row[2].strip()
            data['year'][i] = row[3].strip()
            data['type'][i] = row[4].strip()
            data['transition'][i] = row[5].strip()
            data['z'][i] = row[8]   # Set z equal to Z_LINE
            data['ez'][i] = row[9]  # Set ez equal to ERR_Z_LINE
            # If Z_LINE is -999 set z equal to Z_OPT and ez equal to ERR_Z_OPT
            if row[8] == '-999':
                data['z'][i] = row[6]
                data['ez'][i] = row[7]
            # If Z_OPT is -999 set z equal to Z_LINE and ez equal to ERR_Z_LINE
            if row[6] == '-999':
                data['z'][i] = row[8]
                data['ez'][i] = row[9]
            data['FWHM'][i] = row[10]
            data['eFWHM'][i] = row[11]
            data['SdV'][i] = row[12]
            data['eSdV'][i] = row[13]
            data['magnification'][i] = row[14]
            data['reference'][i] = row[16]
            data['reference_url'][i] = row[24]
            data['NED_url'][i] = row[25]
            data['err_magnification'][i] = row[15]
            if row[18] != '':
                data['lir_8_1000_literature'][i] = float(row[18])
            if row[19] != '':
                data['elir_8_1000_literature'][i] = float(row[19])
            if row[22] != '':
                data['lir_8_1000_cigale'][i] = float(row[22])
                data['elir_8_1000_cigale'][i] = 0.2*float(row[22])
            if row[23] != '':
                data['lir_40_120_cigale'][i] = float(row[23])
                data['elir_40_120_cigale'][i] = 0.2*float(row[23])
            i = i + 1
    data = data[0:i]

    # Extract according to input criteria.
    if object_id != 'all':
        data = data[data['ID'] == object_id]
    if object_type != 'all':
        data = data[data['type'] == object_type]
    if transition != 'all':
        data = data[data['transition'] == transition]
    if reference != 'all':
        data = data[data['reference'] == reference]
    if flag_include:
        data = data[data['include'] == '1']

    # Remove empty ID strings
    data = data[data['ID'] != '']

    if verbose:
        list_emg(data)

    return data

#
#
def list_emg(data):
    """ List the formatted content of data to terminal.

       STARTED: ?
       UPDATED: 10.02.17
    """

    # Format header.
    print(' ')
    print('%s %35s %7s %9s %9s %8s %8s %9s %11s %11s %17s' % ('ID', 'Type',
          'z', 'line', 'u', 'FWHM', 'eFWHM', 'SdV', 'eSdV', 'L', 'eL'))
    print('%s %35s %7s %7s %7s %13s %7s %12s %11s %14s %16s' % ('  ', '  ',
          ' ', '    ', ' ', '[km/s]', '[km/s]', '[Jy km/s]', '[Jy km/s]',
          '[K km/s pc^2', '[K km/s pc^2]'))
    j = 0
    for i in data:
        # Format entries with upper flux limits.
        if data['eSdV'][j] == 99:
            print('%s %s %8.4f %10s %7s %7i %7i %11.3f %11i %14.2e %15.2e'
                  % (data['ID'][j].ljust(33), data['type'][j].ljust(5),
                     data['z'][j], data['transition'][
                         j], data['magnification'][j],
                     data['FWHM'][j], data['eFWHM'][
                         j], data['SdV'][j], data['eSdV'][j],
                     data['L_transition'][j], data['eL_transition'][j]))
        # Format normal entries.
        else:
            print('%s %s %8.4f %10s %7s %7i %7i %11.3f %11.3f %14.2e %15.2e'
                  % (data['ID'][j].ljust(33), data['type'][j].ljust(5),
                     data['z'][j], data['transition'][
                         j], data['magnification'][j],
                     data['FWHM'][j], data['eFWHM'][
                         j], data['SdV'][j], data['eSdV'][j],
                     data['L_transition'][j], data['eL_transition'][j]))
        j = j + 1

#
#
def make_unique_emg_fluxes(data):
    """
       Creates a unique list of emg entries

       Fix multiple occurrences of certain sources by replacing
       with (weighted) averages (for flux, line fwhm, redhift, etc).
       Fix instances of z=-999, u=-999 etc
    """

    # Set undefined magnification factors (u=-999) unity.
    indices = [i for i, item in enumerate(
               data['magnification']) if item == -999]
    if indices != []:
        for j in indices:
            data['magnification'][j] = 1.0

    # Extract unique list of transitions.
    trans = list(set(data['transition']))
    ids = list(set(data['ID']))

    for var in trans:
        for var1 in ids:
            indices = [i for i, item in enumerate(
                       data['ID']) if item == var1]

            x = data['SdV'][indices]
            ex = data['eSdV'][indices]
            x_wa = np.average(x, weights = 1./ex**2)
            ex_wa = np.sqrt(1./sum(1./ex**2))
            data['SdV'][indices[0]] = x_wa
            data['eSdV'][indices[0]] = ex_wa
            data['ID'][indices[1:]] = ['REMOVE']*(len(indices) - 1)

            x = data['FWHM'][indices]
            ex = data['eFWHM'][indices]
            x_wa = np.average(x, weights = 1./ex**2)
            ex_wa = np.sqrt(1. / sum(1. / ex**2))
            data['FWHM'][indices[0]] = x_wa
            data['eFWHM'][indices[0]] = ex_wa

            x = data['z'][indices]
            ex = data['ez'][indices]
            x_wa = np.average(x, weights = 1./ex**2)
            ex_wa = np.sqrt(1./sum(1./ex**2))
            data['z'][indices[0]] = x_wa
            data['ez'][indices[0]] = ex_wa

    data = data[data['ID'] != 'REMOVE']

    return data


#
#
def delense_fluxes(data):
    """De-lenses fluxes and preserves undefined and/or upper limits"""

    # where eSdV is undefined, then L_transition is undefined
    indices_undefined = [i for i, item in enumerate(
        data['eSdV']) if item == -999]

    # where SdV is upper limit, then L_transition is upper limit
    indices_upper_limit = [
        i for i, item in enumerate(data['eSdV']) if item == 99]

    data['SdV'] = data['SdV'] / data['magnification']
    data['eSdV'] = data['eSdV'] / data['magnification']

    if indices_undefined != []:
        data['eSdV'][indices_undefined] = -999
    if indices_upper_limit != []:
        data['eSdV'][indices_upper_limit] = 99

    return data

#
#
def add_lum_line_to_data(data):
    """Adds line luminosity and error to data.
    08.05.2020
    """

    indices_ul =  [i for i,x in enumerate(data["eSdV"]) if x == 99]
    indices_ud =  [i for i,x in enumerate(data["eSdV"]) if x == -999]

    data["L_transition"] = line_lum(data["z"], data["SdV"], data["transition"])
    data["eL_transition"] = line_lum(data["z"], data["eSdV"], data["transition"])
    data["eL_transition"][indices_ul] = 99
    data["eL_transition"][indices_ud] = -999

    return data

#
#
def intersection_emg(data_1, data_2):
    """
       Extracts the intersection between data_1 and data_2 and returns
       the trimmed and appropriately ordered data_1 and data_2 structures
       containing the intersection entries.

    """

    ids_1 = list(data_1['ID'])
    ids_2 = list(data_2['ID'])
    intersection_indices = [i for i, item in enumerate(ids_1) if item in ids_2]
    data_1 = data_1[intersection_indices]

    ids_1 = list(data_1['ID'])
    ids_2 = list(data_2['ID'])
    intersection_indices = [i for i, item in enumerate(ids_2) if item in ids_1]
    data_2 = data_2[intersection_indices]

    return data_1, data_2

#
#
def complimentary_emg(data_1, data_2):
    """
       Extracts the non-intersection between data_1 and data_2 and returns
       the trimmed data_1 and data_2 structures containing the intersection
       entries.

    """

    ids_1 = list(data_1['ID'])
    ids_2 = list(data_2['ID'])
    intersection_indices = [i for i, item in enumerate(ids_1) if item in ids_2]
    data_1['ID'][intersection_indices] = 'REMOVE'
    data_1 = data_1[data_1['ID'] != 'REMOVE']

    ids_1 = list(data_1['ID'])
    ids_2 = list(data_2['ID'])
    intersection_indices = [i for i, item in enumerate(ids_2) if item in ids_1]
    data_2['ID'][intersection_indices] = 'REMOVE'
    data_2 = data_2[data_2['ID'] != 'REMOVE']

    return data_1, data_2


#
#
def dump_data_fields_to_csv(data, fields):
    """Writes a csv file (out.csv) of the chosen fields in data.
    """
    mydata = data[fields]
    out_file = open("out.csv", "w")
    with out_file:
        writer = csv.writer(out_file)
        writer.writerows(mydata)


#
#
def convert_file_to_csv(in_file, out_file=None):
    """
        Reads in a file with EMG data and converts it into the EMG_vx.x.csv
        format, such that it can be appended to the database at some stage.

        The default is to save the converted file as [in_file].csv, unless
        a specific out_file name is provided.

        MODIFICATION HISTORY:
            29.07.2019: started
            04.10.2019: works

        TODO:
    """

    # Read in file
    try:
        df = pd.read_csv(in_file, delim_whitespace=True, dtype=str)
        columns = df.columns

        # Check that ID, LINE, IDV, and REF are specified
        proceed = False
        if "ID" in columns and "LINE" in columns and "IDV" in columns and "REF" in columns:
            proceed = True
        else:
            print("Error: File must contain ID, LINE, IDV, and REF keys.")
    except IOError:
        print("Error: File does not appear to exist.")


    if proceed:
        # Remove duplicates in input file and calculate length
        N = len(df)

        # Define df_out
        N_out = 1000
        df_out = pd.DataFrame({"ID": ["" for i in np.arange(0,N_out)],
            "ID_ALT": ["" for i in np.arange(0,N_out)],
            "IO": [1 for i in np.arange(0,N_out)],
            "YEAR": [-999 for i in np.arange(0,N_out)],
            "TYPE": ["" for i in np.arange(0,N_out)],
            "LINE": ["" for i in np.arange(0,N_out)],
            "Z_OPT": [-999 for i in np.arange(0,N_out)],
            "ERR_Z_OPT": [-999 for i in np.arange(0,N_out)],
            "Z_LINE": [-999 for i in np.arange(0,N_out)],
            "ERR_Z_LINE": [-999 for i in np.arange(0,N_out)],
            "FWHM": [-999 for i in np.arange(0,N_out)],
            "ERR_FWHM": [-999  for i in np.arange(0,N_out)],
            "IDV": [-999 for i in np.arange(0,N_out)],
            "ERR_IDV": [-999 for i in np.arange(0,N_out)],
            "MAG": [-999 for i in np.arange(0,N_out)],
            "ERR_MAG": [-999 for i in np.arange(0,N_out)],
            "REF": ["" for i in np.arange(0,N_out)],
            "COMMENTS": ["www.digame-db.online" for i in np.arange(0,N_out)],
            "LIR_LIT": [-999 for i in np.arange(0,N_out)],
            "ERR_LIR_LIT": [-999 for i in np.arange(0,N_out)],
            "LFIR_LIT": [-999 for i in np.arange(0,N_out)],
            "ERR_FLIR_LIT": [-999 for i in np.arange(0,N_out)],
            "MSTAR": [-999 for i in np.arange(0,N_out)],
            "SFR": [-999 for i in np.arange(0,N_out)],
            "GOOD_FIT": ["" for i in np.arange(0,N_out)],
            "LIR_CIGALE": [-999 for i in np.arange(0,N_out)],
            "LFIR_CIGALE": [-999 for i in np.arange(0,N_out)],
            "REF_URL": ["" for i in np.arange(0,N_out)],
            "NED_URL": ["" for i in np.arange(0,N_out)]})

        # Fill out df_out with df
        columns = ["ID","ID_ALT","IO","YEAR","TYPE","LINE","Z_OPT","ERR_Z_OPT",
            "Z_LINE","ERR_Z_LINE", "FWHM","ERR_FWHM","IDV","ERR_IDV","MAG",
            "ERR_MAG","REF","COMMENTS","LIR_LIT","ERR_LIR_LIT", "LFIR_LIT",
            "ERR_LFIR_LIT", "MSTAR","SFR", "GOOD_FIT","LIR_CIGALE",
            "LFIR_CIGALE","REF_URL","NED_URL"]

        for clmn in columns:
            if clmn in df.columns:
                df_out[clmn] = df[clmn]


        # Fix references so that they comply with Smith et al. (XXXX)
        df_out["REF"] = df_out["REF"].str.replace("+"," et al. (")
        df_out["REF"] = df_out["REF"].astype(str)+")"


        # Trim df_out, remove duplicate entries, and save as .csv file.
        df_out = df_out[0:N]
        df_out = df_out.drop_duplicates(["ID","LINE","REF"])
        if out_file:
            df_out.to_csv(out_file, index=False)
        else:
            df_out.to_csv(in_file[:-4]+".csv", index=False)


#
#
def add_csv_to_emg(in_file, out_file=None):
    """
        Reads in a csv file (in_file) with EMG data and adds it to the end of
        EMG_vx.x.csv database file. Unless an out_file is specified in which case,
        final csv is saved to that file name (out_file).

        -Reads in .csv file to be appended and master .csv file
        -Checks for duplicate rows for ID, LINE, IDV, REF. Duplicates are not
        appended.
        -Adds empty row for every 2nd row in input .csv
        -Adds an empty row before appending new data
        -Finds overlap between input and master .csv files and drops
         overlap entries from input .csv.
        -Appends input .csv to master .csv
        -Saves master .csv to file

        MODIFICATION HISTORY:
            31.07.2019: started
            06.08.2019: basically works
            04.10.2019: Fixed a few bugs. Now appends new csv file and removes
                        duplicates. Empty lines are also removed.

        TODO:
        Find a way of adding empty lines after each source ID.
        Strange rounding 'errors'
    """

    # Read in csv file to be appended
    try:
        df = pd.read_csv(in_file)

        # Drop duplicate rows (ID .and. LINE .and. IDV .and. REF)
        df = df.drop_duplicates(["ID","LINE","IDV","REF"])

        # Add empty row after each entry
        s = pd.Series("", df.columns)
        f = lambda d: d.append(s, ignore_index=True)
        grp = np.arange(len(df))
        df =  df.groupby(grp, group_keys=False).apply(f).reset_index(drop=True)


        # Add row
        df.loc[-1] = ["","","","","","","","", "","", "","","","","",
                "","","","","", "", "","","","","","","",""]
        # Shifting index
        df.index = df.index + 1
        df.sort_index(inplace=True)

    except IOError:
        print("Error: File does not appear to exist.")

    # Read in master .csv file
    master_csv_file = "/Users/tgreve/Dropbox/Work/EMGs/web/test-scripts/EMGs-test.csv"
    try:
        df_master = pd.read_csv(master_csv_file, names=["ID",
            "ID_ALT","IO","YEAR","TYPE","LINE","Z_OPT","ERR_Z_OPT",
            "Z_LINE","ERR_Z_LINE", "FWHM","ERR_FWHM","IDV","ERR_IDV","MAG",
            "ERR_MAG","REF","COMMENTS","LIR_LIT","ERR_LIR_LIT", "LFIR_LIT",
            "ERR_LFIR_LIT", "MSTAR","SFR", "GOOD_FIT","LIR_CIGALE",
            "LFIR_CIGALE","REF_URL","NED_URL"])

    except IOError:
        print("Error: File does not appear to exist.")

    # Append df to df_master --> df_final_master
    df_final_master = df_master.append(df, ignore_index = True)
    # Remove duplicates in df_master. Also removes empty lines
    df_final_master = df_final_master.drop_duplicates(["ID","LINE","REF"],
            keep="last")

    # Save file
    if out_file:
        df_final_master.to_csv(out_file, index=False)
    else:
        df_final_master.to_csv(master_csv_file, index=False)



#
#
#def _extract_single_entry_from_emg(data):

##
##
#def add_csv_to_emg(in_file, out_file=None):
#    """
#        Reads in a csv file (in_file) with EMG data and adds it to the EMG_vx.x.csv
#        database. Unless an out_file is specified in which case, final csv
#        is save to that file name (out_file).
#
#        Every entry in in_file is checked to see whether it exists in EMG_vx.x.csv.
#        Three criteria must be fulfilled for it to exist in EMG_vx.x.csv:
#            1) Same ID
#            2) Same transition
#            3) Same reference
#        If these criteria are fulfilled, the entry is not added to EMG_vx.x.csv.
#        If one or more of the criteria are not met, the entry is added to
#        EMG_vx.x.csv at the bottom.
#
#        MODIFICATION HISTORY:
#            31.07.2019: started
