#!/usr/bin/env python
"""
   Formats bandpasses (frq[GHz] wvl[micron] transmission) and saves
   as files.
"""

import numpy as np
import pylab as plt
from astropy.io import fits

c=299792458. # [m/s]

##=== Read in Planck filters===================================================================
## Read in filter transmission function
#RIMO = fits.open("/Users/tgreve/Desktop/HFI_RIMO_R1.10.fits")
#lam = RIMO[2].data['WAVENUMBER']
#frq = lam*1.E-7*c # [GHz]
#frq = frq[1:]
#wvl = [c*1.E6/(x*1.E9) for x in frq]  # GHz --> um
#trans = RIMO[2].data['TRANSMISSION']
#trans = trans[1:]
#
## Cut out sensible range
#indices_use = [i for i,x in enumerate(wvl) if x > 2200. and x < 3800.]   # 100GHz band
##indices_use = [i for i,x in enumerate(wvl) if x > 1600. and x < 2700.]   # 143GHz band
##indices_use = [i for i,x in enumerate(wvl) if x > 940. and x < 1800.]   # 217GHz band
##indices_use = [i for i,x in enumerate(wvl) if x > 640. and x < 1100.]   # 353GHz band
##indices_use = [i for i,x in enumerate(wvl) if x > 400. and x < 800.]   # 545GHz band
##indices_use = [i for i,x in enumerate(wvl) if x > 250. and x < 500.]   # 857GHz band
#wvl = [wvl[i] for i in indices_use]
#trans = [trans[i] for i in indices_use]
##=============================================================================================

##=== create tophat bandpass ===================================================================
## PDBI
#frq_o=84.6

## VLA_A
#frq_o = 1.4


# VLA_K
## 23.9 GHz (VLA), SDSSJ1148+5251
#frq_o = 2.39e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.021875    # [GHz]

## VLA_K_21.0GHz
#bandname = 'VLA_K_22.0GHz'
#frq_o = 2.1999e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 8.0    # [GHz]


## VLA_K_23.2GHz
#bandname = 'VLA_K_23.2GHz'
#frq_o = 2.31991e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 8.0    # [GHz]

## VLA_K_31.06GHz
#bandname = 'VLA_K_31.1GHz'
#frq_o = 3.1056e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 8.0    # [GHz]

# VLA Ka
## VLA_Ka_33.8Hz
#bandname = 'VLA_Ka_33.8GHz'
#frq_o = 3.38e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 8.0    # [GHz]

## JVLA Ka SDP9
#frq_o = 34.4   # [GHz]

## JVLA Ka SDP9
#frq_o = 34.4   # [GHz]
#BW = 2.0

## JVLA Ka SDP11
##frq_o = 31.5   # [GHz]
##BW = 2.0

## '32.0 GHz (EVLA) SMMJ02399-0136
#frq_o = 32.0   # [GHz]
#BW = 2.0


## '32.3 GHz (EVLA) SMMJ02399-0136
#frq_o = 32.3   # [GHz]
#BW = 2.0

## 46.6 GHz (VLA), SDSSJ1148+5251
#frq_o = 4.66e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.0375    # [GHz]



## VLA X H1413+117
#frq_o = 8.48    # [GHz]
#BW = 1.0    # [GHz]

## VLA Ku H1413+117
#frq_o = 14.9    # [GHz]
#BW = 1.0    # [GHz]

## 29.3um (7.7um restframe, IRS) SMMJ02399-0136
#frq_o = 1.02E+13    # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 2000micron SCUBA
#frq_o = 1.49896229e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 10.0    # [GHz]

## 117 GHz, VCVJ14096+5628
#frq_o = 1.40e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 92.9 GHz (IRAM)
#frq_o = 9.29e+10 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 96.5 GHz (PdBI), VCVJ14096+5628
#frq_o = 1.17e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 129.244 GHz (PdBI), SMMJ02399-0136
#frq_o = 1.29e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]


## 212.537 GHz (PdBI), SMMJ02399-0136
#frq_o = 2.13e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 225.12 GHz (PdBI), VCVJ14096+5628
#frq_o = 2.25e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 1.3 mm (PdBI), RXJ0911.4+0551
#frq_o = 2.31e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 2.8 mm (PdBI)
#frq_o = 1.07e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]


## 1270 microns (PDBI), SMMJ02399-0136
#frq_o = 2.35e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 4830 MHz (NRAO/GBT), MG0751+2716
#frq_o = 4.83e+09 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.5    # [GHz]

# 8.6mm (NRAO/GBT)
#frq_o = 34.0E9 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.5    # [GHz]

## 3mm OVRO
#frq_o = 1.40e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 1.2mm SMA
#frq_o = 2.50e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]

## 434 SMA
#frq_o = 6.91e+11 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 1.0    # [GHz]



# MOLONGLO Radio Telescope - 843MHz
# MOL_843MHz
bandname = 'MOL_843MHz'
frq_o = 843.*1.E6 # [Hz]
frq_o = frq_o*1.E-9 # [GHz]
BW = 3.E-3    # [GHz]

## MOLONGLO Radio Telescope - 408MHz
## MOL_408MHz
#bandname = 'MOL_408MHz'
#frq_o = 408.*1.E6 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 3.E-3    # [GHz]


## Parkes 64m Radio Telescope - 2700MHz
## Parkes_2.7GHz
#bandname = 'Parkes_2.7GHz'
#frq_o = 2.7E9 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.1    # [GHz]

## Parkes 64m Radio Telescope - 4.85GHz
## Parkes_4.85GHz
#bandname = 'Parkes_4.85GHz'
#frq_o = 4.85E9 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.1    # [GHz]

# Parkes 64m Radio Telescope - 5GHz
# Parkes_5GHz
#bandname = 'Parkes_5GHz'
#frq_o = 5.01E9 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.15    # [GHz]


## Effelsberg - 4.8GHz
## Effels_4.8GHz
#bandname = 'Effels_4.8GHz'
#frq_o = 4.8E9 # [Hz]
#frq_o = frq_o*1.E-9 # [GHz]
#BW = 0.1    # [GHz]



# Create tophat bandpass
frq = np.arange(frq_o-BW*1.0,frq_o+BW*1.0,0.0001)
wvl = [c*1.E6/(x*1.E9) for x in frq]  # GHz --> um
indices = [i for i, item in enumerate(frq) if frq[i] > frq_o-BW*0.5 and frq[i] < frq_o+BW*0.5]
print(indices)
trans = np.zeros(len(frq))
trans[indices]=1
#==============================================================================================

## === ISOCAM LW2 ==============================================================================
#bandname = 'ISOCAM7'
#file = np.loadtxt('raw-bandpass-files/ISO_ISO.LW2.dat',skiprows=0)
#wvl = file[:,0]
#trans = file[:,1]
#wvl[:] = [x * 1.E-10/1.E-06 for x in wvl]     # AA ---> um
#frq = [c*1.E-09/(x*1.E-06) for x in wvl]      # um --> GHz
## =============================================================================================

##=== read in and format_bandpass ==============================================================
#file = np.loadtxt('/Users/tgreve/Downloads/ISO_ISO.LW2.dat',skiprows=0)

#wvl = file[:,0]
#frq = file[:,1]
#trans = file[:,1]
##wvl[:] = [x * 1.E-03/1.E-10 for x in wvl]   # mm --> AA
##wvl[:] = [x * 1.E-06/1.E-10 for x in wvl]   # um --> AA
##wvl[:] = [x * 1.E-09/1.E-06 for x in wvl]   # nm --> um
#wvl[:] = [x * 1.E-10/1.E-06 for x in wvl]     # AA ---> um
#wvl[:] = [x * 1.E3 for x in wvl]   # mm --> um
#frq = [c*1.E-09/(x*1.E-06) for x in wvl]      # um --> GHz
##wvl = [c*1.E6/(x*1.E9) for x in frq]
##trans = [x/100. for x in trans]
##trans = 0.62*trans/max(trans)
##==============================================================================================

##=== save to dictionary_bandpass===============================================================
target = open(bandname+'.dat','w')
target.write(bandname+'\n')
target.write('frq          wvl          transmission\n')
target.write('[GHz]        [micron]                 \n')
for i in np.arange(0,len(wvl),1):
    output_string = '{0:2.5e}  {1:2.5e}  {2:2.5e}'.format(frq[i],wvl[i],trans[i])
    output_string = output_string+'\n'
    target.write(output_string)
target.close()

wvl_min=min(wvl)
wvl_max=max(wvl)
plt.ioff()
plt.figure()
plt.xlim(wvl_min,wvl_max)
plt.ylim(-0.1,1.1)
plt.xlabel('Wavelength [micron]')
plt.ylabel('Transmission [%]')
plt.title('Transmission curve')
plt.plot(wvl, trans,'b-')
plt.show()
#====================================================================================================

#=== save to pcigale format==========================================================================
wvl[:] = [x * 1.E-06/1.E-10 for x in wvl]   # um --> AA
wvl = wvl[::-1]
trans = trans[::-1]
target = open('/Users/tgreve/local/bin/cigale-v0.12.1-0769fddf935408e1d22c9f8233517f504f10ad80/database_builder/filters/'
        + bandname+'.dat','w')
target.write('# '+bandname+'\n')
target.write('#photon'+'\n')
target.write('# '+bandname+'\n')
for i in np.arange(0,len(wvl),1):
    output_string = '{0:2.5f}  {1:2.5f}'.format(wvl[i],trans[i])
    print(output_string)
    output_string = output_string+'\n'
    target.write(output_string)
target.close()

wvl_min=min(wvl)   # [AA]
wvl_max=max(wvl)   # [AA]
plt.ioff()
plt.figure()
plt.xlim(wvl_min,wvl_max)
plt.ylim(-0.1,1.1)
plt.xlabel('Wavelength [AA]')
plt.ylabel('Transmission [%]')
plt.title('Transmission curve')
plt.plot(wvl, trans,'b-')
plt.show()
#====================================================================================================
