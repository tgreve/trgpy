#!/usr/bin/env python
"""
   Formats bandpasses (frq[GHz] wvl[micron] transmission) and saves
   as files.
"""

import numpy as np
from os import listdir
from os.path import isfile, join

# Read in bandpass file names
bandfiles = [f for f in listdir('.') if isfile(join('.',f)) and '.dat' in f]

# Write dictionary header
dictionary_name="dictionary_bandpass.py"
target = open(dictionary_name,'w')
target.write("from collections import OrderedDict\n")
target.write("bndpass=OrderedDict()\n")

# Format_bandpasses
for bandname in bandfiles:

    print("===========")
    print(bandname)
    # Open file
    f = open(bandname, 'r')
    
    # Read and ignore header lines
    header1 = f.readline()
    header2 = f.readline()
    header3 = f.readline()
    
    frq=[]
    wvl=[]
    trans=[]
    # Loop over lines and extract variables of interest
    for line in f:
        line = line.strip()
        columns = line.split()
        frq.append(float(columns[0]))
        wvl.append(float(columns[1]))
        trans.append(float(columns[2]))
    f.close()
    

    #frq
    output_string='[['
    N1 = 4*(len(wvl)/4)
    N2 = len(wvl)%4
    for i in np.arange(0,4*(len(wvl)/4)-3,4):
        output_string = output_string + '{0:2.5e}'.format(frq[i])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(frq[i+1])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(frq[i+2])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(frq[i+3])
        output_string = output_string + ',\n'
    
    output_string = output_string[:-1]
    output_string = output_string[:-1]
    output_string = output_string+'],\n'
    output_string = output_string+'['
    #wvl
    for i in np.arange(0,4*(len(wvl)/4)-3,4):
        output_string = output_string + '{0:2.5e}'.format(wvl[i])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(wvl[i+1])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(wvl[i+2])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(wvl[i+3])
        output_string = output_string + ',\n'
    
    output_string = output_string[:-1]
    output_string = output_string[:-1]
    output_string = output_string+'],\n'
    output_string = output_string+'['
    ##transmisssion
    for i in np.arange(0,4*(len(wvl)/4)-3,4):
        output_string = output_string + '{0:2.5e}'.format(trans[i])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(trans[i+1])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(trans[i+2])
        output_string = output_string + ','
        output_string = output_string + '{0:2.5e}'.format(trans[i+3])
        output_string = output_string + ',\n'
    
    output_string = output_string[:-1]
    output_string = output_string[:-1]
    output_string = output_string+']]\n'
    
    output_string = 'bndpass[\''+bandname[:-4]+'\']=' + output_string
    
    target = open(dictionary_name,'a')
    target.write(output_string)

target.close()
