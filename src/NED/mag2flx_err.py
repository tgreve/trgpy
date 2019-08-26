#!/usr/bin/env python
"""
   Fit SEDs (templates and modified black body laws) to EMGs.
"""

import numpy as np
import re, sys

if __name__=='__main__':

    m = sys.argv[1]
    dm = sys.argv[2]
    band = sys.argv[3]
    
    m = float(m)
    dm = float(dm)
    f = 10.**(23. - 0.4*(m + 48.6))
    df1 = 10.**(23. - 0.4*(m + dm + 48.6)) - f
    df2 = 10.**(23. - 0.4*(m - dm + 48.6)) - f


    if band == 'AB':
        fo = 3631.00
    if band == '2MASSJ':
        fo = 1582.23
    if band == 'JohnU':
        fo = 1810.
    if band == 'JohnJ':
        fo = 1600.
    if band == 'KPNOU':
        fo = 3631.00
    if band == 'ACSF814W':
        fo = 3631.00
    if band == 'NIC2F160W':
        fo = 1072.3
    if band == 'WIRCAM':
        fo = 1551.0
    if band == 'WHTU':
        fo = 3631.00
    if band == 'IRACCH2':
        fo = 3631.00

    f = fo*10.**(-0.4*m)
    df1 = fo*10.**(-0.4*(m + dm)) - f
    df2 = fo*10.**(-0.4*(m - dm)) - f

    print('{0:1.3e} {1:1.3e} {2:1.3e}'.format(f, df1, df2))
