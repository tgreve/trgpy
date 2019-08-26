"""This is a utility module, which simply is a container for useful
routines.
"""

# --- Modules
import numpy as np



# --- Global variables
Val_UnDef = -999
Val_UL = 99
Val_LL = -99



# ---
def err_prop(x, ex, y, ey, operation='div'):
    """Returns f(x,y) and ef(x,y) using law of error
    propagation, where

            operation='mul' : f = x*y
            operation='div' : f = x/y
            operation='add' : f = x+y
            operation='sub' : f = x-y

    f and ef is assigned -999, 99, and -99 for undefined, upper
    limits, and lower limits, respectively.

    Updated: 31.01.18
    """

    x = np.array(x)
    ex = np.array(ex)
    y = np.array(y)
    ey = np.array(ey)

    # Indices where x is undefined
    indices_undefined_x = np.where(x == Val_UnDef)[0]
    # Indices where y is undefined
    indices_undefined_y = np.where(y == Val_UnDef)[0]
    # Indices where x is upper limit
    indices_ul_x = np.where(ex == Val_UL)[0]
    # Indices where y is upper limit
    indices_ul_y = np.where(ey == Val_UL)[0]
    # Indices where x is lower limit
    indices_ll_x = np.where(ex == Val_LL)[0]
    # Indices where y is upper limit
    indices_ll_y = np.where(ey == Val_LL)[0]
    # Get indices where x and y are both upper limits
    indices_ul_x_ul_y = np.intersect1d(indices_ul_x, indices_ul_y)
    # Get indices where x and y are both lower limits
    indices_ll_x_ll_y = np.intersect1d(indices_ll_x, indices_ll_y)
    # Get indices where x is upper limit and y is lower limits
    indices_ul_x_ll_y = np.intersect1d(indices_ul_x, indices_ll_y)
    # Get indices where x is lower limit and y is upper limits
    indices_ll_x_ul_y = np.intersect1d(indices_ll_x, indices_ul_y)

    #Indices where either x or y is undefined, or both are ll or ul
    indices_undefined = np.concatenate((indices_undefined_x, indices_undefined_y), axis=0)
    indices_undefined = np.concatenate((indices_undefined, indices_ll_x_ll_y), axis=0)
    indices_undefined = np.concatenate((indices_undefined, indices_ul_x_ul_y), axis=0)

    try:
        if operation == 'sum':
            f = x+y
            ef = np.sqrt( ex**2 + ey**2 )
        elif operation == 'sub':
            f = x+y
            ef = np.sqrt( ex**2 + ey**2 )
        elif operation == 'div':
            f = x/y
            ef = f*np.sqrt( (ex/x)**2 + (ey/y)**2 )
        elif operation == 'mul':
            f = x*y
            ef = f*np.sqrt( (ex/x)**2 + (ey/y)**2 )
    except ValueError:
        print("Unavailable operation")

    ef[indices_ul_x] = Val_UL
    ef[indices_ll_x] = Val_LL
    ef[indices_ul_y] = Val_LL
    ef[indices_ll_y] = Val_UL
    f[indices_undefined] = Val_UnDef
    ef[indices_undefined] = Val_UnDef

    return f, ef
