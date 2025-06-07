import numpy as np
import numpy as np
from . import Pick
from scipy.optimize import minimize

class Minimizer:

    def fit( self, measured_levels, measured_levels_idxs, n_g = 2.687e25, T_g = 300, f_e_guess = 10**-5, T_e_guess = 5, f_e_log = False, howmany = 100 ):

        picks = Pick.ManyPick( howmany )

        if( f_e_log ):
            toMinimize = lambda x: \
                picks.averagedChiSquared_frompars( measured_levels, measured_levels_idxs, n_g, 10**( -x[0] ), x[1], T_g ).n
            x0 = [ -np.log10(f_e_guess) , T_e_guess ]
            bounds = [ ( 1, 15 ), ( 0.11, 19.8 ) ]
        else:
            toMinimize = lambda x: \
                picks.averagedChiSquared_frompars( measured_levels, measured_levels_idxs, n_g, x[0], x[1], T_g ).n
            x0 = [ f_e_guess , T_e_guess ]
            bounds = [ ( 10**-15, 0.1 ), ( 0.11, 19.8 ) ]

        minimiz = minimize(
            toMinimize,
            x0,
            bounds = bounds
            )
        
        if( f_e_log ):
            return 10**( -minimiz.x[0] ), minimiz.x[1]
        else:
            return minimiz.x