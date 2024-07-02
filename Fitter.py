from standard_imports import *
from scipy.optimize import minimize
from . import Solver, LinesFitter

class Fitter:
    def __init__(self, slv: Solver, ft: LinesFitter ) -> None:
        self.slv = slv
        self.ft = ft

        self.states = ['2p9', '2p8', '2p6', '2p4']

    def second_order_differential( self, f, x_0 ):
        h = 0.005 * x_0
        return ( f(x_0+h) - 2 * f(x_0) + f(x_0-h) ) / h**2

    def sigma( self, func, x_0 ):
        return np.sqrt( 2 / self.second_order_differential( func, x_0 ) )


    def fit( self, experimental_densities, n_g = 2.687e25, T_g = 300, minimize_step = True ):
        den_tofit = np.array( experimental_densities )[ [ np.where( self.ft.levels == l)[0][0] for l in self.states ] ]

        T_es = np.linspace( 0.1, 5, 10 )
        f_es =  np.logspace( -8, -2, 10 )

        best_value = -1
        best_params = []

        for f_e in tqdm( f_es ):
            for T_e in T_es:
                    chisq = self.slv.chiSquared( self.states, den_tofit, n_g, f_e, T_e, T_g )

                    if( chisq < best_value or best_value == -1 ):
                        best_value = chisq
                        best_params = [ n_g, f_e, T_e, T_g ]

        if( minimize_step ):
            (best_params[1], best_params[2]) = minimize(
                lambda x: self.slv.chiSquared( self.states, den_tofit, best_params[0], x[0], x[1], best_params[3] ),
                [ best_params[1], best_params[2] ],
                 bounds=[
                    (1e-15, 1),
                    (0.1, np.inf)
                 ]
                 ).x

        f_e = ufloat( best_params[1], self.sigma( lambda x: self.slv.chiSquared( self.states, den_tofit, best_params[0], x, best_params[2], best_params[3] ), best_params[1] ) )
        T_e = ufloat( best_params[2], self.sigma( lambda x: self.slv.chiSquared( self.states, den_tofit, best_params[0], best_params[1], x, best_params[3] ), best_params[2] ) )

        return {
            'f_e': f_e,
            'T_e': T_e,
            'n_g': n_g,
            'T_g': T_g,
            'fitted_dens': den_tofit
        }