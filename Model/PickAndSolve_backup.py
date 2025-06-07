import numpy as np
import numpy as np
from scipy.linalg import null_space
from . import ElectronsTransitionsData, AtomsTransitionsData, Levels, RadiativeTransitions, TransMatrix
from .utilities import n_,s_
from scipy.optimize import minimize
from tqdm import tqdm

class PickAndSolve:

    def __init__(self):
        self.lv = Levels.Levels()

        # Load cross sections
        self.eimpact_data = ElectronsTransitionsData.ElectronsTransitionsData_Factory().montecarloPick()
        self.aimpact_data = AtomsTransitionsData.AtomsTransitionsData_Factory().montecarloPick()
        self.radiative_data = RadiativeTransitions.RadiativeTransitions().A()

    def getEquilibrium(self, n_g, f_e, T_e, T_g):

        # n_g: gas density, m^(-3)
        # f_e: electron fraction, unitless
        # T_e: electron temperature, eV
        # T_g: gas temperature, K

        # Total rate matrix
        R = ( self.eimpact_data.transMatrix( T_e ) * f_e + self.radiative_data / n_g + self.aimpact_data.transMatrix( T_g ) ).M

        # Normalize
        R = R / np.max(R)

        # Assert null diagonal
        assert np.sum(R * np.identity(self.lv.levcount)) == 0

        # Transition matrix
        T = R.T - np.diag(R.dot(np.ones(self.lv.levcount)))

        eq_densities = null_space(T)

        if (eq_densities.shape[1] > 1):
            print(
                f"Warning! More than one solution found for T_e = {T_e}; {eq_densities.shape[1]} solutions available. Displaying only the first one.")

        # self.n = eq_densities[:, 0] / eq_densities[0, 0] * n_g # Normalize such that the density of the ground state is n_g
        # self.n = eq_densities[:, 0] / eq_densities[0, 0] * 1e6 # Normalize such that the density of the ground state is 1e6
        self.n = eq_densities[:, 0] / eq_densities[0, 0] # Normalize such that the density of the ground state is 1

        return self.n

    def chiSquared( self, measured_levels, simulated_levels ):

        normfact = s_( measured_levels )
        scale_factor = np.sum( n_( measured_levels ) * simulated_levels / normfact / normfact ) / np.sum( simulated_levels * simulated_levels / normfact / normfact )

        return np.sum( np.power( ( n_( measured_levels ) - simulated_levels * scale_factor ) / normfact, 2 ) )
    
    def chiSquared_frompars( self, measured_levels, measured_levels_idxs, n_g, f_e, T_e, T_g ):
        return self.chiSquared( measured_levels, self.getEquilibrium( n_g, f_e, T_e, T_g )[measured_levels_idxs] )

    def logChiSquared( self, measured_levels, simulated_levels ):

        normfact = s_( measured_levels )
        scale_factor = np.sum( n_( measured_levels ) * simulated_levels / normfact / normfact ) / np.sum( simulated_levels * simulated_levels / normfact / normfact )

        return np.log( np.sum( np.power( ( n_( measured_levels ) - simulated_levels * scale_factor ) / normfact, 2 ) ) )
    
    def fit( self, measured_levels, measured_levels_idxs, n_g = 2.687e25, T_g = 300, minus_log_f_e_guess = -5, T_e_guess = 2, tol = 1e-9 ):

        # f_es = np.logspace( -10, 0, 10 )
        # T_es = np.linspace( 0.11, 19.8, 10 )
        # f_es_min = f_es[0]
        # T_es_min = T_es[0]
        # val_min = -1

        # for f_e in f_es:
        #     for T_e in T_es:
        #         val = self.chiSquared( measured_levels, self.getEquilibrium( n_g, f_e, T_e, T_g )[measured_levels_idxs] )

        #         if( val < val_min or val_min == -1 ):
        #             val_min = val
        #             minus_log_f_e_val = -np.log10( f_e )
        #             f_es_min = f_e
        #             T_es_min = T_e
    
        # return {
        #     'minus_log_f_e': -np.log10( f_es_min ),
        #     'f_e': f_es_min,
        #     'T_e': T_es_min,
        #     # 'fitted_data_all': fitted_data,
        #     # 'fitted_data': fitted_data[ measured_levels_idxs ]
        # }
    
        minimiz = minimize(
            lambda x: self.logChiSquared( measured_levels, self.getEquilibrium( n_g, 10**( -x[0] ), x[1], T_g )[measured_levels_idxs] ),
            [ minus_log_f_e_guess, T_e_guess ],
                bounds=[
                (1, 15),
                (0.11, 19.8)
                ],
                tol = tol
                )
        
        ( minus_log_f_e_val, T_e_val ) = minimiz.x

        fitted_data = self.getEquilibrium( n_g, 10**( -minus_log_f_e_val ), T_e_val, T_g )

        return {
            'minus_log_f_e': minus_log_f_e_val,
            'f_e': 10**( -minus_log_f_e_val ),
            'T_e': T_e_val,
            'fitted_data_all': fitted_data,
            'fitted_data': fitted_data[ measured_levels_idxs ],
            'last_result': minimiz.fun
        }

    def fitUsingCG( self, measured_levels, measured_levels_idxs, n_g = 2.687e25, T_g = 300, minus_log_f_e_guess = -5, T_e_guess = 2, tol = 1e-9 ):

        # f_es = np.logspace( -10, 0, 10 )
        # T_es = np.linspace( 0.11, 19.8, 10 )
        # f_es_min = f_es[0]
        # T_es_min = T_es[0]
        # val_min = -1

        # for f_e in f_es:
        #     for T_e in T_es:
        #         val = self.chiSquared( measured_levels, self.getEquilibrium( n_g, f_e, T_e, T_g )[measured_levels_idxs] )

        #         if( val < val_min or val_min == -1 ):
        #             val_min = val
        #             minus_log_f_e_val = -np.log10( f_e )
        #             f_es_min = f_e
        #             T_es_min = T_e
    
        # return {
        #     'minus_log_f_e': -np.log10( f_es_min ),
        #     'f_e': f_es_min,
        #     'T_e': T_es_min,
        #     # 'fitted_data_all': fitted_data,
        #     # 'fitted_data': fitted_data[ measured_levels_idxs ]
        # }
    
        minimiz = minimize(
            lambda x: self.logChiSquared( measured_levels, self.getEquilibrium( n_g, 10**( -x[0] ), x[1], T_g )[measured_levels_idxs] ),
            [ minus_log_f_e_guess, T_e_guess ],
                bounds=[
                (1, 15),
                (0.11, 19.8)
                ],
                method = 'CG',
                options = { 'gtol': tol }
                )
        
        ( minus_log_f_e_val, T_e_val ) = minimiz.x

        fitted_data = self.getEquilibrium( n_g, 10**( -minus_log_f_e_val ), T_e_val, T_g )

        return {
            'minus_log_f_e': minus_log_f_e_val,
            'f_e': 10**( -minus_log_f_e_val ),
            'T_e': T_e_val,
            'fitted_data_all': fitted_data,
            'fitted_data': fitted_data[ measured_levels_idxs ],
            'last_result': minimiz.fun
        }

    @staticmethod
    def manyPicksAndSolveUsingCG( measured_levels, measured_levels_idxs, n_g = 2.687e25, T_g = 300, minus_log_f_e_guess = -5, T_e_guess = 2, howManyPicks = 1000, tol = 1e-9  ):
        minus_log_f_es = np.zeros( howManyPicks )
        T_es = np.zeros( howManyPicks )

        for i in tqdm(range( howManyPicks )):
            result = PickAndSolve().fit( measured_levels, measured_levels_idxs, n_g, T_g, minus_log_f_e_guess, T_e_guess, tol = tol )
            minus_log_f_es[i] = result[ 'minus_log_f_e' ]
            T_es[i] = result[ 'T_e' ]

        return minus_log_f_es, T_es

    @staticmethod
    def manyPicksAndSolve( measured_levels, measured_levels_idxs, n_g = 2.687e25, T_g = 300, minus_log_f_e_guess = -5, T_e_guess = 2, howManyPicks = 1000, tol = 1e-9  ):
        minus_log_f_es = np.zeros( howManyPicks )
        T_es = np.zeros( howManyPicks )

        for i in tqdm(range( howManyPicks )):
            result = PickAndSolve().fit( measured_levels, measured_levels_idxs, n_g, T_g, minus_log_f_e_guess, T_e_guess, tol = tol )
            minus_log_f_es[i] = result[ 'minus_log_f_e' ]
            T_es[i] = result[ 'T_e' ]

        return minus_log_f_es, T_es