import numpy as np
from scipy.linalg import null_space
from . import ElectronsTransitionsData, AtomsTransitionsData, Levels, RadiativeTransitions, TransMatrix, UFloat
from .utilities import n_,s_

class Pick:

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

class ManyPick:

    def __init__(self, howmany):
        self.picks = [ Pick() for _ in range(howmany) ]

    def averagedChiSquared_frompars( self, measured_levels, measured_levels_idxs, n_g, f_e, T_e, T_g ):
        chisquareds = [ pick.chiSquared_frompars( measured_levels, measured_levels_idxs, n_g, f_e, T_e, T_g ) for pick in self.picks ]
        return UFloat.UFloat( np.mean(chisquareds), np.std(chisquareds) )

