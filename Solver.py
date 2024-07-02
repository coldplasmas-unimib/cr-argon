import numpy as np
from scipy.linalg import null_space
from . import Levels, LxCatData, NistEinsteinData, AtTransData
from .utilities.ufloat_functs import n,s

class Solver:

    def __init__(self):
        self.lv = Levels.Levels()


        # Load cross sections
        self.eimpact_data = LxCatData.LxCatData( self.lv )

        self.radiative_data = NistEinsteinData.NistEinsteinData( self.lv )
        self.A = self.radiative_data.A

        self.atimpact_data = AtTransData.AtTransData( self.lv )

        # Default lines to consider
        # self.defaultLines = [
        #     ('2p2', '1s5'),
        #     ('2p6', '1s5'),
        #     ('2p6', '1s2'),
        #     ('2p6', '1s4'),
        #     ('2p4', '1s3'),
        #     ('2p9', '1s5'),
        #     ('2p2', '1s2'),
        #     ('2p10', '1s5'),
        # ]

    def findEquilibrium(self, n_g, f_e, T_e, T_g):
        # n_g: gas density, m^(-3)
        # f_e: electron fraction, unitless
        # T_e: electron temperature, eV
        # T_g: gas temperature, K
        self.n_g = n_g
        self.f_e = f_e
        self.T_e = T_e
        self.T_g = T_g

        self.Q_e = self.eimpact_data.Q_e(T_e)
        self.Q_a = self.atimpact_data.Q_a(T_g)
        self.A = self.radiative_data.A

        # Total rate matrix
        R = self.Q_e.M * f_e + self.A.M / n_g + self.Q_a.M

        # Normalize
        R = R / np.max( R )
        
        # Assert null diagonal
        assert np.sum( R * np.identity( self.lv.levcount ) ) == 0

        # Transition matrix
        T = R.T - np.diag( R.dot( np.ones( self.lv.levcount ) ) )

        eq_densities = null_space(T)

        if (eq_densities.shape[1] > 1):
            print(
                f"Warning! More than one solution found for T_e = {T_e}; {eq_densities.shape[1]} solutions available. Displaying only the first one.")
            
        # self.n = eq_densities[:, 0] / eq_densities[0, 0] * n_g # Normalize such that the density of the ground state is n_g
        self.n = eq_densities[:, 0] / eq_densities[0, 0] * 1e6 # Normalize such that the density of the ground state is 1e6

        return self.n
    
    def scaleFactor( self, nv_sel, den_tofit, norm_on = [] ):
        if( len( norm_on ) == 0 ):
            norm_on = s( den_tofit )
        return np.sum( n( den_tofit ) * nv_sel / norm_on / norm_on ) / np.sum( nv_sel * nv_sel / norm_on / norm_on )



    def chiSquared( self, states, den_tofit,  n_g, f_e, T_e, T_g ):
        nv = self.findEquilibrium( n_g, f_e, T_e, T_g )
        nv_sel = nv[ [ self.lv.ID(l) for l in states ] ]

        norm_on = s( den_tofit )

        scaling = self.scaleFactor( nv_sel, den_tofit, norm_on )
        return np.sum( np.power( ( n( den_tofit ) - nv_sel * scaling ) / norm_on, 2 ) )


    # def getSpectrum(self, n, lines=[], all_lines=False, with_lines = False):
    #     if (len(lines) == 0):
    #         lines = self.defaultLines

    #     if (all_lines):
    #         lines = []
    #         for i in self.levdict.keys():
    #             for j in self.levdict.keys():
    #                 if (i == j):
    #                     continue
    #                 if( self.levdict[i]['Energy_cm'] > self.levdict[j]['Energy_cm'] ):
    #                     lines.append( (i, j) )

    #     lambdas = []
    #     intensities = []
    #     for (i, j) in lines:
    #         lambdas.append(
    #             1.0 / (self.levdict[i]['Energy_cm'] - self.levdict[j]['Energy_cm']) * 1e7)
    #         intensities.append(
    #             self.A[self.levdict[i]['id'], self.levdict[j]['id']] * n[self.levdict[i]['id']])

    #     if( with_lines ):
    #         return np.array(lambdas), np.array(intensities), np.array( lines )
    #     return np.array(lambdas), np.array(intensities)

    # def getLambdas(self, lines=[], all_lines=False):
    #     return self.getSpectrum( np.zeros( self.levcount ), lines, all_lines )[0]
    
    # def getIntensities(self, lines=[], all_lines=False):
    #     return self.getSpectrum( np.ones( self.levcount ), lines, all_lines )[1]