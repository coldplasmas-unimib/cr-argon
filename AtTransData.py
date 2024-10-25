import numpy as np
import pandas as pd
from os.path import dirname
from . import Levels, TransMatrix

class AtTransData:

    def __init__(self, lv: Levels.Levels, weight_on = 'gT' ): # weight_on: 'g', 'gT' (g and maxwell), 'T' (maxwell) or '1' (uniform)
        assert weight_on in ['g', 'gT', 'T', '1', 'B']

        self.transit = {}
        self.lv = lv

        data = pd.read_excel(
            dirname( __file__ )  + "/data/NguyenChangs.xlsx", skiprows=4, index_col=None).to_dict('records')

        self.__Q_a_300K = TransMatrix.TransMatrix( 0, lv ) # MUST BE THEN CALIBRATED

        for d in data:
            if( lv.exists( d['To']) ):
                self.__Q_a_300K[ d['From'], d['To'] ] = d['k']
            else:
                split_on = []
                weights  = []

                for l in lv.all_names():
                    if( l.startswith( d['To'] ) ):
                        split_on.append( l )

                        if( weight_on == 'g' ):
                            weights.append( lv[ l ]['g'] )
                        elif( weight_on == 'gT' ):
                            weights.append( lv[ l ]['g'] * np.exp( - np.abs( lv[ l ]['Energy_ev'] - lv[ d['From'] ]['Energy_ev'] ) / 0.025 ) ) # 0.025 eV = 300 K
                        elif( weight_on == 'T' ):
                            weights.append( np.exp( - np.abs( lv[ l ]['Energy_ev'] - lv[ d['From'] ]['Energy_ev'] ) / 0.025 ) ) # 0.025 eV = 300 K
                        elif( weight_on == 'B' ):
                            deltaE = np.abs( lv[ l ]['Energy_ev'] - lv[ d['From'] ]['Energy_ev'] )
                            weights.append( ( deltaE - 0.025 ) / np.power( deltaE, 2.26 ) ) # 0.025 eV = 300 K
                        elif( weight_on == '1' ):
                            weights.append( 1 )

                weights = np.array( weights ) / np.sum( weights )

                if( len( split_on ) > 0 ):
                    for l, w in zip( split_on, weights ):
                        self.__Q_a_300K[ d['From'], l ] = d['k'] * w
                else:
                    print( f"Unknown level: {d['To']}" )
        
        # Calibrate 10^-12 cm^3 -> 10^-18 m^3
        self.__Q_a_300K = self.__Q_a_300K * 1e-18

        self.KtoEv = 8.61732814974056E-05

    def detbal( self, rev_Q_e, new_from_lev, new_to_lev, Tg_ev ):
        expfact = ( new_to_lev['Energy_ev'] - new_from_lev['Energy_ev'] ) / Tg_ev
        Q_rev = new_to_lev['g'] / new_from_lev['g'] * rev_Q_e * np.exp( - expfact )
        return Q_rev
    
    def Q_a_without_detbal( self, Tg = 300 ):
        return self.__Q_a_300K * np.sqrt( Tg / 300 )

    def Q_a(self, Tg ):
        uneven = self.__Q_a_300K * np.sqrt( Tg / 300 )

        for st in self.lv.all_names():
            for ed in self.lv.all_names():
                if( uneven[st,ed] == 0 ):
                    if( uneven[ed,st] > 0 ):
                        uneven[st,ed] = self.detbal( uneven[ed,st], self.lv[st], self.lv[ed], Tg * self.KtoEv )

        return uneven

    def Q_a_onlydetbal(self ):
        uneven = self.__Q_a_300K.copy()

        detbaltrans = []
        for st in self.lv.all_names():
            for ed in self.lv.all_names():
                if( uneven[st,ed] == 0 ):
                    if( uneven[ed,st] > 0 ):
                        uneven[st,ed] = self.detbal( uneven[ed,st], self.lv[st], self.lv[ed], 300 * self.KtoEv )
                        detbaltrans.append((st,ed))

        return detbaltrans