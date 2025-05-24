import numpy as np
from os.path import dirname
from . import Levels, TransMatrix
import pandas as pd
from .SingletonMeta import SingletonMeta

class AtomsTransitionsData_Factory(metaclass = SingletonMeta):
    def __init__(self):
        self.lv = Levels.Levels()

        self.T_gs = np.arange( 200.0, 1000.0 )

        self.data = pd.read_csv( dirname( __file__ ) + "/data/parseddata_a.csv", index_col=False ).to_dict('records')
        self.ref = TransMatrix.TransMatrix(0) ## Integers transitions matrix, such that: 0 if non defined, else defined in self.data[i]

        self.Tfacts = np.sqrt( self.T_gs / 300 )

        for i, d in enumerate( self.data ):
            self.ref[ d['from'], d['to'] ] = i
            
        print(f"Loaded {len(self.data)} rows")

    def montecarloPick( self ):
        return AtomsTransitionsData( self )

class AtomsTransitionsData:

    def __init__(self, factory: AtomsTransitionsData_Factory ):

        self.factory = factory
        self.ks = [ # Here the Montecarlo picking happens!
            self.factory.Tfacts * ( f * d['k_max'] + ( 1.0 - f ) * d['k_min'] ) for d, f in zip( factory.data, np.random.rand( len( factory.data ) ) )
        ]
        self.rev_ks = [
            self._detbal( ks_props, ks_data ) for ( ks_props, ks_data ) in zip( factory.data, self.ks )
        ]

        self._transMatrix = TransMatrix.TransMatrix(0)

    def _detbal( self, ks_props, ks_data ):
        new_from_lev = self.factory.lv[ ks_props['to'] ]
        new_to_lev = self.factory.lv[ ks_props['from'] ]
        KtoEv = 8.61732814974056E-05
        expfact = ( new_to_lev['Energy_ev'] - new_from_lev['Energy_ev'] ) / KtoEv / self.factory.T_gs
        return new_to_lev['g'] / new_from_lev['g'] * ks_data * np.exp( - expfact )

    def transMatrix( self, T_g ):
        i_frac = np.interp( T_g, self.factory.T_gs, np.arange( len( self.factory.T_gs ) ) )
        i = int( np.floor( i_frac ) )
        frac = i_frac - i


        for st in self._transMatrix.lv.all_names():
            for ed in self._transMatrix.lv.all_names():
                key = int( self.factory.ref[st,ed] )
                if( key > 0 ):
                    self._transMatrix[st,ed] = self.ks[key][ i ] * ( 1 - frac ) +  self.ks[key][ i + 1 ] * frac
                else:
                    revkey = int( self.factory.ref[ed, st] )
                    if( revkey > 0 ):
                        self._transMatrix[st,ed] = self.rev_ks[revkey][ i ] * ( 1 - frac ) +  self.rev_ks[revkey][ i + 1 ] * frac
                    else:
                        self._transMatrix[st,ed] = 0

        return self._transMatrix