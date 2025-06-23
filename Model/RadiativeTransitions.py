import numpy as np
import pandas as pd
from os.path import dirname
from . import TransMatrix, Levels
from .SingletonMeta import SingletonMeta


class RadiativeTransitions(metaclass=SingletonMeta):
    def __init__(self):
        data = pd.read_excel(
            dirname(__file__) + "/data/NistCoefficients.xlsx", skiprows=4, index_col=None)
        self.lv = Levels.Levels()

        self.__A = TransMatrix.TransMatrix(0)

        for _, d in data.iterrows():
            self.__A[ self.lv.ID(d['Start']), self.lv.ID(d['End'])] = d['A']

        self.evToNm = 1239.8

    def A(self):
        return self.__A.copy()

    def linesList( self ):
        lines = []
        for i in range( self.lv.levcount ):
            for j in range( self.lv.levcount ):
                if( self.__A[i,j] > 0 ):
                    lines.append( {
                        'from_id': i, 'from': self.lv.name( i ),
                        'to_id': j, 'to': self.lv.name( j ),
                        'A': self.__A[i,j],
                        'wl': self.evToNm / ( self.lv[ i ]['Energy_ev'] - self.lv[ j ]['Energy_ev'] )
                    } )
        return np.array( lines )[ np.argsort( [ l['wl'] for l in lines ] ) ]
