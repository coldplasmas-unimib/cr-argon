import numpy as np
from . import Levels
from typeguard import typechecked
from tabulate import tabulate

class TransMatrix ( ):

    def __init__(self, init_value ):
        # print("TransMatrix initialized")
        self.lv = Levels.Levels()
        if( isinstance( init_value, np.ndarray ) ):
            assert init_value.shape[0] == self.lv.levcount
            assert init_value.shape[1] == self.lv.levcount
            self.M = np.copy( init_value )
        else:
            self.M = np.ones( ( self.lv.levcount, self.lv.levcount ) ) * init_value

    @typechecked
    def __getitem__(self, k: tuple[ int, int ] ):
        return self.M[ k[0], k[1] ]
    
    @typechecked
    def __setitem__(self, k: tuple[ int , int ], v ):
        self.M[ k[0], k[1] ] = v

    def __mul__(self, multfor ):
        output = TransMatrix( 0 )
        if( isinstance( multfor, TransMatrix ) ):
            output.M = self.M * multfor.M
        else:
            output.M = self.M * multfor
        return output

    def __truediv__(self, divfor ):
        output = TransMatrix( 0 )
        output.M = self.M / divfor
        return output
    
    def __add__(self, multfor ):
        output = self.copy()
        if( isinstance( multfor, TransMatrix ) ):
            output.M = output.M + multfor.M
        else:
            output.M = output.M + multfor
        return output
    
    def copy(self):
        output = TransMatrix( 0 )
        output.M = self.M * 1 # Multiplying by 1 guarantees that a copy is created
        return output
        
    def print(self):
        print( tabulate(
            self.M,
            headers = self.lv.all_names(),
            showindex = self.lv.all_names()
        ))