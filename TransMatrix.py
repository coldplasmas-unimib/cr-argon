import numpy as np
from . import Levels
from typeguard import typechecked

class TransMatrix ( ):

    def __init__(self, init_value, lv : Levels.Levels ):
        self.lv = lv
        if( isinstance( init_value, np.ndarray ) ):
            assert init_value.shape[0] == self.lv.levcount
            assert init_value.shape[1] == self.lv.levcount
            self.M = np.copy( init_value )
        else:
            self.M = np.ones( ( lv.levcount, lv.levcount ) ) * init_value

    @typechecked
    def getId( self, k: int | str ):
        if( isinstance( k, str ) ):
            return self.lv.ID( k )
        return k

    @typechecked
    def __getitem__(self, k: tuple[ int | str, int | str] ):
        return self.M[ self.getId( k[0] ), self.getId( k[1] ) ]
    
    @typechecked
    def __setitem__(self, k: tuple[ int | str, int | str], v ):
        self.M[ self.getId( k[0] ), self.getId( k[1] ) ] = v

    def __mul__(self, multfor ):
        output = TransMatrix( 0, self.lv )
        if( isinstance( multfor, TransMatrix ) ):
            output.M = self.M * multfor.M
        else:
            output.M = self.M * multfor
        return output

    def __truediv__(self, divfor ):
        output = TransMatrix( 0, self.lv )
        output.M = self.M / divfor
        return output
