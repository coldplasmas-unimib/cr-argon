from math import sqrt
import numpy as np
from . import UFloat

@np.vectorize
def n_( x ):
    return x.n if 'UFloat' in str( type( x ) ) else x

@np.vectorize
def s_( x ):
    return x.s if 'UFloat' in str( type( x ) ) else 0

def mean( array, exclude_nan = False ):
    if( len( array ) == 1 ):
        return array[0]
    if( exclude_nan ):
        array = np.array( array )[ ~ np.isnan( array ) ]
    mean = np.mean( array )
    compatible = True
    for i1, a1 in enumerate( array ):
        for i2, a2 in enumerate( array ):
            if( i1 < i2 ):
                continue
            if( abs( n_( a1 ) - n_( a2 ) ) > 3 * sqrt( s_( a1 )**2 + s_( a2 )**2 ) ):
                compatible = False
                break
    if( compatible ):
        std = sqrt( np.sum( s_( array )**2 ) / len( array )**2 )
    else:
        std = np.std( n_( array ), ddof = 1 ) / np.sqrt( len( array ) )

    return UFloat.UFloat( n_( mean ), std )
