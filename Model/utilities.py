from uncertainties import *
from math import sqrt
import numpy as np

@np.vectorize
def n( x ):
    if( isinstance( x, list ) ):
        return [ n( xi ) for xi in x ]
    return x.n if 'uncertainties' in str( type( x ) ) else x

@np.vectorize
def s( x ):
    if( isinstance( x, list ) ):
        return [ s( xi ) for xi in x ]
    return x.s if 'uncertainties' in str( type( x ) ) else 0

@np.vectorize
def n_( x ):
    if( isinstance( x, list ) ):
        return [ n_( xi ) for xi in x ]
    return x.n

@np.vectorize
def s_( x ):
    if( isinstance( x, list ) ):
        return [ s_( xi ) for xi in x ]
    return x.s

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
            if( abs( n( a1 ) - n( a2 ) ) > 3 * sqrt( s( a1 )**2 + s( a2 )**2 ) ):
                compatible = False
                break
    if( compatible ):
        std = sqrt( np.sum( s( array )**2 ) / len( array )**2 )
    else:
        std = np.std( n( array ), ddof = 1 ) / np.sqrt( len( array ) )

    return ufloat( n( mean ), std )
