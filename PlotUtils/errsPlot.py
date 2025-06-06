import numpy as np
from matplotlib.pyplot import *
import itertools

old_plot = plot
old_bar = bar

# Utils for UFloat

def unp_n( x ):
    return np.array( [ xx.n for xx in x ] if 'UFloat' in str( type( x[0] ) ) else x )
def unp_s( x ):
    return np.array( [ xx.s for xx in x ] if 'UFloat' in str( type( x[0] ) ) else x )

# Utils for markers
_markers_list = 'ovsDPX'
_markers_cycle = itertools.cycle( _markers_list )

def next_marker():
    global _markers_cycle
    return _markers_cycle.__next__()
def reset_markers():
    global _markers_cycle
    global _markers_list
    _markers_cycle = itertools.cycle( _markers_list )
def list_markers():
    global _markers_list
    return list( _markers_list )

# Utilities for colors
def next_color( label = None ):
    line = old_plot( [], [], label = label )[0]
    return line.get_color()
def last_color():
    return gca().lines[-1].get_color()
def list_colors():
    return rcParams['axes.prop_cycle'].by_key()['color']


# Utils for data treatment

def sorted( x, y ):
    idx = np.argsort( x )
    return np.array(x)[idx], np.array(y)[idx]

def normalized( y ):
    return np.array( y ) / np.max( y )

def plot_with_optional_fmt( x, y, fmt = None, **plot_info ):
    if( fmt ):
        return old_plot( x, y, fmt, **plot_info )
    return old_plot( x, y, **plot_info )

# Utils for plotting

def unp_plot( x, y = [], *args, as_area = False, avoid_errors = False, xy_sorted = False, normalize = False, keep_color = False, **plot_info ):

    # Single-point plot
    try:
        len( x )
    except TypeError:
        x = [x]
        
    try:
        len( y )
    except TypeError:
        y = [y]

    if( keep_color ):
        plot_info['color'] = gca().lines[-1].get_color()

    if( len( x ) == 0 and len( y ) == 0 ):
        return old_plot( [], [], *args, **plot_info )    
    
    if( len( y ) == 0 ):
        y = x
        x = list( range( len( y ) ) )

    if( xy_sorted ):
        x, y = sorted( x, y )

    if( as_area ):
        return unp_plot_area( x, y, *args, **plot_info )
    
    if( normalize ):
        y = normalized( y )
    
    is_y = 'UFloat' in str( type( y[0] ) )
    is_x = 'UFloat' in str( type( x[0] ) )

    if( avoid_errors ):
        return old_plot( unp_n(x), unp_n(y), *args, **plot_info )
        


    if( is_x and is_y ):
        return errorbar( unp_n(x), unp_n(y), unp_s(y), unp_s(x), *args, **plot_info )
        
    if( is_y ):
        return errorbar( x, unp_n(y), unp_s(y), *args, **plot_info )
        
    if( is_x ):
        return errorbar( unp_n(x), y, np.zeros_like(y), unp_s(x), *args, **plot_info )
        
    else:
        return plot_with_optional_fmt( x, y, *args, **plot_info )

plot = unp_plot

def unp_plot_area( x, y, **plot_info ):
    is_y = 'UFloat' in str( type( y[0] ) )
    is_x = 'UFloat' in str( type( x[0] ) )

    x_n = unp_n( x ) if is_x else x
    y_n = unp_n( y ) if is_y else y
    y_s = unp_s( y ) if is_y else np.zeros_like( y )

    lines = plot_with_optional_fmt( x_n, y_n, **plot_info )
    fill_between( x_n, y_n - y_s, y_n + y_s, color=lines[0].get_color(), alpha=0.2, label=None )

def events( evt_xlsx ):
    y = ylim()[0] + 0.1 * np.ptp( plt.ylim() )
    for e in evt_xlsx:
        annotate( e['label'], ( e['time'], y ),rotation = 'vertical' )

def extract_key_array( keys, item ):
    if( len( keys ) == 1 ):
        return item[ keys[0] ]
    return extract_key_array( keys[1:], item[ keys[0] ] )

def plot_bykeys( x_keys, y_keys, data, *args, **kargs ):
    unp_plot(
        [ extract_key_array( x_keys, d ) for d in data ],
        [ extract_key_array( y_keys, d ) for d in data ],
        *args,
        **kargs
    )

    
def bar( x, y = [], *args, avoid_errors = False, keep_color = False, **plot_info ):

    # Single-point plot
    if( not callable( getattr( x, '__len__', None ) ) ):
        x = [x]
    if( not callable( getattr( y, '__len__', None ) ) ):
        y = [y]

    if( keep_color ):
        plot_info['color'] = get_last_color()

    is_y = 'UFloat' in str( type( y[0] ) )

    if( avoid_errors ):
        return old_plot( unp_n(x), unp_n(y), *args, **plot_info )
        
    if( is_y ):
        return old_bar( x, unp_n(y), yerr = unp_s(y), *args, **plot_info )
        
    else:
        return old_bar( x, y, *args, **plot_info )

def get_last_color():
    if( len( gca().lines ) ):
        return gca().lines[-1].get_color()
    return gca().containers[-1][-1].get_fc()