from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from uncertainties import unumpy as unp
import numpy as np
from matplotlib import cm
from math import ceil
import os
import seaborn as sns
import itertools

_plot_params = 0
_plot_tred = False
_plot_ca = None
_export_folder = "exports"
_factor = 1

_markers_list = 'ovsDPX'
_markers_cycle = itertools.cycle( _markers_list )

_bar_stack = None

# Change default stuff
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['errorbar.capsize'] = 2


def init( numrows = 1, numcols = 1, init = False, size = (6,4), tred = False, seaborn = False, grid = True ):
    if( grid ):
        plt.rcParams.update({"axes.grid" : True, 'axes.axisbelow':True})
        # plt.rcParams.
    plt.figure( figsize = ( numcols * size[0] / _factor, numrows * size[1] / _factor  ) )
    if( seaborn ):
        sns.set_theme()
    global _plot_params
    global _plot_tred
    _plot_params = [ int( numrows ), int( numcols ), 0 ]
    _plot_tred = tred
    if( init or ( numrows == 1 and numcols == 1 ) ):
        next()
def set_scale_factor_raw( new_factor ):
    global _factor
    _factor = new_factor
    # Otherwise, one can use:
    # sns.set_context("paper")
    # sns.set_context("paper", font_scale=1.5)
    # sns.set_context("notebook")
    # sns.set_context("talk")
    # sns.set_context("poster")
def enable_seaborn():
    sns.set_theme()
def init_bytot( numtot, cols = 4, **args ):
    if( numtot < cols ):
        return init( 1, numtot, **args )
    return init( ceil( numtot/cols), cols, **args )
def init_bylen( array, **args ):
    return init_bytot( len( array ), **args )
def next():
    global _plot_params
    global _plot_tred
    global _plot_ca
    global _bar_stack
    _bar_stack = None

    if( _plot_params[0] * _plot_params[1] == 1 and _plot_params[2] == 1 ):
        pass
    else:
        _plot_params[2] += 1
    if( _plot_tred ):
        _plot_ca = plt.subplot( *_plot_params, projection="3d" )
    else:
        _plot_ca = plt.subplot( *_plot_params )
    return _plot_ca
def surface_plot( x, y, z ):
    global _plot_ca
    _plot_ca.plot_surface( *np.meshgrid( x, y ), z, cmap=cm.jet, linewidth=0, antialiased=False )
def goto( i_row, i_col ):
    global _plot_params
    _plot_params[2] = i_col + i_row * _plot_params[1] - 1
    next()
def hline( y, format = ':k' ):
    xl = plt.xlim()
    plt.plot( xl, [y, y], format )
    plt.xlim( xl )
def vline( x, format = ':k' ):
    yl = plt.ylim()
    plt.plot( [x, x], yl, format )
    plt.ylim( yl )
def lacking( text="" ):
    plt.plot( [0, 1], [0, 1], 'r', lw=5)
    plt.plot( [0, 1], [1, 0], 'r', lw=5)
    plt.xlim( 0, 1 )
    plt.ylim( 0, 1 )
    plt.xticks([])
    plt.yticks([])
    plt.annotate( "Lacking", [0.5, 0.8], ha='center', fontsize=40)
    plt.annotate( "plot!", [0.5, 0.2], ha='center', fontsize=40)
    plt.annotate( text, [0.5, 0], ha='center', fontsize=15)
    plt.box(False)
def set_export_folder( folder ):
    global _export_folder
    _export_folder = folder
def export( filename, draft = False, transparent = True ):
    global _export_folder

    if not os.path.exists( _export_folder ):
       os.makedirs( _export_folder )

    if( draft ):
        plt.annotate( "DRAFT", [ 0.5, 0.7 ], ha='center', va='center', color='red', weight='bold', fontsize=40, xycoords= 'figure fraction' )
        plt.annotate( "All panels should be considered draft", [ 0.5, 0.5 ], ha='center', va='center', color='red', fontsize=20, xycoords= 'figure fraction' )

    plt.savefig( _export_folder + "/" + filename + ".png", bbox_inches="tight", dpi=600, transparent = transparent )
    plt.savefig( _export_folder + "/" + filename + ".pdf", bbox_inches="tight", transparent = transparent )

    print(_export_folder + "/" + filename + ".pdf")

def marker():
    global _markers_cycle
    return _markers_cycle.__next__()
def markers_reset():
    global _markers_cycle
    global _markers_list
    _markers_cycle = itertools.cycle( _markers_list )
def markers_list():
    global _markers_list
    return list( _markers_list )

def colors_list():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def bar( x, y, stack = False, **args ):
    global _bar_stack

    if( stack ):
        if( _bar_stack is None ):
            print("First bar plot cannot be stacked!")
        if( len( _bar_stack ) != len( y ) ):
            print("Only bar with the same size can be stacked!")
        args['bottom'] = _bar_stack
    else:
        _bar_stack = np.zeros_like( y )

    plt.bar( x, y, **args )
    
    if( stack ):
        _bar_stack += y
    else:
        _bar_stack = y