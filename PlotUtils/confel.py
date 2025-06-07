from matplotlib import transforms
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

vec_cols = [ 'tab:green', 'tab:red']

def confidence_ellipse(xsys, n_std=1.0, color='tab:green', alpha = 0.3 ):
    x = [ i[0] for i in xsys ]
    y = [ i[1] for i in xsys ]

    sigmas2, vecs = np.linalg.eig( np.cov( x,y ) )
    sigmas = np.sqrt( sigmas2 )
    corrvec0 = np.array( [ vecs[0,0] * sigmas[0], -vecs[0,1] * sigmas[0] ] ) * n_std
    corrvec1 = np.array( [ vecs[1,0] * sigmas[1], -vecs[1,1] * sigmas[1] ] ) * n_std

    ks = np.linspace( 0, 2 * np.pi, 400 )
    xs = np.zeros( len(ks) )
    ys = np.zeros( len(ks) )
    for i, k in enumerate(ks):
        xs[i], ys[i] = corrvec0 * np.cos( k ) + corrvec1 * np.sin( k )

    plt.fill( xs + np.mean(x), ys + np.mean(y), color, alpha = alpha )

    return corrvec0, corrvec1

def project( xsys, vec0, vec1 ):
    normon = vec0[1] * vec1[0] - vec0[0] * vec1[1]
    projmat = np.array([
        [ -vec0[1] / normon, vec0[0] / normon ],
        [ -vec1[1] / normon, vec1[0] / normon ]
    ])

    return [ np.dot( projmat, i ) for i in xsys ], np.dot( projmat, vec0 ), np.dot( projmat, vec1 )

def plot_corrvec( center_x, center_y, corrvec0, corrvec1 ):
    plt.plot( [ 0 + center_x, corrvec0[0] + center_x ], [ 0 + center_y, corrvec0[1] + center_y ], color = vec_cols[0] )
    plt.plot( [ 0 + center_x, corrvec1[0] + center_x ], [ 0 + center_y, corrvec1[1] + center_y ], color = vec_cols[1] )
