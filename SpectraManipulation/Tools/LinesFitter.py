import numpy as np
from scipy.optimize import curve_fit
from uncertainties import *
import matplotlib.pyplot as plt

class LinesFitter:

    # def __init__(self, levels, radiative_data):
    #     self.lv = levels
    #     self.tr = radiative_data

    #     self.selected_lines = [
    #         # 2p2
    #         696.7, 727.5, 826.7,
    #         # 2p3
    #         706.9, 738.6,
    #         # 2p4
    #         714.9, 795.0, 852.3,
    #         # 2p6
    #         763.7, 922.7,
    #         # 2p7
    #         867.0, 935.6,
    #         # 2p8
    #         978.7,
    #         # 2p10
    #         912.5, 966.0
    #     ]

    #     self.lines = [ l for l in self.tr.linesList() if l['from'].startswith('2p') and np.round(l['wl']) in np.round(self.selected_lines) ]

    # @staticmethod
    # def gauss(x, H, A, x0, sigma):
    #     return H + A / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # def linebyline_densities( self, wl, int_calibrated, limit, plot = False ):
    #     int = int_calibrated
    #     densities = []
    #     offset = -2
        
    #     for l in  self.lines:
    #         idx_where = np.where( wl > l['wl'] + offset + 0.5 )[0]
    #         if( len( idx_where ) < 1 ):
    #             continue
    #         idx = np.min( idx_where )
    #         idx_max = np.argmax( int[idx-3:idx+4] ) + idx - 3
    #         if( int[idx_max] < limit ):
    #             continue

    #         delta = 4
    #         try:
    #             p, pcov = curve_fit(LinesFitter.gauss, wl[idx_max-delta:idx_max+delta+1], int[idx_max-delta:idx_max+delta+1],
    #                             p0=[ 0, int[idx_max], l['wl'] + offset, 1 ])
    #         except RuntimeError:
    #             continue
            
    #         if( p[1] / l['A'] > 0.1 or p[1] < 0 or np.sqrt(pcov[1,1]) / l['A'] > 0.1 ):
    #             continue

    #         if( plot ):
    #             xs = np.linspace( wl[idx_max-2], wl[idx_max+2], 100 )
    #             plt.plot( xs, LinesFitter.gauss( xs, *p ) )
            
    #         densities.append(
    #             { 'level': l['from'], 'density': p[1] / l['A'], 'density_s': np.sqrt(pcov[1,1]) / l['A'], 'wl': l['wl'] }
    #         )
    #     return densities
    
    # def summarized_densities( self, wl, int, limit ):
    #     dens = self.linebyline_densities( wl, int, limit )
    #     summ_densities = { l: { 'lines': [ d for d in dens if d['level'] == l ] } for l in np.unique( [ d['level'] for d in dens ] ) }
    #     for l in summ_densities.keys():
    #         summ_densities[l]['density_n'] = np.mean( [ d['density'] for d in summ_densities[l]['lines'] ] )
    #         summ_densities[l]['density_s'] = max( np.std( [ d['density'] for d in summ_densities[l]['lines'] ] ), np.max( [ d['density_s'] for d in summ_densities[l]['lines'] ] ) )
    #     return summ_densities
