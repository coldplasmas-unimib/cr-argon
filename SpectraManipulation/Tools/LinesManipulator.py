import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from standard_imports import mean, ufloat
from plots import splt

class LinesManipulator:
    def __init__(self):
        pass

    def realign_wl( self, wl, int, plot = False, return_lineslist = False, polydeg = 2 ):
        lines = [696.7115300076855, 706.8926682008145, 727.4693122281092, 738.5763937032756, 763.6948510434079, 795.0092910218337, 826.6513295441124, 852.3493706233827, 912.51615, 922.67170 ]
        thresh = 0.04 * np.max(int)


        lines_theo = []
        lines_exp = []

        for l in lines:
            idx = np.where( wl > l )[0]
            if( len( idx ) < 1 ):
                continue
            idx = idx[0]
            idx = np.argmax( int[idx-10:idx+11] ) + idx - 10

            if( int[idx] > thresh ):
                pars = np.polyfit( wl[idx-2:idx+3], int[idx-2:idx+3], 2 )
                lines_theo.append( l )
                lines_exp.append( - pars[1] / 2 / pars[0] )

                # splt.next()
                # plt.plot( wl[idx-2:idx+3], int[idx-2:idx+3], 'o' )
                # plt.plot( wl[idx-2:idx+3], np.polyval( pars, wl[idx-2:idx+3] ), '--' )
                # plt.plot( - pars[1] / 2 / pars[0], np.polyval( pars, - pars[1] / 2 / pars[0]), 'o' )


        p = np.polyfit( lines_exp, lines_theo, polydeg )

        if( plot ):
            plt.plot( lines_exp, lines_theo, 'o' )
            plt.plot( lines_exp, np.polyval( p, lines_exp ), '--', label = f"{p[0]:.2f}x${{}}^2$ {p[1]:+.2f}x {p[2]:+.2f}" )
            plt.legend( loc = 'upper left')
            plt.xlabel("Peaks positions (nm, experimental)")
            plt.ylabel("Peaks positions (nm, theoretical)")

        print( f"Realigned with y = {p[0]:.2f}x^2 {p[1]:+.2f}x {p[2]:+.2f}" )

        if( return_lineslist ):
            return np.polyval( p, wl ), lines_theo
        return np.polyval( p, wl )
    
    @staticmethod
    def gauss(x, H, A, sigma):
        return H + A / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / (2 * sigma**2) )
    
    def estimate_amplitude( self, wl_centered, int, plot = False, error = False ):
        fallback = ufloat( -1, 0 ) if error else -1

        idx_where = np.where( wl_centered > 0 )[0]
        if( len( idx_where ) < 1 ):
            return fallback
        idx = np.min( idx_where )

        delta = 4
        
        if( plot ):
            plt.plot(  wl_centered[idx-delta:idx+delta], int[idx-delta:idx+delta], 'o')
            xs = np.linspace( wl_centered[idx-delta], wl_centered[idx+delta], 100 )
            
        try:
            p, pcov = curve_fit( LinesManipulator.gauss, wl_centered[idx-delta:idx+delta], int[idx-delta:idx+delta],
                            p0=[ 0, int[idx], 1 ],
                            bounds=(
                                [ -0.1, 0, 0 ],
                                [ 0.1, np.inf, np.inf ]
                            ))
        except RuntimeError:
            return fallback


        hvert = LinesManipulator.gauss( 0, *p )
        if( hvert < 4 * np.abs( p[0] ) or hvert < 2 * p[2] ):
            return fallback

        if( plot ):
            plt.plot( xs, LinesManipulator.gauss( xs, *p ), '--' )
            plt.plot( [0,0], [p[0], hvert], 'g')

        if(error):
            return ufloat( p[1], np.sqrt( pcov[1,1] ) )
        return p[1]
    
    
    @staticmethod
    def doublegauss(x, H, A1, sigma1, A2, sigma2, delta):
        return H + A1 / np.sqrt(2 * np.pi) / sigma1 * np.exp(-x**2 / (2 * sigma1**2) ) + A2 / np.sqrt(2 * np.pi) / sigma2 * np.exp(-(x-delta)**2 / (2 * sigma2**2) )
    
    def estimate_amplitude_double( self, wl_centered, spec, delta_wl, plot = False, error = False ):
        fallback = ufloat( -1, 0 ) if error else -1

        idx_where = np.where( wl_centered > 0 )[0]
        if( len( idx_where ) < 1 ):
            return fallback

        idx = np.min( idx_where )
        delta_i = int( delta_wl / ( wl_centered[idx+1] - wl_centered[idx] ) )

        delta = 4
        
        if( plot ):
            plt.plot(  wl_centered[idx-delta:idx+delta_i+delta], spec[idx-delta:idx+delta_i+delta], 'o')
            xs = np.linspace( wl_centered[idx-delta], wl_centered[idx+delta_i+delta], 200 )
            
        try:
            p, pcov = curve_fit( LinesManipulator.doublegauss, wl_centered[idx-delta:idx+delta_i+delta], spec[idx-delta:idx+delta_i+delta],
                            p0=[ 0, spec[idx], 1, spec[idx+delta_i], 1, delta_wl ],
                            bounds=(
                                [ -0.1, 0, 0, 0, 0, delta_wl - 0.01 ],
                                [ 0.1, np.inf, np.inf, np.inf, np.inf, delta_wl + 0.01 ]
                            ))
        except RuntimeError:
            return fallback

        if( plot ):
            plt.plot( xs, LinesManipulator.doublegauss( xs, *p ), '--' )

        res1 = ufloat( p[1], np.sqrt( pcov[1,1] ) ) if error else p[1]
        hvert1 = LinesManipulator.doublegauss( 0, *p )
        if( hvert1 < 4 * np.abs( p[0] ) or hvert1 < 2 * p[2] ):
            res1 = fallback
        elif(plot):
            plt.plot( [0,0], [p[0], hvert1], 'g')
    
        res2 = ufloat( p[3], np.sqrt( pcov[3,3] ) ) if error else p[3]
        hvert2 = LinesManipulator.doublegauss( delta_wl, *p )
        if( hvert2 < 4 * np.abs( p[0] ) or hvert2 < 2 * p[4] ):
            res2 = fallback
        elif(plot):
            plt.plot( [delta_wl,delta_wl], [p[0], hvert2], 'g')

        return res1, res2
    
    def remove_baseline( self, int ):
        return int - np.mean( int[:185] )
        
    def estimate_calibration( self, wl_realigned, int_withoutbaseline, base_calib, all_lines, polydeg = 1 ):
        lines = [ '2p2->1s5', '2p2->1s4', '2p2->1s2', '2p3->1s5', '2p3->1s4', '2p4->1s5', '2p4->1s3', '2p4->1s2', '2p6->1s5', '2p6->1s2', '2p10->1s5', '2p10->1s4' ]
        lines_calib = [ l for l in all_lines if f"{l['from']}->{l['to']}" in lines ]
        assert( len( lines ) == len( lines_calib ) )

        den_dict = {}

        for l in lines_calib:
            ampl = self.estimate_amplitude( wl_realigned - l['wl'], int_withoutbaseline )
            if( ampl > 0 ):
                if( l['from'] not in den_dict.keys() ):
                    den_dict[ l['from'] ] = []
                den_dict[ l['from'] ].append( { 'wl': l['wl'], 'den': ampl / l['A'] } )
        
        xs = np.linspace( 650, 980 )

        theocal = np.log10( np.interp( xs, base_calib['wl'], base_calib['f_normalized'] ) )

        def calfunc( xxx, *pars ):
            return [
                pars[ int(x[0]) + polydeg + 1 ] + np.interp( x[1], xs, theocal ) + np.polyval( pars[:polydeg+1], x[1] ) for x in xxx
            ]
        
        xf = []
        yf = []
        i = 0
        
        for lv in den_dict.keys():
            if( len( den_dict[lv] ) > 1 ):
                wls = [ l['wl'] for l in den_dict[lv] ]
                kkk = - np.log10( np.array( [ l['den'] for l in den_dict[lv] ] ) )
                
                for wl, kk in zip( wls, kkk ):
                    xf.append( ( i, wl ) )
                    yf.append( kk )
                i += 1
        
        p,_ = curve_fit( calfunc, xf, yf, p0 = [ *np.full( polydeg+1, 0 ), *np.full( i, 2 ) ], bounds=(
            ( [ *np.full( polydeg, -1 ), -100, *np.full( i, 0 ) ],
            [ *np.full( polydeg, 1 ), 100, *np.full( i, 5 ) ] )
        ) )

        newcalib = {
            'wl': xs,
            'f_normalized': 10**( theocal + np.polyval( p[:polydeg+1], xs ) )
        }
        int_calib = int_withoutbaseline * np.interp( wl_realigned, newcalib['wl'], newcalib['f_normalized'] )

        return {
            'den_dict': den_dict,
            'xf': [ x[1] for x in xf ],
            'ys': [ y - p[ x[0] + polydeg +1] for x, y in zip( xf, yf ) ],
            'label': [ x[0] for x in xf ],
            'p': p,
            'newcalib': newcalib,
            'int_calib': int_calib
        }
    
    def estimate_densities( self, wl_realigned, int_calibrated, all_lines, plot = False ):
        lines = [ '2p2->1s5', '2p3->1s5', '2p4->1s5', '2p2->1s4', '2p3->1s4', '2p6->1s5', '2p4->1s3', '2p2->1s2', '2p4->1s2', '2p10->1s5', '2p6->1s2', '2p10->1s4' ]
        lines_double = [ '2p1->1s2', '2p5->1s4', '2p6->1s4', '2p8->1s5', '2p7->1s4', '2p9->1s5', '2p3->1s2', '2p8->1s4' ]
        
        lines_dens = [ l for l in all_lines if f"{l['from']}->{l['to']}" in lines ]
        lines_double_dens = [ l for l in all_lines if f"{l['from']}->{l['to']}" in lines_double ]

        if( plot ):
            splt.init_bytot( len( lines_dens ) + len( lines_double_dens ) / 2 )

        densities = {}

        for l in lines_dens:
            if(plot):
                splt.next()

            d = self.estimate_amplitude( wl_realigned - l['wl'], int_calibrated, error = True, plot = plot ) / l['A']
            if( d.n > 0 ):
                if( l['from'] not in densities.keys() ):
                    densities[ l['from'] ] = { 'all': [] }
                densities[ l['from'] ]['all'].append( d )

        for i in range( 0, len( lines_double_dens ), 2 ):
            if(plot):
                splt.next()

            l1 = lines_double_dens[i]
            l2 = lines_double_dens[i+1]

            a1, a2 = self.estimate_amplitude_double( wl_realigned - l1['wl'],int_calibrated,  l2['wl'] - l1['wl'], error = True, plot = plot )
            
            for a, l in zip( [ a1, a2 ], [ l1, l2 ] ):
                d = a / l['A']
                if( d.n > 0 ):
                    if( l['from'] not in densities.keys() ):
                        densities[ l['from'] ] = { 'all': [] }
                    densities[ l['from'] ]['all'].append( d )

        for lv in densities.keys():
            if( len( densities[lv]['all'] ) > 0 ):
                densities[lv]['avg'] = mean( densities[lv]['all'] )
            else:
                densities[lv]['avg'] = densities['all'][0]
        
        return densities