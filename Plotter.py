import numpy as np
from .utilities import adv_plt as plt, splt
from . import TransMatrix, Solver

class Plotter:

    def __init__( self, solver: Solver.Solver ):
        self.solver = solver
        self.lv = solver.lv

        # Prepare subarrays for internal and external exchanges
        self.internal_exchanges = TransMatrix.TransMatrix( 0, self.lv )
        self.external_exchanges = TransMatrix.TransMatrix( 0, self.lv )
        self.gs_exchanges = TransMatrix.TransMatrix( 0, self.lv )
        for x in self.lv.all_names():
            for y in self.lv.all_names():
                if( x == 'gs' or y == 'gs' ):
                    self.gs_exchanges[x, y] = 1
                elif (x[:2] == y[:2]):
                    self.internal_exchanges[x, y] = 1
                else:
                    self.external_exchanges[x, y] = 1

        self.plotStuff = [
            ( 'tab:blue'	,	"Electron impact (internal)",  lambda slv: ( slv.Q_e * self.internal_exchanges * slv.f_e     ).M ),
            ( 'tab:cyan'	,	"Electron impact (external)",  lambda slv: ( slv.Q_e * self.external_exchanges * slv.f_e     ).M ),
            ( 'tab:purple'	,	"Electron impact (ground state)",      lambda slv: ( slv.Q_e * self.gs_exchanges * slv.f_e           ).M ),
            ( 'tab:green'	,	"Atom impact (internal)",      lambda slv: ( slv.Q_a * self.internal_exchanges               ).M ),
            ( 'tab:olive'	,	"Atom impact (external)",      lambda slv: ( slv.Q_a * self.external_exchanges               ).M ),
            ( 'tab:grey'	,	"Atom impact (ground state)",          lambda slv: ( slv.Q_a * self.gs_exchanges                     ).M ),
            ( 'tab:orange'	,	"Radiative emission",           lambda slv: ( slv.A / slv.n_g                                 ).M ),
        ]

        self.split_in = [
            ( 1, 5 ),
            ( 5, 15 )
        ]

    def four_plots(self, n, title):

        self.densities_plots( n, title )
        self.transitions_plots( n, title )

    def densities_plots(self, n, title):

        for start, end in self.split_in:
            splt.next()
            plt.bar( np.arange( end - start ), n[start:end] )
            plt.ylabel('n')
            plt.xticks( np.arange( end - start), self.lv.all_names()[start:end], rotation=45 )
            if( start == 1 ):
                plt.title( title )
                plt.yscale('log')

    def transitions_plots(self, n, title):
        for start, end in self.split_in:
            splt.next()
            stack = False
            for ( col, lab, func ) in self.plotStuff:
                splt.bar( np.arange( end - start), ( -func( self.solver ).dot( np.ones( self.lv.levcount ) ) * n)[start:end], stack=stack, color=col )
                stack = True

            stack = False
            for ( col, lab, func ) in self.plotStuff:
                splt.bar( np.arange( end - start), func( self.solver ).T.dot( n )[start:end], stack=stack, label=lab, color=col)
                stack = True

            xlim = plt.xlim()
            plt.plot( xlim, [0,0],'-k')
            plt.xlim(xlim)
            plt.ylim( np.array([ -1, 1 ]) * np.max( np.abs( plt.ylim())))
            plt.xticks( np.arange( end - start), self.lv.all_names()[start:end], rotation=45 )
            plt.ylabel("Transition rate [a.u.]\nDepopulation | Population")

    def prepare_legend(self, andPlot = True):
        for ( col, lab, func ) in self.plotStuff:
            splt.bar( [0],[1 if andPlot else 0], label=lab, color=col)
        splt.bar( [0],[1 if andPlot else 0], label=None, color='white')
        if( andPlot ):
            plt.legend(loc='center')

    def states_plot( self, n, tresh = -1 ):
        splt.init( 1, 1, size = (18, 10) )
        energies = { s: self.lv[s]['Energy_ev'] for s in self.lv.all_names() }
        energies['gs'] = 10.5

        for s in self.lv.all_names():
            plt.plot( [-0.5, 0.5], np.array([1,1]) * energies[s], lw = 1, c='k' )

        tp = [ TransMatrix.TransMatrix( ( f( self.solver ).T * n ).T, self.lv ) for _,_,f in self.plotStuff ]
        norm_on = np.max( [ np.max( t.M ) for t in tp ] )
        if( tresh < 0 ):
            tresh = 10**( np.log10( norm_on ) - 1.5 )
            print(f"Auto tresh {tresh:.1e}")
        mult_by = 10**( np.floor( np.log10( tresh ) ) )

        cont = 0.7
        step = 0.1
        # Rising
        for st in self.lv.all_names():
            for ed in self.lv.all_names():
                if( self.lv[ ed ]['Energy_ev'] < self.lv[ st ]['Energy_ev'] ):
                    continue
                for ( col, lab, _ ), M in zip( self.plotStuff, tp ):
                    if( M[st,ed] > tresh ):
                        plt.arrow( cont, energies[ st ], 0, energies[ ed ] - energies[ st ], head_width = 0.05, color = col, length_includes_head = True, alpha = ( M[st,ed] / norm_on / 2 ) + 0.5, width = ( M[st,ed] / norm_on / 40 ) )
                        plt.text( cont, energies[ ed ] + 0.05, f"{M[st,ed]/mult_by:.0f}", ha='center', va='bottom', rotation = 90 )
                        cont += step

        cont = -0.7
        step = -step
        # Falling
        for st in self.lv.all_names():
            for ed in self.lv.all_names():
                if( self.lv[ ed ]['Energy_ev'] > self.lv[ st ]['Energy_ev'] ):
                    continue
                for ( col, lab, _ ), M in zip( self.plotStuff, tp ):
                    if( M[st,ed] > tresh ):
                        plt.arrow( cont, energies[ st ], 0, energies[ ed ] - energies[ st ], head_width = 0.05, color = col, length_includes_head = True, alpha = ( M[st,ed] / norm_on / 2 ) + 0.5, width = ( M[st,ed] / norm_on / 40 ) )
                        plt.text( cont, energies[ st ] + 0.05, f"{M[st,ed]/mult_by:.0f}", ha='center', va='bottom', rotation = 90 )
                        cont += step

        ticks = []
        for s in ['gs', '1s', '2p', '3d+2s', '3p']:
            engs_filtered = { l: e for l, e in energies.items() if l.startswith( s ) }
            st = max( engs_filtered, key=engs_filtered.get)
            ticks.append( engs_filtered[st] )
            plt.text( 0, ticks[-1] + 0.03, st, ha='center' )
            if( len( engs_filtered ) > 1 ):
                ed = min( engs_filtered, key=engs_filtered.get)
                plt.text( 0, engs_filtered[ed] - 0.03, ed, ha='center', va='top' )


        cropped_ticks = [ f"{t:.2f}" for t in ticks ]
        cropped_ticks[0] = 0
        plt.xticks([])
        plt.yticks(ticks, cropped_ticks)

        plt.plot( [], [], color='white', label = fr"Rates ($10^{{{np.log10(mult_by):.0f}}}s^{{-1}}$)")
        for ( col, lab, _ ), M in zip( self.plotStuff, tp ):
            plt.plot( [], [], color = col, label = lab )
            plt.legend(loc='lower right')


