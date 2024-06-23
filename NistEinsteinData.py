import numpy as np
import pandas as pd
from os.path import dirname
from . import TransMatrix, Levels

class NistEinsteinData:

    def __init__(self, lv: Levels.Levels):
        data = pd.read_csv( dirname( __file__ ) + "/data/NIST_TRANS.txt")
        self.lv = lv

        self.A = TransMatrix.TransMatrix( 0, lv )
        self.statesSets = {}

        for i, d in data.iterrows():
            st = self.parseState( d, 'k' )
            ed = self.parseState( d, 'i' )

            if( st not in lv.all_names() or ed not in lv.all_names() ):
                continue
            if( st == ed ):
                continue
            if( self.A[st,ed] > 0 ):
                # print(f"Transition duplicated! Summing. {st} -> {ed}")
                pass
            if( np.isnan(  d['Aki(s^-1)'] ) ):
                continue
            self.A[st,ed] = self.A[st,ed] + d['Aki(s^-1)']

        for st in lv.all_names():
            for ed in lv.all_names():
                norm_on = len( self.statesSets[st] )
                # if( norm_on > 1 ):
                #     print(f"Normalizing {st}->{ed} on {norm_on}")
                self.A[st,ed] = self.A[st,ed] / norm_on

        self.evToNm = 1239.8

    def parseState(self, row, idx):
        st, st_complete = self.parseStateSub( row, idx )

        if( st not in self.statesSets ):
            self.statesSets[st] = set()

        self.statesSets[st].add( st_complete )

        return st

    def parseStateSub( self, row, idx ):
        
        keys = {
            f"{l['conf']}%{l['term']}%{l['J']}": l['Paschen'] for k,l in self.lv.levdict.items()
        }
        
        keys[ "3d" ] = "3d+2s"
        keys[ "5s" ] = "3d+2s"
        keys[ "5p" ] = "3p"

        conf = f"{row['conf_' + idx]}%{row['term_' + idx]}%{row['J_' + idx]:.0f}"
        if( conf in keys.keys() ):
            return keys[conf], conf
        
        ss = str(row['conf_' + idx])[-2:]
        if( ss in keys.keys() ):
            return keys[ss], conf
        
        return row['conf_' + idx], conf
        
    def all_lines(self):
        lines = []
        for i in self.lv.all_names():
            for j in self.lv.all_names():
                if( self.A[i,j] > 0 ):
                    lines.append( {
                        'from': i,
                        'to': j,
                        'wl': self.evToNm / ( self.lv[i]['Energy_ev'] - self.lv[j]['Energy_ev'] ),
                        'A': self.A[ i,j ]
                    } )

        lines = np.array( lines )
        lines = lines[ np.argsort( [ l['wl'] for l in lines ] ) ]
        return lines