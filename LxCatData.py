import numpy as np
import re
from os.path import dirname
from . import Levels, TransMatrix

class LxCatData:

    def __init__(self, lv: Levels.Levels):
        self.lv = lv
        
        self.transit = {}
        self.dbs = {
            'bsr': dirname( __file__ ) + "/data/BSR.txt",
            'IST': dirname( __file__ ) + "/data/IST.txt"
        }


        # To thicken the data
        self.common_E = np.arange(0.01, 300, 0.01)

        self.last_Te = -1
            
        m     = 9.109e-31 # kG
        eVtoJ = 1.60218e-19
        self.prefact = np.sqrt( 2.0 / m * eVtoJ )

        self.read_db( self.dbs['bsr'] )
        self.read_db( self.dbs['IST'] )


    def computeSigma( self, E_readed, sigma_readed, thresh ):
        sigma =  np.zeros(len(self.common_E))
        to_predict = (self.common_E > thresh ) * (self.common_E >
                                                E_readed[0]) * (self.common_E < E_readed[-1])
        sigma[to_predict] = np.interp(self.common_E[to_predict], E_readed, sigma_readed, left = 0, right = 0 )
        return sigma

    def read_db( self, db ):
        statesSets = {}
        loc_transit = {}
        with open( db ) as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if (line.startswith('EXCITATION')):
                    line = f.readline().split( '->' )
                    st = self.parseState( line[0], statesSets )
                    ed = self.parseState( line[1], statesSets )
                    thresh = 0
                    temp_E = []
                    temp_sigma = []

                    line = line[0]
                    while not line.startswith('---'):
                        if (line.startswith("PARAM")):
                            # (thresh' line)
                            match = re.match(
                                '.*E ?= ?([0-9\.]*) ?eV.*', line)
                            if (match):
                                thresh = float(match.group(1))
                            else:
                                print(line)
                        line = f.readline()
                    xy = f.readline()

                    while not xy.startswith('---'):
                        x, y = xy.split('\t')
                        temp_E.append(float(x))
                        temp_sigma.append(float(y))
                        xy = f.readline()
                    
                    if( st == ed ):
                        continue
                    if( st not in loc_transit.keys() ):
                        loc_transit[st] = {}
                    if( ed in loc_transit[st].keys() ):
                        # print(f"Duplicate transition! Summing cross sections {st} -> {ed}")
                        loc_transit[st][ed] += self.computeSigma( temp_E, temp_sigma, thresh )
                    else:
                        loc_transit[st][ed] = self.computeSigma( temp_E, temp_sigma, thresh )

        # Normalize on the number of substates!
        for st in loc_transit.keys():
            for ed in loc_transit[st].keys():
                norm_on = len( statesSets[st] )
                if( st == '3p' or ed == '3p'):
                    norm_on = norm_on * 2
                # print(f"Normalizing {st}->{ed} on {norm_on}")
                loc_transit[st][ed] = loc_transit[st][ed] / norm_on

        # Populate reverse reactions!
        for st in loc_transit.keys():
            for ed in loc_transit[st].keys():
                if( ed in loc_transit.keys() ):
                    if( st not in loc_transit[ed].keys() ):
                        deltaE = self.lv[ed]['Energy_ev'] - self.lv[st]['Energy_ev']
                        new_tempsigma = ( self.common_E + deltaE ) / self.common_E * np.interp( self.common_E + deltaE, self.common_E, loc_transit[st][ed], left = 0, right = 0 )
                        loc_transit[ed][st] = self.lv[st]['g'] / self.lv[ed]['g'] * new_tempsigma
                        # print("Reverse reaction populated")

        # Save in the common database!
        for st in loc_transit.keys():
            if( st not in self.transit ):
                self.transit[st] = {}
            for ed in loc_transit[st].keys():
                if( ed in self.transit[st].keys() ):
                    print(f"Overriding cross section! {st}->{ed} with {db}, fact of {self.k(loc_transit[st][ed],1)/self.k(self.transit[st][ed],1)}")
                self.transit[st][ed] = loc_transit[st][ed]

    def parseState(self, name, statesSets):
        st, st_complete = self.parseStateSub( name )

        if( st not in statesSets ):
            statesSets[st] = set()

        statesSets[st].add( st_complete )

        return st


    def parseStateSub(self, name):
        name = name.strip()
        if( name == 'Ar' ):
            return 'gs', 'Ar'

        content = re.match( '.*\((.*)\)', name ).group(1)

        statesdict = {
            "4s[3/2]2" : "1s5",
            "4s[3/2]1" : "1s4",
            "4s'[1/2]0" : "1s3",
            "4s'[1/2]1" : "1s2",
            "4p[1/2]1" : "2p10",
            "4p[5/2]3" : "2p9",
            "4p[5/2]2" : "2p8",
            "4p[3/2]1" : "2p7",
            "4p[3/2]2" : "2p6",
            "4p[1/2]0" : "2p5",
            "4p'[3/2]1" : "2p4",
            "4p'[3/2]2" : "2p3",
            "4p'[1/2]1" : "2p2",
            "4p'[1/2]0" : "2p1",
            "3d": "3d+2s",
            "5s": "3d+2s",
            "Ry": "3p"
        }

        # BSR excited
        match = re.match("(4[sp]'?\[[135]/2\][0-9]+)", content)
        if( match ):
            if( match.group(1) in statesdict.keys() ):
                return statesdict[match.group(1)], content
            
        # BSR excited but generalized
        if( content[:2] in statesdict.keys() ):
            # print( f"{content} summarized as {statesdict[content[:2]]}" )
            return statesdict[content[:2]], content
        
        # IST excited to skip
        match = re.match("(\dP\d)", content)
        if( match ):
            return 'hl'
        match = re.match("([456][sdp]'?)", content)
        if( match ):
            return 'hl'
        match = re.match("(3d'.*)", content)
        if( match ):
            return 'hl'
        
        print(f"Unmatched! {name}")

        return name, name

    def k(self, transition, Te):
        if (Te != self.last_Te):
            self.updateTe(Te)

        return np.trapz( transition * self.multFact, self.common_E ) / self.normFact * self.prefact

    def updateTe(self, Te):
        F_e = np.exp( - self.common_E / Te )
        self.multFact = self.common_E * F_e
        # self.normFact = np.sqrt( np.pi / 4 ) * np.power( Te, 1.5 )
        self.normFact = np.trapz( np.sqrt( self.common_E ) * F_e, self.common_E )

        self.last_Te = Te

    
    def detbal( self, rev_Q_e, new_from_lev, new_to_lev, Te ):
        expfact = ( new_to_lev['Energy_ev'] - new_from_lev['Energy_ev'] ) / Te
        Q_rev = new_to_lev['g'] / new_from_lev['g'] * rev_Q_e * np.exp( - expfact )
        return Q_rev

    def Q_e(self, Te, verbose = False):

        Q_e = TransMatrix.TransMatrix( -1, self.lv ) # Q_e[from, to]; trans is from, col is to

        cont = 0

        for st in self.transit.keys():
            for ed in self.transit[st].keys():
                if( not self.lv.exists( st ) or not self.lv.exists( ed ) ):
                    if( verbose ):
                        print("Transition between unknown levels: {st} -> {ed}; skipped")
                    continue
                Q_e[ st, ed ] = self.k( self.transit[st][ed], Te )

        for st in self.lv.all_names():
            for ed in self.lv.all_names():

                if( st == ed ):
                    Q_e[ st, ed ] = 0

                elif Q_e[ st, ed ] == -1:

                    # Missing coefficient!
                    # Try to compute from detailed balance
                    if( Q_e[ ed, st ] != -1 ):
                        Q_e[ st, ed ] = self.detbal( Q_e[ ed, st ], self.lv[ st ], self.lv[ ed ], Te )
                        # print(f"Missing cross section calculated via d
                        # et.bal. {st} -> {ed}")
                    else:
                        Q_e[ st, ed ] = 0
                        if( self.lv.ID(st) < self.lv.ID(ed) ):
                            if( verbose ):
                                print(f"Missing coefficient! {st} to {ed}; threated as 0")
                            cont += 1

        if( cont > 0 ):
            if( verbose ):
                    print(f"E. impact: missing {cont} coefficients!")

        # Rates
        return Q_e
