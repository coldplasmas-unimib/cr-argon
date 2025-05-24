import numpy as np
import pandas as pd
from os.path import dirname
from .SingletonMeta import SingletonMeta

class Levels( metaclass = SingletonMeta ):

    def __init__(self):
        print("Initializing class Levels")
        
        levels = pd.read_excel(
            dirname( __file__ )  + "/data/Levels.xlsx", skiprows=4, index_col=None).to_dict('records')

        # Sort levels by energy
        levels = np.array(levels)[np.argsort(
            [l['Energy_ev'] for l in levels])]
        
        # Store in a dictionary indexed by Paschen
        self.levdict = {l['Paschen']: {'id': i, **l}
                        for (i, l) in enumerate(levels)}
        
        # Easy access to total count
        self.levcount = len(self.levdict)

        self.grouped_levels = [ "3d+2s", "3p" ]

    def exists( self, lv_name ):
        return lv_name in self.levdict.keys()

    def ID( self, lv_name ):
        return self.levdict[lv_name]['id']

    def __getitem__( self, lv_name ):
        return self.levdict[lv_name]

    def all_names(self):
        return list(self.levdict.keys())
    
    def namesToIdxs(self, lv_names):
        return [ self.ID(l) for l in lv_names]