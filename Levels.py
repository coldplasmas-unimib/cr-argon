import numpy as np
import pandas as pd
from os.path import dirname
import re 

class Levels:

    def __init__(self):
        self.levels = pd.read_excel(
            dirname( __file__ )  + "/data/Levels.xlsx", skiprows=4, index_col=None).to_dict('records')

        # Sort levels by energy
        self.levels = np.array(self.levels)[np.argsort(
            [l['Energy_ev'] for l in self.levels])]
        self.levdict = {l['Paschen']: {'id': i, **l}
                        for (i, l) in enumerate(self.levels)}
        self.levcount = len(self.levdict)

    def exists( self, lv_name ):
        return lv_name in self.levdict.keys()

    def ID( self, lv_name ):
        return self.levdict[lv_name]['id']

    def __getitem__( self, lv_name ):
        return self.levdict[lv_name]

    def all_names(self):
        return list(self.levdict.keys())