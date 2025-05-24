import numpy as np
import re
from os.path import dirname
from . import Levels, TransMatrix
from copy import deepcopy
import pandas as pd
from functools import lru_cache
from .SingletonMeta import SingletonMeta
from glob import glob



class ElectronsTransitionsData_Factory(metaclass = SingletonMeta):
# class ElectronsTransitionsData_Factory():  # TODO make this a singleton
    def __init__(self):
        self.lv = Levels.Levels()

        self.data = []
        # Integers transitions matrix, such that: 0 if non defined, else defined in self.data[i]
        self.ref = TransMatrix.TransMatrix(0)

        for f in glob(dirname(__file__) + "/data/parseddata_e/k_*.csv"):
            st, ed = re.search(r'k_([a-z\d\+]+)_([a-z\d\+]+)\.csv', f).groups()
            self.data.append(pd.read_csv(f, index_col=False).to_dict('list'))
            self.data[-1].update(
                {'from': st, 'to': ed}
            )

            if (len(self.data) > 1):
                # Assert all temperature scales are the same
                assert np.all(
                    np.array(self.data[-1]['T_e']) == np.array(self.data[-2]['T_e']))

            self.ref[st, ed] = len(self.data) - 1

        if( len( self.data) > 1):
            self.T_es = np.array( self.data[0]['T_e'] )

        print(f"Loaded {len(self.data)} files")

    def montecarloPick(self):
        return ElectronsTransitionsData(self)


class ElectronsTransitionsData:

    def __init__(self, factory: ElectronsTransitionsData_Factory):

        self.factory = factory
        self.ks = [  # Here the Montecarlo picking happens!
            f * np.array(d['k_max']) + (1.0 - f) * np.array(d['k_min']) for d, f in zip(factory.data, np.random.rand(len(factory.data)))
        ]
        self.rev_ks = [  # Here instead the reverse (detbal) coefficients are computed
            self._detbal(ks_props, ks_data) for (ks_props, ks_data) in zip(factory.data, self.ks)
        ]

        self._transMatrix = TransMatrix.TransMatrix(0)

    def transMatrix(self, T_e):
        # print(T_e,  self.factory.T_es[-1],  self.factory.T_es[0])
        assert T_e < self.factory.T_es[-1]
        assert T_e > self.factory.T_es[0]

        i_frac = np.interp(T_e, self.factory.T_es, np.arange(len(self.factory.T_es)))
        i = int(np.floor(i_frac))
        frac = i_frac - i

        for st in self._transMatrix.lv.all_names():
            for ed in self._transMatrix.lv.all_names():
                key = int(self.factory.ref[st, ed])
                if (key > 0):
                    self._transMatrix[st, ed] = self.ks[key][i] * \
                        (1 - frac) + self.ks[key][i + 1] * frac
                else:
                    revkey = int(self.factory.ref[ed, st])
                    if (revkey > 0):
                        self._transMatrix[st, ed] = self.rev_ks[revkey][i] * \
                            (1 - frac) + self.rev_ks[revkey][i + 1] * frac
                    else:
                        self._transMatrix[st, ed] = 0

        return self._transMatrix

    def _detbal(self, ks_props, ks_data ):
        new_from_lev = self.factory.lv[ ks_props['to'] ]
        new_to_lev = self.factory.lv[ ks_props['from'] ]
        expfact = (new_to_lev['Energy_ev'] - new_from_lev['Energy_ev']) / self.factory.T_es
        return new_to_lev['g'] / new_from_lev['g'] * ks_data * np.exp(- expfact)
