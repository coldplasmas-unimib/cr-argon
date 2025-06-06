import numpy as np
import pandas as pd
from os.path import dirname
from .SingletonMeta import SingletonMeta
from typeguard import typechecked


class Levels(metaclass=SingletonMeta):

    def __init__(self):
        print("Initializing class Levels")

        levels = pd.read_excel(
            dirname(__file__) + "/data/Levels.xlsx", skiprows=4, index_col=None).to_dict('records')

        # Sort levels by energy
        self.levels = np.array(levels)[np.argsort(
            [l['Energy_ev'] for l in levels])]

        # Store in a dictionary indexed by Paschen
        self.levdict = { l['Paschen']: i
                        for (i, l) in enumerate(levels)}

        # Easy access to total count
        self.levcount = len(self.levdict)

        self.grouped_levels = ["3d+2s", "3p"]

    # @typechecked
    def exists(self, lv_name: str) -> bool:
        return lv_name in self.levdict.keys()

    # @typechecked
    def ID(self, lv_name: str) -> int:
        return self.levdict[lv_name]

    # @typechecked
    def __getitem__(self, lv_id: int) -> dict:
        return self.levels[lv_id]

    # @typechecked
    def getFromName(self, lv_name: str) -> dict:
        return self.levels[ self.levdict[lv_name] ]

    # @typechecked
    def all_names(self) -> list[str]:
        return list(self.levdict.keys())

    # @typechecked
    def namesToIdxs(self, lv_names: list[str]) -> list[int]:
        return [self.ID(l) for l in lv_names]
