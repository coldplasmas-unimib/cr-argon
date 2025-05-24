import numpy as np
import pandas as pd
from os.path import dirname
from . import TransMatrix, Levels
from .SingletonMeta import SingletonMeta


class RadiativeTransitions(metaclass=SingletonMeta):
    def __init__(self):
        data = pd.read_excel(
            dirname(__file__) + "/data/NistCoefficients.xlsx", skiprows=4, index_col=None)
        self.lv = Levels.Levels()

        self.__A = TransMatrix.TransMatrix(0)

        for _, d in data.iterrows():
            self.__A[d['Start'], d['End']] = d['A']

        self.evToNm = 1239.8

    def A(self):
        return self.__A.copy()