import numpy as np
from .utilities import adv_plt as plt, splt
from . import Levels, RadiativeTransitions
from .utilities.ufloat_functs import mean, n
from scipy.optimize import curve_fit


class LinesFitter:

    def __init__(self):
        pass
    #     self.lv = Levels.Levels()
    #     self.tr = NistEinsteinData.NistEinsteinData(self.lv)

    #     self.unct = 0.7
    #     self.shift = 0.6

    #     # Select only certain lines
    #     self.updateLines([
    #         # 2p2
    #         697, 727, 827,
    #         # 2p3
    #         739, 841,
    #         # 2p4
    #         795, 852,
    #         # 2p6
    #         764,
    #         # 923,
    #         # 2p7
    #         # 867,
    #         # 2p8
    #         843,
    #         # 802,
    #         # 2p9
    #         812,
    #         # 2p10
    #         913,
    #         966
    #     ])

    # def updateLines(self, selected_wls):
    #     self.lines = self.tr.all_lines()

    #     self.lines = self.lines[[
    #         np.round(l['wl']) in selected_wls for l in self.lines]]

    #     # Prepare levels dict
    #     levels = np.unique([l['from'] for l in self.lines])
    #     self.levels = levels[np.argsort(
    #         [self.lv[l]['Energy_ev'] for l in levels])]

    # @staticmethod
    # def gaus(x, a, x0, sigma):
    #     return a*np.exp(-(x-x0)**2/(2*sigma**2)) / np.sqrt(2*np.pi) / sigma

    # def fitLines(self, wl, intensities, plot=False):
    #     wl = wl + self.shift  # Shift
    #     ampl = []
    #     dens = []

    #     if (plot):
    #         splt.init_bylen(self.lines)

    #     for l in self.lines:
    #         idxs = (wl >= l['wl'] - self.unct) & (wl <= l['wl'] + self.unct)

    #         try:
    #             par, err = curve_fit(LinesFitter.gaus, wl[idxs], intensities[idxs], p0=[
    #                                  np.max(intensities[idxs]), l['wl'], self.unct / 8])
    #         except RuntimeError as e:
    #             par, err = [0,0,1], [[0]]

    #         ampl.append(ufloat(par[0], np.sqrt(err[0][0])))
    #         dens.append(ufloat(par[0], np.sqrt(err[0][0])) / l['A'])

    #         if (plot):
    #             splt.next()
    #             plt.plot(wl[idxs], intensities[idxs])
    #             xs = np.linspace(l['wl'] - self.unct, l['wl'] + self.unct, 100)
    #             plt.plot(xs, LinesFitter.gaus(xs, *par))
    #             plt.title(f"{l['wl']:.1f}")

    #     return np.array(ampl), np.array(dens)

    # def estimate_levels(self, wl, intensities):

    #     _, densities = self.fitLines(wl, intensities)

    #     output = []
    #     for lv in self.levels:
    #         idxs = [l['from'] == lv for l in self.lines]
    #         output.append(mean(densities[idxs]))

    #     # Normalize
    #     output = output / np.mean(n(output))

    #     # TO BE REMOVED
    #     for i in range(len(output)):
    #         output[i] = ufloat(output[i].n, 0.05)  # TO BE REMOVED
    #         # print(f"{lv} estimated over {sum(idxs)} lines")

    #     return output
