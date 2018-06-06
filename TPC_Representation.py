"""
File creating a representative TPC out of bins as defined in Binning.py
"""

from Binning import Bin
import numpy as np




class TpcRep(object):

    def __init__(self,geometry, scorer, num_evnt):
        self.radius = 0
        self.height = 0
        self.bins = {}
        self.tot_score = 0
        self.geometry = geometry
        self.num_of_bins = 8
        self.bin_scoring_function = scorer
        self.tpc_scoring_function = 'tpc_scorer'
        self.num_of_events = num_evnt

    def construct_bin_geometry(self):
        if self.geometry == 'loop':
            sequence = [0, 1, 2, 3, 4, 3, 2, 1]
            for i in range(0,self.num_of_bins):
                connectedness= sequence[-i:]+sequence[:-i]
                self.bins['BinNum'+str(i)]=Bin(self.scoring_function, connectedness,
                                               self.num_of_events/self.num_of_bins + 5)

    def fill_bins(self, Patterns):
        fill_num=self.num_of_events/self.num_of_bins
        for i in range(self.num_of_bins):
            self.bins['BinNum'+str(i)].add_patterns(Patterns[i*fill_num:(i+1)*fill_num])

    def score_bins(self):
        for i in range(self.num_of_bins):
            self.bins['BinNum'+str(i)].get_score()


    # def score_bins_across(self):
    #     for i in range



def ScoringFunction(list_of_patterns):
    a = []
    b = []
    Errs = (np.logical_xor(a, b) == True).sum()

