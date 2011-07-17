# -*- coding: utf-8 -*-
'''


@author alle.veenstra@gmail.com
'''

import math
import itertools
import numpy
import matplotlib
import csv
import string
from scipy.sparse import lil_matrix

class DummyLoader(object):
    
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
    def getM(self):
        return self.m
    
    def getN(self):
        return self.n
    
    def loadStatic(self):
        self.m = 4
        self.n = 6
        X = [[1, 3, 0, 5, 4, 0], [1, 3, 3, 3, 5, 0], [1, 0, 0, 0, 1, 0], [0, 4, 5, 3, 5, 5]]
        return lil_matrix(X)
    
    def loadFull(self):
        X = numpy.random.uniform(1, 5, (self.getM(), self.getN()))
        X = numpy.round(X)
        X = lil_matrix(X)
        return X
    
    def loadMovies(self):
        X = self.loadFull()
        for m in range(self.getM()):
            for n in range(self.getN()):
                if numpy.random.uniform(0, 10) < 4:
                    X[m, n] = 0
        return X
    
    def loadTitles(self):
        pass
