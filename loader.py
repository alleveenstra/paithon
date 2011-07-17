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

def getM():
    return 943

def getN():
    return 1682

def loadMovies():
    X = lil_matrix((943, 1682))
    filename = "ml-data/u.data"
    reader = csv.reader(open(filename, 'rb'), delimiter = '\t', quotechar = '|')
    for row in reader:
        user = string.atoi(row[0]) - 1
        movie = string.atoi(row[1]) - 1
        rating = string.atoi(row[2])
        X[user, movie] = rating
    return X

def loadTitles():
    X = {}
    filename = "ml-data/u.item"
    reader = csv.reader(open(filename, 'rb'), delimiter = '|', quotechar = '|')
    for row in reader:
        id = string.atoi(row[0]) - 1
        movie = row[1]
        X[id] = movie
    return X
