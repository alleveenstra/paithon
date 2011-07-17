# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import math
import itertools
import numpy
import matplotlib
import loader
import pickle
import dummyloader

loader = dummyloader.DummyLoader(4, 6)

p = 1
lmba = 0.001
l2factor = 0.001
epochs = 1000

print 'Loading movies...'

X = loader.loadStatic()
m = loader.getM()
n = loader.getN()

print 'Processing data...'

A = numpy.matrix(numpy.random.normal(0, 0.1, (p, m))).astype(numpy.float32)
B = numpy.matrix(numpy.random.normal(0, 0.1, (p, n))).astype(numpy.float32)

for i in range(X.shape[1]):
    Bdata = X[:, i].todense()
    B[0, i] = numpy.mean(Bdata[Bdata > 0])
    
def update_A(X, A, B):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Anew = numpy.matrix(numpy.zeros((p, m))).astype(numpy.float32)
    for i in range(m):
        Xlocal = numpy.array(X[i, :].todense())[0]
        items = Xlocal > 0
        Bi = B[:, items]
        vector = numpy.dot(Bi, numpy.matrix(Xlocal[items]).T)
        matrix = numpy.dot(Bi, Bi.T) + numpy.dot(lamI, A[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Anew[:, i] = solution[0]
    return Anew

def update_B(X, A, B):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Bnew = numpy.matrix(numpy.zeros((p, n))).astype(numpy.float32)
    for i in range(n):
        Xlocal = numpy.array(X[:, i].todense()).T[0]
        items = Xlocal > 0
        Ai = A[:, items]
        vector = numpy.dot(Ai, numpy.matrix(Xlocal[items]).T)
        matrix = numpy.dot(Ai, Ai.T) + numpy.dot(lamI, B[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Bnew[:, i] = solution[0]
    return Bnew

def RMSE(X, A, B):
    C = X.tocoo()
    count = 0
    square_sum = 0
    for i, j, v in itertools.izip(C.row, C.col, C.data):
        square_sum += (v - numpy.dot(A[:, i].T, B[:, j])) ** 2
        count += 1
    return math.sqrt(square_sum / count)

for epoch in range(epochs):
    A = update_A(X, A, B) 
    B = update_B(X, A, B)
    E = RMSE(X, A, B)
    print "Epoch %d: RMSE %.4f..." % (epoch, E)

print X.todense()
numpy.set_printoptions(precision = 2)
print numpy.dot(A.T, B)

print 'Saving to factorization.pkl...'

data = {'m': m,
        'n': n,
        'lmba': lmba,
        'A': A,
        'B': B}
output = open('factorization.pkl', 'wb')
pickle.dump(data, output)
output.close()

print 'Done!'
