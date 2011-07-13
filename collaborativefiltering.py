# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib
import loader
import pickle

m = 943
n = 1682
p = 5
lmba = 0.003
epochs = 30

print 'Loading movies...'

X = loader.loadMovies()

print 'Processing data...'

A = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, m))).astype(numpy.float32)
B = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, n))).astype(numpy.float32)

def update_A(X, B):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Anew = numpy.matrix(numpy.zeros((p, m))).astype(numpy.float32)
    for i in range(m):
        Xlocal = numpy.array(X[i, :].todense())[0]
        items = Xlocal > 0
        Bi = B[:, items]
        vector = Bi * numpy.matrix(Xlocal[items]).T
        matrix = Bi * Bi.T + numpy.multiply(lamI, A[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Anew[:, i] = solution[0]
    return Anew

def update_B(X, A):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Bnew = numpy.matrix(numpy.zeros((p, n))).astype(numpy.float32)
    for i in range(n):
        Xlocal = numpy.array(X[:, i].todense())[0]
        items = Xlocal > 0
        Ai = A[:, items]
        vector = Ai * numpy.matrix(Xlocal[items]).T
        matrix = Ai * Ai.T + numpy.multiply(lamI, B[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Bnew[:, i] = solution[0]
    return Bnew

for epoch in range(epochs):
    A = update_A(X, B)
    B = update_B(X, A)
    E = numpy.abs(X - numpy.dot(A.transpose(), B))
    print "Epoch %d..." % epoch

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
