# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib

m = 40
n = 30
p = 5
lmba = 0.003

X = numpy.random.uniform(1, 5, (m, n)).astype(numpy.float32)
X = X * numpy.random.uniform(0, 1.5, (m, n)).astype(numpy.int32)

X[1, 1] = 1
X[2, 2] = 2
X[3, 3] = 3
X[4, 4] = 4
X[5, 5] = 5

A = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, m))).astype(numpy.float32)
B = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, n))).astype(numpy.float32)

def update_A(X, B):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Anew = numpy.matrix(numpy.zeros((p, m))).astype(numpy.float32)
    for i in range(m):
        items = X[i, :] > 0
        Bi = B[:, items]
        vector = Bi * numpy.matrix(X[i, items]).T
        matrix = Bi * Bi.T + numpy.multiply(lamI, A[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Anew[:, i] = solution[0]
    return Anew

def update_B(X, A):
    lamI = numpy.matrix(numpy.eye(p) * lmba).astype(numpy.float32)
    Bnew = numpy.matrix(numpy.zeros((p, n))).astype(numpy.float32)
    for i in range(n):
        items = X[:, i] > 0
        Ai = A[:, items]
        vector = Ai * numpy.matrix(X[items, i]).T
        matrix = Ai * Ai.T + numpy.multiply(lamI, B[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Bnew[:, i] = solution[0]
    return Bnew
        
           
for epoch in range(10):
    A = update_A(X, B)
    B = update_B(X, A)
    E = numpy.abs(X - numpy.dot(A.transpose(), B))
    #print numpy.sum(E[X > 0])
        
print 'done'

print numpy.dot(A[:, 1].T, B[:, 1])
print numpy.dot(A[:, 2].T, B[:, 2])
print numpy.dot(A[:, 3].T, B[:, 3])
print numpy.dot(A[:, 4].T, B[:, 4])
print numpy.dot(A[:, 5].T, B[:, 5])
