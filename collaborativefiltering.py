# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib

m = 4
n = 3
p = 2

X = numpy.random.uniform(1, 5, (m, n)).astype(numpy.float32)
for i in range(m):
    X[i, :] = i + 1
    
A = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, m))).astype(numpy.float32)
B = numpy.matrix(numpy.random.normal(0.5, 0.1, (p, n))).astype(numpy.float32)

def update_A(X, B):
    A = X * B.I
    A[A < 0] = 0
    return A.T

def update_B(X, A):
    lamI = numpy.matrix(numpy.eye(p)).astype(numpy.float32)
    Bnew = numpy.matrix(numpy.zeros((p, n))).astype(numpy.float32)
    for i in range(n):
        users = X[:, i] > 0
        Ai = A[:, users]
        vector = Ai * numpy.matrix(X[users, i]).T
        matrix = Ai * Ai.T + numpy.multiply(lamI, B[:, i])
        solution = numpy.linalg.lstsq(matrix, vector)
        Bnew[:, i] = solution[0]
    return Bnew
        
           
for epoch in range(1000):
    A = update_A(X, B)
    B = update_B(X, A)
    E = numpy.abs(X - numpy.dot(A.transpose(), B))
    print numpy.sum(E)
        
print 'done'

