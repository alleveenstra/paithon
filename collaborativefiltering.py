# -*- coding: utf-8 -*-
'''
Collaborative filtering

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib

Nu = 100
Nm = 10
f = 4
lmb = 0.4

R = numpy.random.uniform(1, 5, (Nu, Nm)).astype(numpy.float32)
R *= numpy.random.uniform(0, 1.1, (Nu, Nm)).astype(numpy.int8)
R = numpy.round(R)

U = numpy.matrix(numpy.random.normal(0, 0.3, (Nu, f)).astype(numpy.float32))
M = numpy.matrix(numpy.random.normal(0, 0.3, (Nm, f)).astype(numpy.float32))

def update_U():
    for u in range(U.shape[0]):
        eyeI = numpy.eye(f) * lmb 
        movies = R[u, :] != 0
        Mu = M[movies, :]
        if (Mu.size > 0):
            vector = numpy.dot(R[u, movies], Mu)
            matrix = vector.transpose() * vector + numpy.multiply(U[u], eyeI)
            U[u, :] = numpy.linalg.solve(matrix, vector.tolist()[0])

def update_M():
    for m in range(M.shape[0]):
        eyeI = numpy.eye(f) * lmb
        users = R[:, m] != 0
        Um = U[users, :]
        if (Um.size > 0):
            vector = numpy.dot(R[users, m], Um)
            matrix = vector.transpose() * vector + numpy.multiply(M[m], eyeI)
            solution = numpy.linalg.lstsq(matrix, vector.tolist()[0])
            M[m, :] = solution[0]

for epoch in range(1000):
    update_M()
    update_U() 
        
print 'done'
