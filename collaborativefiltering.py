# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib

Nu = 100
Nm = 10
f = 4
lmb = 0.5

R = numpy.random.uniform(1, 5, (Nu, Nm)).astype(numpy.float32)
R *= numpy.random.uniform(0, 1.1, (Nu, Nm)).astype(numpy.int8)
R = numpy.round(R) / 5.

R[1, 1] = 1.0
R[2, 2] = 0.8
R[3, 3] = 0.6
R[4, 4] = 0.4
R[5, 5] = 0.2

U = numpy.matrix(numpy.random.normal(0, 0.3, (Nu, f)).astype(numpy.float32))
M = numpy.matrix(numpy.random.normal(0, 0.3, (Nm, f)).astype(numpy.float32))

def update_U():
    for u in range(U.shape[0]):
        eyeI = numpy.eye(f) * lmb 
        movies = R[u, :] != 0
        Mu = M[movies, :]
        if (Mu.size > 0):
            vector = numpy.dot(R[u, movies], Mu).transpose()
            matrix = vector * vector.transpose() + numpy.multiply(U[u], eyeI)
            solution = numpy.linalg.lstsq(matrix, vector)
            sol = solution[0].transpose()
            #sol[sol < 0] = 0
            U[u, :] = sol

def update_M():
    for m in range(M.shape[0]):
        eyeI = numpy.eye(f) * lmb
        users = R[:, m] != 0
        Um = U[users, :]
        if (Um.size > 0):
            vector = numpy.matrix(numpy.dot(R[users, m], Um)).transpose()
            matrix = vector * vector.transpose() + numpy.multiply(M[m], eyeI)
            solution = numpy.linalg.lstsq(matrix, vector)
            sol = solution[0].transpose()
            #sol[sol < 0] = 0
            M[m, :] = sol
            
for epoch in range(1):
    update_M()
    update_U() 
        
print 'done'

print numpy.dot(U[1], M[1].transpose()) * 5
print numpy.dot(U[2], M[2].transpose()) * 5
print numpy.dot(U[3], M[3].transpose()) * 5
print numpy.dot(U[4], M[4].transpose()) * 5
print numpy.dot(U[5], M[5].transpose()) * 5

