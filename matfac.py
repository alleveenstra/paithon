# -*- coding: utf-8 -*-
'''
Collaborative filtering

Alternating least squares approach

@author alle.veenstra@gmail.com
'''

import numpy
import matplotlib

m = 20
n = 10
p = 2

X = numpy.random.uniform(1, 5, (m, n)).astype(numpy.float32)
for i in range(m):
    X[i, :] = i + 1
    
print X

A = numpy.matrix(numpy.random.normal(0.5, 0.1, (m, p)).astype(numpy.float32))
B = numpy.matrix(numpy.random.normal(0.5, 0.1, (n, p)).astype(numpy.float32))

def update_A(X, B):
    A = X * B.T.I
    A[A < 0] = 0
    return A

def update_B(R, A):
    B = (A.I * X).T
    B[B < 0] = 0
    return B
            
for epoch in range(100):
    A = update_A(X, B)
    B = update_B(X, A)
    E = numpy.abs(X - numpy.dot(A, B.transpose()))
    print numpy.sum(E)
        

        
print 'done'
