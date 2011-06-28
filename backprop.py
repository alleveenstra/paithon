# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:41:39 2011

@author: alle.veenstra@gmail.com
"""

import numpy
import math
import matplotlib.pyplot as plt

class BackProp:
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput + 1
        self.nHidden = nHidden
        self.nOutput = nOutput
        
        self.activationInput = numpy.matrix(numpy.zeros(self.nInput))
        self.activationHidden = numpy.matrix(numpy.zeros(self.nHidden))
        self.activationOutput = numpy.matrix(numpy.zeros(self.nOutput))

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.3, (self.nInput, self.nHidden)))
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.3, (self.nHidden, self.nOutput)))
    
        self.activation = numpy.vectorize(self.activationFunction)
        self.gradient = numpy.vectorize(self.gradientFunction)
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def gradientFunction(self, value):
        return 1.0 - value ** 2
        
    def update(self, inputs):
        if len(inputs) != self.nInput - 1:
            raise ValueError('Input vector not long enough')
        
        for k in range(len(inputs)):
            self.activationInput[0, k] = inputs[k]
            
        self.activationHidden = self.activation(self.activationInput * self.hiddenWeight)
        
        self.activationOutput = self.activation(self.activationHidden * self.outputWeight)
        
        return self.activationOutput
            
    def learn(self, target, alpha):
        if len(target) != self.nOutput:
            raise ValueError('Target vector nog long enough')
            
        deltaOutput = numpy.multiply(self.gradient(self.activationOutput),
                                     (target - self.activationOutput))
        
        error = deltaOutput * self.outputWeight.transpose()
        deltaHidden = numpy.multiply(self.gradient(self.activationHidden), error)
        
        self.outputWeight = self.outputWeight + (self.activationHidden.transpose() * deltaOutput) * alpha
        self.hiddenWeight = self.hiddenWeight + (self.activationInput.transpose() * deltaHidden) * alpha       
        
        return numpy.sum(0.5 * numpy.power(self.activationOutput - target, 2))

    def train(self, example, classes, epochs, alpha):
        errors = [0.0] * epochs 
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                target = example[cls,:].tolist()[0]
                self.update(target)
                error += self.learn([classes[cls]], alpha)
            errors[epoch] = error
        return errors

def testBackProp():
    bp = BackProp(2,3,1)
    examples = numpy.matrix([[0,0],[1,0],[0,1],[1,1]])
    classes = [ 0, 1, 1, 0 ]
    errors = bp.train(examples, classes, 400, 0.3)
    
    print '[0,0] -> %.2f' % bp.update([0,0])[0,0]
    print '[1,0] -> %.2f' % bp.update([1,0])[0,0]
    print '[0,1] -> %.2f' % bp.update([0,1])[0,0]
    print '[1,1] -> %.2f' % bp.update([1,1])[0,0]

    plt.plot(range(len(errors)), errors)
    plt.show()
        

#===============================================================================
#     x = numpy.divide(range(-100,100), 100.0)
#     y = [0.0] * len(x)
#     for i in range(len(y)):
#        y[i] = bp.activationFunction(x[i])
#     plt.plot(x,y)
#     plt.show()
#===============================================================================
        
    
testBackProp()