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
        
        self.inputActivation = numpy.matrix(numpy.ones(self.nInput))
        self.hiddenActivation = numpy.matrix(numpy.ones(self.nHidden))
        self.outputActivation = numpy.matrix(numpy.ones(self.nOutput))

        self.hiddenBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nHidden))
        self.outputBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nOutput))

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nInput, self.nHidden)))
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nHidden, self.nOutput)))
    
        self.activation = numpy.vectorize(self.activationFunction)
        self.gradient = numpy.vectorize(self.gradientFunction)
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def gradientFunction(self, value):
        return 1.0 - math.tanh(value) ** 2
        
    def update(self, inputs):
        if len(inputs) != self.nInput - 1:
            raise ValueError('Input vector not long enough')
        
        for k in range(len(inputs)):
            self.inputActivation[0, k] = inputs[k]
            
        self.hiddenActivation = self.activation(self.inputActivation * self.hiddenWeight)

        self.outputActivation = self.activation(self.hiddenActivation * self.outputWeight)

        return self.outputActivation
            
    def learn(self, target, alpha):
        if len(target) != self.nOutput:
            raise ValueError('Target vector nog long enough')
            
        error = (target - self.outputActivation)

        deltaOutput = numpy.multiply(self.gradient(self.outputActivation), error)

        error = deltaOutput * self.outputWeight.transpose()
        deltaHidden = numpy.multiply(self.gradient(self.hiddenActivation), error)
        
        self.outputWeight = self.outputWeight + (self.hiddenActivation.transpose() * deltaOutput) * alpha
        self.hiddenWeight = self.hiddenWeight + (self.inputActivation.transpose() * deltaHidden) * alpha       
        
        return numpy.sum(0.5 * numpy.power(self.outputActivation - target, 2))

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
    bp = BackProp(2, 4, 1)
    examples = numpy.matrix([[0,0], [1,0], [0,1], [1,1]])
    classes = [ 1, -1, -1, 1 ]
    errors = bp.train(examples, classes, 1000, 0.08)
    
    print '[0,0] -> %.2f' % bp.update([0,0])[0,0]
    print '[1,0] -> %.2f' % bp.update([1,0])[0,0]
    print '[0,1] -> %.2f' % bp.update([0,1])[0,0]
    print '[1,1] -> %.2f' % bp.update([1,1])[0,0]


#===============================================================================
    plt.plot(range(len(errors)), errors)
    plt.show()
#===============================================================================
#    x = numpy.divide(range(-200,200), 100.0)
#    plt.plot(x, bp.gradient(x))
#    plt.show()
#===============================================================================
        
    
testBackProp()
