# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:41:39 2011

@author: -
"""

import numpy
import math
import matplotlib.pyplot as plt

class BackProp:
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput + 1
        self.nHidden = nHidden
        self.nOutput = nOutput
        
        self.activationInput = [1.0] * self.nInput
        self.activationHidden = [1.0] * self.nHidden
        self.activationOutput = [1.0] * self.nOutput

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.3, (self.nInput, self.nHidden)))
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.3, (self.nHidden, self.nOutput)))
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def derivedActivationFunction(self, value):
        return 1.0 - value ** 2
        
    def update(self, inputs):
        if len(inputs) != self.nInput - 1:
            raise ValueError('Input vector not long enough')
        
        for k in range(len(inputs)):
            self.activationInput[k] = inputs[k]
            
        for j in range(self.nHidden):
            activationSum = 0.0
            for k in range(self.nInput):
                activationSum += self.activationInput[k] * self.hiddenWeight[k,j]
            self.activationHidden[j] = self.activationFunction(activationSum)
        
        for i in range(self.nOutput):
            activationSum = 0.0
            for j in range(self.nHidden):
                activationSum += self.activationHidden[j] * self.outputWeight[j,i]
            self.activationOutput[i] = self.activationFunction(activationSum)
        
        return self.activationOutput
            
    def learn(self, target, alpha):
        if len(target) != self.nOutput:
            raise ValueError('Target vector nog long enough')
            
        deltaOutput = [0.0] * self.nOutput
        for i in range(self.nOutput):
            error = target[i] - self.activationOutput[i]
            deltaOutput[i] = self.derivedActivationFunction(self.activationOutput[i]) * error
        
        deltaHidden = [0.0] * self.nHidden
        for j in range(self.nHidden):
            error = 0.0
            for i in range(self.nOutput):
                error += deltaOutput[i] * self.outputWeight[j,i]
            deltaHidden[j] = self.derivedActivationFunction(self.activationHidden[j]) * error
        
        for i in range(self.nOutput):
            for j in range(self.nHidden):
                self.outputWeight[j,i] += deltaOutput[i] * self.activationHidden[j] * alpha
        
        for k in range(self.nInput):
            for j in range(self.nHidden):
                self.hiddenWeight[k,j] += deltaHidden[j]  * self.activationInput[k] * alpha
        
        error = 0.0
        for i in range(self.nOutput):
            error += 0.5 * (self.activationOutput[i] - target[i]) ** 2
        return error

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
    bp = BackProp(2,2,1)
    examples = numpy.matrix([[0,0],[1,0],[0,1],[1,1]])
    classes = [ 0, 1, 1, 0 ]
    errors = bp.train(examples, classes, 400, 0.3)
    x = numpy.divide(range(-100,100), 100.0)
    y = [0.0] * len(x)
    for i in range(len(y)):
       y[i] = bp.activationFunction(x[i])
    plt.plot(x,y)
    plt.show()
    
    plt.plot(range(len(errors)), errors)
    plt.show()
    
testBackProp()