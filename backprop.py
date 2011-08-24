# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:41:39 2011

@author: alle.veenstra@gmail.com
"""

import numpy
import math

class MultiLayerPerceptron:
    def __init__(self, nInput, nHidden, nOutput, eta = 0.08, eta_bias = 0.04, eta_L1 = 0.005):
        
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        
        self.inputActivation = numpy.matrix(0 - numpy.ones(self.nInput)).astype(numpy.float32)
        self.hiddenActivation = numpy.matrix(numpy.ones(self.nHidden)).astype(numpy.float32)
        self.outputActivation = numpy.matrix(numpy.ones(self.nOutput)).astype(numpy.float32)

        self.hiddenBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nHidden)).astype(numpy.float32)
        self.outputBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nOutput)).astype(numpy.float32)

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nInput, self.nHidden))).astype(numpy.float32)
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nHidden, self.nOutput))).astype(numpy.float32)
        
        self.deltaOutput = False
        self.deltaHidden = False

        self.activation = numpy.vectorize(self.activationFunction)
        self.derivative = numpy.vectorize(self.derivedActivationFunction)

        self.eta = eta
        self.eta_bias = eta_bias
        self.eta_L1 = eta_L1
        
        self.noiser = False
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def derivedActivationFunction(self, value):
        return 1.0 - math.tanh(value) ** 2
        
    def evaluateNetwork(self, inputs):
        if len(inputs) != self.nInput:
            raise ValueError('Input vector not long enough')
        
        for k in range(len(inputs)):
            self.inputActivation[0, k] = inputs[k]
        
        self.inHidden = self.inputActivation * self.hiddenWeight + self.hiddenBias
        self.hiddenActivation = self.activation(self.inHidden)
        
        self.inOutput = self.hiddenActivation * self.outputWeight + self.outputBias
        self.outputActivation = self.activation(self.inOutput)
        
        return self.outputActivation
        
    def runHidden(self, input):
        self.outputActivation = self.activation(input * self.outputWeight)
        return self.outputActivation
            
    def learn(self, inputs, target):
        if len(target) != self.nOutput:
            raise ValueError('Target vector of wrong size')
        
        self.evaluateNetwork(inputs)
        
        self.deltaOutput = numpy.multiply(self.derivative(self.inOutput), (target - self.outputActivation))
        self.deltaHidden = numpy.multiply(self.derivative(self.inHidden), (self.deltaOutput * self.outputWeight.T))

        self.outputBias += self.deltaOutput * self.eta_bias
        self.hiddenBias += self.deltaHidden * self.eta_bias
        
        L1 = numpy.sign(self.hiddenActivation) * -self.eta_L1
        
        self.hiddenWeight += (self.inputActivation.T * self.deltaHidden) * self.eta               
        self.outputWeight += (self.hiddenActivation.T * self.deltaOutput) * self.eta
        
        return numpy.sum(0.5 * numpy.power(self.outputActivation - target, 2))

    def train(self, example, classes, epochs):
        errors = numpy.zeros(epochs) - 1.0
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                input = example[cls, :].tolist()[0]
                target = classes[cls].tolist()[0]
                if self.noiser:
                    input = self.noiser.addNoise(input)
                error += self.learn(input, target)
            errors[epoch] = math.sqrt(error / len(classes))
            if epoch > 500:
                self.eta *= 0.95
        return errors[errors > 0]
            
    def verifyGradient(self, input, target):
        epsilon = 0.0001
        self.learn(input, target)
        savedHiddenWeight = numpy.copy(self.hiddenWeight)
        savedOutputWeight = numpy.copy(self.outputWeight)
        for i in range(self.nInput):
            for j in range(self.nHidden):
                positive = numpy.copy(self.hiddenWeight)
                negative = numpy.copy(self.hiddenWeight)
                   
                positive[i, j] += epsilon
                negative[i, j] -= epsilon
                
                self.hiddenWeight = positive
                output = self.evaluateNetwork(input)
                errorP1 = numpy.sqrt(numpy.sum((output - target) ** 2))
                
                self.hiddenWeight = negative
                output = self.evaluateNetwork(input)
                errorP2 = numpy.sqrt(numpy.sum((output - target) ** 2))
                
                approx = (errorP1 - errorP2) / (epsilon * 2)
                gradient = -(self.inputActivation[0, i] * self.deltaHidden[0, j])
                
                print 'hidden ', approx, gradient, gradient - approx
                
                self.hiddenWeight = savedHiddenWeight
        
        for i in range(self.nHidden):
            for j in range(self.nOutput):
                positive = numpy.copy(self.outputWeight)
                negative = numpy.copy(self.outputWeight)
                   
                positive[i, j] += epsilon
                negative[i, j] -= epsilon
                
                self.outputWeight = positive
                output = self.evaluateNetwork(input)
                errorP1 = 0.5 * (numpy.sum((output - target) ** 2))
                
                self.outputWeight = negative
                output = self.evaluateNetwork(input)
                errorP2 = 0.5 * (numpy.sum((output - target) ** 2))
                
                approx = (errorP1 - errorP2) / (epsilon * 2)
                gradient = -(self.hiddenActivation[0, i] * self.deltaOutput[0, j])
                
                print 'output ', approx, gradient, gradient - approx 
                
                self.outputWeight = savedOutputWeight

class SaltPepperNoiser(object):
    
    def __init__(self, amount = 0.1, salt = 0.5, pepper = -0.5):
        self.amount = amount
        self.salt = salt
        self.pepper = pepper
        
    def addNoise(self, input):
        output = numpy.array(input)
        half = len(output) * self.amount / 2.0
        salt = numpy.random.randint(0, len(output) - 1, half)
        pepper = numpy.random.randint(0, len(output) - 1, half)
        output[salt] = self.salt
        output[pepper] = self.pepper
        return output
        
bp = MultiLayerPerceptron(2, 4, 1, 0.1, 0, 0)
examples = numpy.matrix(numpy.random.uniform(-1, 1, (4, 2))).astype(numpy.float32)
classes = numpy.matrix(numpy.random.uniform(-1, 1, (4, 1))).astype(numpy.float32)
errors = bp.train(examples, classes, 1000)
bp.verifyGradient(examples[0, :].tolist()[0], classes[0, :].tolist()[0])
print errors[-1]
