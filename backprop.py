# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:41:39 2011

@author: alle.veenstra@gmail.com
"""

import numpy
import math
import pylab
import matplotlib.pyplot as plt

class SaltPepperNoiser(object):
    
    def __init__(self, amount = 0.3, salt = 0.5, pepper = -0.5):
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
    
class DefaultNetworkEvaluator(object):
    
    def __init__(self, perceptron):
        self.perceptron = perceptron
    
    def evaluateNetwork(self, inputs):
        perceptron = self.perceptron
        perceptron.hiddenActivation = perceptron.activation(perceptron.inputActivation * perceptron.hiddenWeight + perceptron.hiddenBias)
        perceptron.outputActivation = perceptron.activation(perceptron.hiddenActivation * perceptron.outputWeight + perceptron.outputBias)
        return perceptron.outputActivation

class MultiLayerPerceptron:
    def __init__(self, nInput, nHidden, nOutput, eta = 0.08, eta_bias = 0.04, eta_L1 = 0.005):
        
        self.nInput = nInput + 1
        self.nHidden = nHidden
        self.nOutput = nOutput
        
        self.inputActivation = numpy.matrix(0 - numpy.ones(self.nInput)).astype(numpy.float32)
        self.hiddenActivation = numpy.matrix(numpy.ones(self.nHidden)).astype(numpy.float32)
        self.outputActivation = numpy.matrix(numpy.ones(self.nOutput)).astype(numpy.float32)

        self.hiddenBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nHidden)).astype(numpy.float32)
        self.outputBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nOutput)).astype(numpy.float32)

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nInput, self.nHidden))).astype(numpy.float32)
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nHidden, self.nOutput))).astype(numpy.float32)

        self.activation = numpy.vectorize(self.activationFunction)
        self.gradient = numpy.vectorize(self.gradientFunction)

        self.eta = eta
        self.eta_bias = eta_bias
        self.eta_L1 = eta_L1
        
        self.noiser = False
        self.evaluationFunction = DefaultNetworkEvaluator(self)
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def gradientFunction(self, value):
        return 1.0 - math.tanh(value) ** 2
        
    def evaluateNetwork(self, inputs):
        if len(inputs) != self.nInput - 1:
            raise ValueError('Input vector not long enough')
        for k in range(len(inputs)):
            self.inputActivation[0, k] = inputs[k]
        return self.evaluationFunction.evaluateNetwork(inputs)
        
    def runHidden(self, input):
        self.outputActivation = self.activation(input * self.outputWeight + self.outputBias)
        return self.outputActivation
            
    def learn(self, target):
        if len(target) != self.nOutput:
            raise ValueError('Target vector nog long enough')
            
        error = (target - self.outputActivation)
        deltaOutput = numpy.multiply(self.gradient(self.outputActivation), error)

        error = (deltaOutput * self.outputWeight.transpose())
        deltaHidden = numpy.multiply(self.gradient(self.hiddenActivation), error)
        
        self.outputBias += deltaOutput * self.eta_bias
        self.hiddenBias += deltaHidden * self.eta_bias
        
        L1 = numpy.sign(self.hiddenActivation) * -self.eta_L1 
        self.hiddenWeight += (self.inputActivation.transpose() * (deltaHidden + L1)) * self.eta               
        self.outputWeight += (self.hiddenActivation.transpose() * deltaOutput) * self.eta
        
        return numpy.sum(0.5 * numpy.power(self.outputActivation - target, 2))

    def train(self, example, classes, epochs):
        errors = numpy.zeros(epochs) - 1.0
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                target = example[cls, :].tolist()[0]
                if self.noiser:
                    target = self.noiser.addNoise(target)
                self.evaluateNetwork(target)
                error += self.learn(classes[cls].tolist()[0])
            errors[epoch] = math.sqrt(error / len(classes))
            if self.stoppingCriteria(errors):
                break
        return errors[errors > 0]
        
    def stoppingCriteria(self, errors):
        stopping_count = 10
        errs = errors[errors > 0][::-1]
        if len(errs) > 2 * stopping_count:
            errs_new = errs[0 : stopping_count]
            errs_old = errs[stopping_count + 1: stopping_count * 2]
            if (numpy.mean(errs_new) - numpy.mean(errs_old)) / numpy.mean(errs_old) > 0.0:
                return True
        else:
            return False
        return False

def readImage(filename, dist_width = 0.3):
    image = numpy.reshape(pylab.imread(filename), 64 * 64)
    image = ((image - numpy.mean(image)) / numpy.std(image)) * dist_width
    return image
    
def showImages(before, after, n_images = 1, n_image = 1):
    before = numpy.matrix(numpy.reshape(before, (64, 64)))
    after = numpy.matrix(numpy.reshape(after, (64, 64)))    
    if n_image == 1:
        plt.figure(1)
    plt.subplot(n_images / 2, 4, 1 + (n_image - 1) * 2)
    plt.imshow(before, origin = 'lower')
    plt.gray()
    plt.subplot(n_images / 2, 4, 1 + (n_image - 1) * 2 + 1)
    plt.imshow(after, origin = 'lower')
    plt.gray()
    if n_images == n_image:
        plt.show()

def testImage():
    bp = MultiLayerPerceptron(64 * 64, 4, 64 * 64, 0.08)
    bp.noiser = SaltPepperNoiser()
    
    c1e1 = readImage('class1_example1.pgm')
    c1e2 = readImage('class1_example2.pgm')
    c2e1 = readImage('class2_example1.pgm')
    c2e2 = readImage('class2_example2.pgm')
    c3e1 = readImage('class3_example1.pgm')
    c3e2 = readImage('class3_example2.pgm')
    c4e1 = readImage('class4_example1.pgm')
    c4e2 = readImage('class4_example2.pgm')
    examples = numpy.matrix([c1e1, c1e2, c2e1, c2e2, c3e1, c3e2, c4e1, c4e2])
    
    errors = bp.train(examples, examples, 400)
        
    index = 1
    for image in (c1e1, c1e2, c2e1, c2e2, c3e1, c3e2, c4e1, c4e2):
        image = bp.noiser.addNoise(image)
        showImages(image, bp.evaluateNetwork(image), 8, index)
        index += 1
 
    plt.figure(2)
    plt.plot(range(len(errors)), errors)
    plt.show()
    
#testImage()
#testSimpleXor()
