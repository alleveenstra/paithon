# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:41:39 2011

@author: alle.veenstra@gmail.com
"""

import numpy
import math
import pylab
import matplotlib.pyplot as plt
from ctypes import ARRAY

class SaltPepperNoiser(object):
    def __init__(self):
        self.amount = 0.3
        self.salt = 0.5
        self.pepper = -0.5
        
    def addNoise(self, input):
        output = numpy.array(input)
        half = len(output) * self.amount / 2.0
        salt = numpy.random.randint(0, len(output) - 1, half)
        pepper = numpy.random.randint(0, len(output) - 1, half)
        output[salt] = self.salt
        output[pepper] = self.pepper
        return output

class BackProp:
    def __init__(self, nInput, nHidden, nOutput, eta):
        
        self.nInput = nInput + 1
        self.nHidden = nHidden
        self.nOutput = nOutput
        
        self.inputActivation = numpy.matrix(0 - numpy.ones(self.nInput))
        self.hiddenActivation = numpy.matrix(numpy.ones(self.nHidden))
        self.outputActivation = numpy.matrix(numpy.ones(self.nOutput))

        self.hiddenBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nHidden))
        self.outputBias = numpy.matrix(numpy.random.normal(0, 0.5, self.nOutput))

        self.hiddenWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nInput, self.nHidden)))
        self.outputWeight = numpy.matrix(numpy.random.normal(0, 0.5, (self.nHidden, self.nOutput)))

        self.activation = numpy.vectorize(self.activationFunction)
        self.gradient = numpy.vectorize(self.gradientFunction)

        self.eta = eta
        self.eta_bias = 0.1
        self.eta_L1 = 0.005
        
        self.noiser = False
    
    def activationFunction(self, value):
        return math.tanh(value)        
        
    def gradientFunction(self, value):
        return 1.0 - math.tanh(value) ** 2
        
    def update(self, inputs):
        if len(inputs) != self.nInput - 1:
            raise ValueError('Input vector not long enough')
        
        for k in range(len(inputs)):
            self.inputActivation[0, k] = inputs[k]
            
        self.hiddenActivation = self.activation(self.inputActivation * self.hiddenWeight + self.hiddenBias)

        self.outputActivation = self.activation(self.hiddenActivation * self.outputWeight + self.outputBias)

        return self.outputActivation
        
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
                self.update(target)
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

def testSimpleXor():
    bp = BackProp(2, 4, 1, 0.1)
    examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
    classes = numpy.matrix([ [1], [-1], [-1], [1] ])
    errors = bp.train(examples, classes, 5000)
    
    print '[0,0] -> %.2f' % bp.update([0, 0])[0, 0]
    print bp.hiddenActivation    
    print '[1,0] -> %.2f' % bp.update([1, 0])[0, 0]
    print bp.hiddenActivation
    print '[0,1] -> %.2f' % bp.update([0, 1])[0, 0]
    print bp.hiddenActivation
    print '[1,1] -> %.2f' % bp.update([1, 1])[0, 0]
    print bp.hiddenActivation
    
    plt.plot(range(len(errors)), errors)
    plt.show()

#===============================================================================
    #plt.plot(range(len(errors)), errors)
    #plt.show()
#===============================================================================
#    x = numpy.divide(range(-200,200), 100.0)
#    plt.plot(x, bp.gradient(x))
#    plt.show()
#===============================================================================

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
    bp = BackProp(64 * 64, 4, 64 * 64, 0.08)
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
        showImages(image, bp.update(image), 8, index)
        index += 1
 
    plt.figure(2)
    plt.plot(range(len(errors)), errors)
    plt.show()
    
testImage()
#testSimpleXor()
