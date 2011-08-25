# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Multi-layer perceptron

"""

import numpy
import math

class FeedForwardNetwork:
    def __init__(self, inputSize, hiddenShape = [], outputSize = 1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        hiddenShape.append(outputSize)
        self.hiddenSize = len(hiddenShape)
        self.hiddenShape = hiddenShape
        self.nLayers = 1 + self.hiddenSize
        
        self.size = {}
        self.activations = {}
        self.biases = {}
        self.weights = {}
        self.deltas = {}
        
        self.size[0] = inputSize
        self.activations[0] = (numpy.matrix(0 - numpy.ones(self.inputSize)).astype(numpy.float32))
        
        for i in range(self.hiddenSize):
            if i == 0:
                prevLayerSize = self.inputSize
            else:
                prevLayerSize = self.hiddenShape[i - 1]
            layerSize = self.hiddenShape[i]
            self.size[i + 1] = layerSize
            self.weights[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, (prevLayerSize, layerSize))).astype(numpy.float32))
            self.activations[i + 1] = (numpy.matrix(numpy.ones(layerSize)).astype(numpy.float32))
            self.biases[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, layerSize)).astype(numpy.float32))

class BackpropagationTrainer:
    def __init__(self, network, eta = 0.08, eta_bias = 0.04, eta_L1 = 0.005):
        self.network = network
        self.eta = eta
        self.eta_bias = eta_bias
        self.eta_L1 = eta_L1
        self.activation = numpy.vectorize(self.activationFunction)
        self.derivative = numpy.vectorize(self.derivedActivationFunction)
        self.noiser = False

    def activationFunction(self, value):
        return math.tanh(value)        

    def derivedActivationFunction(self, value):
        # The derived activation function is actually 1 - tanh^2(x)
        return 1.0 - value ** 2
    
    def evaluate(self, inputs):
        network = self.network
        
        if len(inputs) != network.inputSize:
            raise ValueError('Input vector of incorrect size')
        
        for layer in range(network.nLayers):
            if layer == 0:
                for k in range(len(inputs)):
                    network.activations[layer][0, k] = inputs[k]
            else:
                network.activations[layer] = self.activation(network.activations[layer - 1] * 
                                                             network.weights[layer] + 
                                                             network.biases[layer])
        return network.activations[network.nLayers - 1]
    
    def learn(self, inputs, target):
        network = self.network
        
        if len(target) != network.outputSize:
            raise ValueError('Target vector of incorrect size')
        
        self.evaluate(inputs)
        
        for layer in reversed(range(1, network.nLayers)):
            if layer == network.nLayers - 1:
                network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                                                       (target - network.activations[layer]))
            else:
                network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                                                       (network.deltas[layer + 1] * network.weights[layer + 1].T))
            network.biases[layer] += network.deltas[layer] * self.eta_bias

        for layer in range(1, network.nLayers):
            network.weights[layer] += (network.activations[layer - 1].T * network.deltas[layer])
        
        return numpy.sum(0.5 * numpy.power(network.activations[network.nLayers - 1] - target, 2))

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
        network = self.network
        epsilon = 0.0001
        differences = []
        self.learn(input, target)
        for layer in range(1, network.nLayers):
            savedWeight = numpy.copy(network.weights[layer])
            for i in range(network.size[layer - 1]):
                for j in range(network.size[layer]):
                    positive = numpy.copy(savedWeight)
                    negative = numpy.copy(savedWeight)
                       
                    positive[i, j] += epsilon
                    negative[i, j] -= epsilon
                    
                    network.weights[layer] = positive
                    output = self.evaluate(input)
                    errorP1 = 0.5 * (numpy.sum((output - target) ** 2))
                    
                    network.weights[layer] = negative
                    output = self.evaluate(input)
                    errorP2 = 0.5 * (numpy.sum((output - target) ** 2))
                    
                    approx = (errorP1 - errorP2) / (epsilon * 2)
                    gradient = -(network.activations[layer - 1][0, i] * network.deltas[layer][0, j])
                    
                    differences.append(numpy.abs(gradient - approx))
                    
                    network.weights[layer] = savedWeight
        return differences

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
