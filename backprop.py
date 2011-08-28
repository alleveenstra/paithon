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

class BackpropagationTrainer:
    def __init__(self, network, eta = 0.08, eta_bias = 0.04, eta_L1 = 0, eta_L2 = 0, eta_momentum = 0, eta_decay = 0):
        self.network = network
        self.eta = eta
        self.eta_bias = eta_bias
        self.eta_L1 = eta_L1
        self.eta_L2 = eta_L2
        self.eta_momentum = eta_momentum
        self.eta_decay = eta_decay
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
        
        # copy the input into the input network activation
        for k in range(len(inputs)):
                    network.activations[0][0, k] = inputs[k]
        
        self.feedForward(network)
        
        return network.activations[network.nLayers - 1]
    
    def feedForward(self, network):
        for layer in range(1, network.nLayers):
            network.activations[layer] = self.activation(network.activations[layer - 1] * 
                                                         network.weights[layer] + 
                                                         network.biases[layer])
    
    def train(self, example, classes, epochs):
        errors = numpy.zeros(epochs) - 1.0
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                input = example[cls, :].tolist()[0]
                target = classes[cls].tolist()[0]
                if self.noiser:
                    input = self.noiser.addNoise(input)
                error += self.feedBackward(input, target)
            errors[epoch] = math.sqrt(error / len(classes))
        return errors[errors > 0]
    
    def feedBackward(self, inputs, target):
        network = self.network
        
        if len(target) != network.outputSize:
            raise ValueError('Target vector of incorrect size')
        
        self.evaluate(inputs)
        self.calculateDeltas(network, target)
        self.updateWeights(network)
        
        return numpy.sum(0.5 * numpy.power(network.activations[network.nLayers - 1] - target, 2))
    
    def calculateDeltas(self, network, target):
        for layer in reversed(range(1, network.nLayers)):
            if layer == network.nLayers - 1:
                network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                                                       (target - network.activations[layer]))
            else:
                network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                                                       (network.deltas[layer + 1] * network.weights[layer + 1].T))
            network.biases[layer] += network.deltas[layer] * self.eta_bias
    
    def updateWeights(self, network):
        for layer in range(1, network.nLayers):
            L1 = numpy.sign(network.activations[layer]) * -self.eta_L1
            L2 = network.activations[layer] * -self.eta_L2
            update = (network.activations[layer - 1].T * (network.deltas[layer] + L1 + L2))
            update += network.previousUpdate[layer] * self.eta_momentum
            update -= network.weights[layer] * self.eta_decay
            network.weights[layer] += update
            network.previousUpdate[layer] = update
    
    def verifyGradient(self, input, target):
        network = self.network
        epsilon = 0.0001
        differences = []
        self.feedBackward(input, target)
        for layer in range(1, network.nLayers):
            savedWeight = numpy.copy(network.weights[layer])
            for i in range(network.size[layer - 1]):
                for j in range(network.size[layer]):
                    positive = numpy.copy(savedWeight)
                    positive[i, j] += epsilon
                    network.weights[layer] = positive
                    output = self.evaluate(input)
                    errorP1 = 0.5 * (numpy.sum((output - target) ** 2))
                    
                    negative = numpy.copy(savedWeight)
                    negative[i, j] -= epsilon
                    network.weights[layer] = negative
                    output = self.evaluate(input)
                    errorP2 = 0.5 * (numpy.sum((output - target) ** 2))
                    
                    approx = (errorP1 - errorP2) / (epsilon * 2)
                    gradient = -(network.activations[layer - 1][0, i] * network.deltas[layer][0, j])
                    
                    differences.append(numpy.abs(gradient - approx))
                    
                    network.weights[layer] = savedWeight
        return numpy.mean(differences)
