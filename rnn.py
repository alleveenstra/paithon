# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Recurrect network trainer

"""

import numpy
import math
import backprop

class RecurrentTrainer(backprop.BackpropagationTrainer):
    def __init__(self, network, eta = 0.08):
        self.network = network
        self.eta = eta
        self.activation = numpy.vectorize(self.activationFunction)
        self.derivative = numpy.vectorize(self.derivedActivationFunction)
        
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
                                                         network.weights[layer])
            if layer < network.nLayers - 1:
                network.activations[layer] += self.activation(network.historyActivation[layer] * 
                                                              network.historyWeights[layer])
                network.historyActivation[layer] = network.activations[layer]
    
    def train(self, example, classes, epochs):
        errors = numpy.zeros(epochs) - 1.0
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                input = example[cls, :].tolist()[0]
                target = classes[cls].tolist()[0]
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
                print (network.deltas[layer] * network.weights[layer].T).shape
                print self.derivative(network.historyActivation[layer]).shape
                
                #network.historyDeltas[layer] = numpy.multiply(self.derivative(network.historyActivation[layer]),
                #                                       (network.deltas[layer] * network.weights[layer].T))
    
    def updateWeights(self, network):
        for layer in range(1, network.nLayers):
            update = (network.activations[layer - 1].T * network.deltas[layer])
            network.weights[layer] += update
            if layer < network.nLayers - 1:
                network.historyWeights += (network.historyActivation[layer].T * network.historyDeltas[layer])
