# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon: machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Elman-style recurrent network

"""

import feedforward
import backprop
import numpy
import math
import matplotlib.pyplot as plt

class RecurrentNetwork(feedforward.FeedForwardNetwork):
    def __init__(self, inputSize, hiddenShape = [], outputSize = 1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        hiddenShape.append(outputSize)
        self.hiddenSize = len(hiddenShape)
        self.hiddenShape = hiddenShape
        self.nLayers = 1 + self.hiddenSize
        
        self.size = {}
        self.activations = {}
        self.historyWeights = {}
        self.historyActivation = {}
        self.weights = {}
        self.deltas = {}
        self.historyDeltas = {}
        
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
            self.historyActivation[i + 1] = (numpy.matrix(numpy.zeros(layerSize)).astype(numpy.float32))
            self.historyWeights[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, (layerSize, layerSize))).astype(numpy.float32))

    def reset(self):
        for i in range(1, self.hiddenSize):
            layerSize = self.hiddenShape[i]
            self.historyActivation[i + 1] = (numpy.matrix(numpy.zeros(layerSize)).astype(numpy.float32))

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
    
    def evaluateSerie(self, serie):
        out = []
        for item in range(len(serie)):
            input = numpy.matrix(serie[item]).astype(numpy.float32)
            out.append(self.evaluate(input)[0, 0])
        return out
    
    def evaluate(self, inputs):
        network = self.network
        
        if len(inputs) != network.inputSize:
            raise ValueError('Input vector of incorrect size')
        
        # copy the input into the input network activation
        for k in range(len(inputs)):
            network.activations[0][0, k] = inputs[k]
        # copy the hidden layers
        self.copyHistory()
        
        self.feedForward(network)
        
        return network.activations[network.nLayers - 1]
    
    def feedForward(self, network):
        for layer in range(1, network.nLayers):
            if layer < network.nLayers - 1:
                network.activations[layer] = self.activation(network.historyActivation[layer] * 
                                                             network.historyWeights[layer] + 
                                                             network.activations[layer - 1] * 
                                                             network.weights[layer])
            else:
                network.activations[layer] = self.activation(network.activations[layer - 1] * 
                                                             network.weights[layer])
    
    def train(self, example, classes, epochs):
        errors = numpy.zeros(epochs) - 1.0
        for epoch in range(epochs):
            error = 0
            for cls in range(len(classes)):
                input = numpy.matrix(example[cls]).astype(numpy.float32)
                target = numpy.matrix(classes[cls]).astype(numpy.float32)
                
                # iterate over the sequence
                for seq in range(input.shape[1]):
                    error += self.feedBackward(input[:, seq], target[:, seq])
                self.network.reset()
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
                
                network.historyDeltas[layer] = numpy.multiply(self.derivative(network.historyActivation[layer]),
                                                       (network.deltas[layer]))
                
    def updateWeights(self, network):
        for layer in range(1, network.nLayers):
            update = (network.activations[layer - 1].T * network.deltas[layer])
            network.weights[layer] += update
            if layer < network.nLayers - 1:
                network.historyWeights[layer] += (network.historyActivation[layer].T * network.historyDeltas[layer])
                
    def copyHistory(self):
        network = self.network
        for layer in range(1, network.nLayers - 1):
            network.historyActivation[layer] = numpy.copy(network.activations[layer])

def plot_vector(errors, n):
    plt.subplot(2, 2, n)
    plt.plot(range(len(errors)), errors)
    if n == 4:
        plt.show()
    
factor = 8.0
input = numpy.matrix(range(200)).astype(numpy.float32) / factor

input_sin = (numpy.sin(input) * 0.5).tolist()
output_sin = (numpy.sin(input + 0.1 * 3.14) * 0.5).tolist()

network = RecurrentNetwork(1, [8], 1)
trainer = RecurrentTrainer(network)
examples = input_sin
classes = output_sin
errors = trainer.train(examples, classes, 100)
x = trainer.evaluateSerie(input_sin[0])
plot_vector(errors, 1)
plot_vector(input_sin[0], 2)
plot_vector(output_sin[0], 3)
plot_vector(x, 4)
