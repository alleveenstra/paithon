# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon: machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Simple recurrent neural network

"""

import numpy
import math
import matplotlib.pyplot as plt

class RecurrentBackpropagationTrainer:
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
    
    def evaluateSerie(self, serie):
        out = []
        for item in range(len(serie)):
            input = numpy.matrix(serie[item]).astype(numpy.float32)
            out.append(self.evaluate(input)[0, 0])
        return out
    
    def evaluate(self, inputs):
        network = self.network
        
        if len(inputs) != network.inputSize - network.hiddenShape[0]:
            raise ValueError('Input vector of incorrect size')
        
        # copy the input into the input network activation
        input_length = len(inputs)
        for k in range(input_length):
            network.activations[0][0, k] = inputs[k]
        # copy the first hidden layer's activation into the input network activation
        for l in range(network.hiddenShape[0]):
            network.activations[0][0, input_length + l] = network.activations[1][0, l]
        
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
            for item in range(len(classes)):
                input = numpy.matrix(example[item]).astype(numpy.float32)
                target = numpy.matrix(classes[item]).astype(numpy.float32)
                
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


class SimpleRecurrentNetwork:
    def __init__(self, inputSize, hiddenShape = [], outputSize = 1):
        self.inputSize = inputSize + hiddenShape[0]
        self.outputSize = outputSize
        hiddenShape.append(outputSize)
        self.hiddenSize = len(hiddenShape)
        self.hiddenShape = hiddenShape
        self.nLayers = 1 + self.hiddenSize
        
        self.size = {}
        self.activations = {}
        self.biases = {}
        self.previousUpdate = {}
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
            self.previousUpdate[i + 1] = (numpy.matrix(numpy.zeros((prevLayerSize, layerSize))).astype(numpy.float32))
            self.weights[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, (prevLayerSize, layerSize))).astype(numpy.float32))
            self.activations[i + 1] = (numpy.matrix(numpy.ones(layerSize)).astype(numpy.float32))
            self.biases[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, layerSize)).astype(numpy.float32))
    
    def reset(self):
        self.activations[1] = (numpy.matrix(numpy.ones(self.hiddenShape[0])).astype(numpy.float32))

def plot_vector(errors, n):
    plt.subplot(2, 2, n)
    plt.plot(range(len(errors)), errors)
    if n == 4:
        plt.show()
    
factor = 8.0
input = numpy.matrix(range(200)).astype(numpy.float32) / factor

input_sin = ((numpy.sin(input)) * 0.5).tolist()
output_sin = ((numpy.sin(input * 2)) * 0.5).tolist()

network = SimpleRecurrentNetwork(1, [8], 1)
trainer = RecurrentBackpropagationTrainer(network)
trainer.eta = 0.08
trainer.eta_bias = 0
trainer.eta_decay = 0
examples = input_sin
classes = output_sin
errors = trainer.train(examples, classes, 100)
x = trainer.evaluateSerie(input_sin[0])
plot_vector(errors, 1)
plot_vector(input_sin[0], 2)
plot_vector(output_sin[0], 3)
plot_vector(x, 4)

