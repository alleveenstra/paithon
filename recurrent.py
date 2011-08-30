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

import numpy
import feedforward

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
