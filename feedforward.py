# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon: machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Feed-forward network

"""

import numpy

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
