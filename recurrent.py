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
        feedforward.FeedForwardNetwork.__init__(self, inputSize, hiddenShape, outputSize)
        
        self.historyActivation = {}
        self.historyWeights = {}
        self.historyDeltas = {}
        
        for i in range(self.hiddenSize):
            if i == 0:
                prevLayerSize = self.inputSize
            else:
                prevLayerSize = self.hiddenShape[i - 1]
            layerSize = self.hiddenShape[i]
            self.historyActivation[i + 1] = (numpy.matrix(numpy.zeros(layerSize)).astype(numpy.float32))
            self.historyWeights[i + 1] = (numpy.matrix(numpy.random.normal(0, 0.5, (layerSize, layerSize))).astype(numpy.float32))

