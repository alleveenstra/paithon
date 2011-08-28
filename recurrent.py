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

class RecurrentNetwork(FeedForwardNetwork):
    def __init__(self, inputSize, hiddenShape = [], outputSize = 1):
        super(RecurrentNetwork, self).__init__(inputSize, hiddenShape, outputSize)
        
        self.history = {}
        
        for i in range(self.hiddenSize):
            if i == 0:
                prevLayerSize = self.inputSize
            else:
                prevLayerSize = self.hiddenShape[i - 1]
            self.history[i + 1] = (numpy.matrix(numpy.zeros(layerSize)).astype(numpy.float32))
