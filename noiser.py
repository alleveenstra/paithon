# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/

Noisers

"""

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
