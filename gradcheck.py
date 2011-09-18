import unittest
import backprop
import numpy
import matplotlib.pyplot as plt

bp = backprop.MultiLayerPerceptron(1, 4, 1, 0.08, 0, 0)
examples = numpy.matrix([[1]]).astype(numpy.float32)
classes = numpy.matrix([[1]]).astype(numpy.float32)
errors = bp.train(examples, classes, 1)
print errors
