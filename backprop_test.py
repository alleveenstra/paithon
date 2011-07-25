import unittest
import backprop
import numpy
import matplotlib.pyplot as plt

class BackpropagationTestcase(unittest.TestCase):
    def test_xor(self):
        bp = backprop.MultiLayerPerceptron(2, 4, 1, 0.08, 0, 0)
        examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
        classes = numpy.matrix([ [1], [-1], [-1], [1] ]).astype(numpy.float32)
        errors = bp.train(examples, classes, 400)
        assert errors[-1] < 0.1
        
    def test_gradientcheck(self):
        bp = backprop.MultiLayerPerceptron(2, 4, 1, 0.08, 0, 0)
        examples = numpy.matrix([[1]]).astype(numpy.float32)
        classes = numpy.matrix([[1]]).astype(numpy.float32)
