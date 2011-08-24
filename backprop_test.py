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
        bp = backprop.MultiLayerPerceptron(2, 4, 1, 0.1, 0, 0)
        examples = numpy.matrix(numpy.random.uniform(-1, 1, (4, 2))).astype(numpy.float32)
        classes = numpy.matrix(numpy.random.uniform(-1, 1, (4, 1))).astype(numpy.float32)
        errors = bp.train(examples, classes, 1000)
        mean_gradient = bp.verifyGradient(examples[0, :].tolist()[0], classes[0, :].tolist()[0])
        assert mean_gradient < 0.0001

if __name__ == '__main__':
    unittest.main()
