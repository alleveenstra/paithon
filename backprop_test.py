import unittest
import backprop
import numpy
import feedforward
import matplotlib.pyplot as plt

class BackpropagationTestcase(unittest.TestCase):
    
    def test_and(self):
        network = feedforward.FeedForwardNetwork(2, [1], 1)
        trainer = backprop.BackpropagationTrainer(network)
        examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
        classes = numpy.matrix([ [-1], [-1], [-1], [1] ]).astype(numpy.float32)
        errors = trainer.train(examples, classes, 500)
        assert errors[-1] < 0.1
    
    def test_or(self):
        network = feedforward.FeedForwardNetwork(2, [1], 1)
        trainer = backprop.BackpropagationTrainer(network)
        examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
        classes = numpy.matrix([ [-1], [1], [1], [1] ]).astype(numpy.float32)
        errors = trainer.train(examples, classes, 500)
        assert errors[-1] < 0.1
    
    def test_xor(self):
        network = feedforward.FeedForwardNetwork(2, [4, 3, 2], 1)
        trainer = backprop.BackpropagationTrainer(network)
        trainer.eta_momentum = 0
        examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
        classes = numpy.matrix([ [1], [-1], [-1], [1] ]).astype(numpy.float32)
        errors = trainer.train(examples, classes, 500)
        assert errors[-1] < 0.1
        
    def test_gradientcheck(self):
        network = feedforward.FeedForwardNetwork(2, [4, 3, 2], 1)
        trainer = backprop.BackpropagationTrainer(network)
        examples = numpy.matrix(numpy.random.uniform(-1, 1, (4, 2))).astype(numpy.float32)
        classes = numpy.matrix(numpy.random.uniform(-1, 1, (4, 1))).astype(numpy.float32)
        errors = trainer.train(examples, classes, 2000)
        differences = trainer.verifyGradient(examples[0, :].tolist()[0], classes[0, :].tolist()[0])
        assert differences < 0.001
        
    def plot_errors(self, errors):
        plt.figure()
        plt.plot(range(len(errors)), errors)
        plt.show()

if __name__ == '__main__':
    unittest.main()
