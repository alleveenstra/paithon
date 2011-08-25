import unittest
import backprop
import numpy

class BackpropagationTestcase(unittest.TestCase):
    
    def test_xor(self):
        network = backprop.FeedForwardNetwork(2, [4, 3, 2], 1)
        trainer = backprop.BackpropagationTrainer(network)
        examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
        classes = numpy.matrix([ [1], [-1], [-1], [1] ]).astype(numpy.float32)
        errors = trainer.train(examples, classes, 2000)
        assert errors[-1] < 0.1
        
    def test_gradientcheck(self):
        network = backprop.FeedForwardNetwork(2, [4, 3, 2], 1)
        trainer = backprop.BackpropagationTrainer(network)
        examples = numpy.matrix(numpy.random.uniform(-1, 1, (4, 2))).astype(numpy.float32)
        classes = numpy.matrix(numpy.random.uniform(-1, 1, (4, 1))).astype(numpy.float32)
        errors = trainer.train(examples, classes, 100)
        differences = trainer.verifyGradient(examples[0, :].tolist()[0], classes[0, :].tolist()[0])
        for difference in differences:
            assert difference < 0.001

if __name__ == '__main__':
    unittest.main()
