import rnn
import numpy
import recurrent
import matplotlib.pyplot as plt

def plot_errors(errors):
    plt.figure()
    plt.plot(range(len(errors)), errors)
    plt.show()

network = recurrent.RecurrentNetwork(2, [2], 1)
trainer = rnn.RecurrentTrainer(network)
examples = [[[1, 1, -1], [1, 1, 0]],
            [[1, 1, -1], [1, 0, 0]],
            [[1, 1, -1], [0, 1, 0]],
            [[1, 1, -1], [0, 0, 0]]]
classes = [[0, 0, 0],
           [0, 0, 1],
           [0, 0, 1],
           [0, 0, 0]]
errors = trainer.train(examples, classes, 2000)
plot_errors(errors)
