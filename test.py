import rnn
import numpy
import recurrent
import matplotlib.pyplot as plt

def plot_vector(errors, n):
    plt.subplot(2, 2, n)
    plt.plot(range(len(errors)), errors)
    if n == 4:
        plt.show()
    
factor = 8.0
input = numpy.matrix(range(200)).astype(numpy.float32) / factor

input_sin = (numpy.sin(input) * 0.5).tolist()
output_sin = (numpy.sin(input + 0.1 * 3.14) * 0.5).tolist()

network = recurrent.RecurrentNetwork(1, [8], 1)
trainer = rnn.RecurrentTrainer(network)
examples = input_sin
classes = output_sin
errors = trainer.train(examples, classes, 1000)
x = trainer.evaluateSerie(input_sin[0])
plot_vector(errors, 1)
plot_vector(input_sin[0], 2)
plot_vector(output_sin[0], 3)
plot_vector(x, 4)
