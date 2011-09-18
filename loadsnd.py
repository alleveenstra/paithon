import scipy.io
import matplotlib.pyplot as plt
import numpy
import scipy.io.wavfile as wavfile

def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()

w = wavfile.read('data/test2.wav')

channel = w[1]
channel = channel / numpy.std(channel) * 0.3

plot_vector(channel)
