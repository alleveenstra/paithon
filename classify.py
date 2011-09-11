import scipy.io
import matplotlib.pyplot as plt
import numpy
from pybrain.structure import *
from pybrain.datasets import *
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer, LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.utilities           import percentError
import scipy.io.wavfile as wavfile

def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()

def load_snd(id):
    w = wavfile.read('data/%i_update.wav' % id)
    channel = w[1]
    channel = channel / numpy.std(channel) * 0.3
    return channel.tolist()

nClasses = 3

def append2DS(DS, snd, cls, nClasses):
    DS.newSequence()
    out = numpy.zeros(nClasses) - 1
    out[cls] = 1
    out = out.tolist()
    for i in range(len(snd)):
        DS.appendLinked([snd[i]], out)

DS = SequentialDataSet(1, nClasses)
    
# bell
snd_18768 = load_snd(18768)
append2DS(DS, snd_18768, 0, nClasses)
# piano
snd_21649 = load_snd(21649)
append2DS(DS, snd_21649, 1, nClasses)
# clock
snd_20043 = load_snd(20043)
append2DS(DS, snd_20043, 2, nClasses)

# fnn = buildNetwork(1, 15, 5, hiddenclass = LSTMLayer, outclass = SoftmaxLayer, outputbias = False, recurrent = True)

fnn = buildNetwork(1, 1, nClasses, hiddenclass = LSTMLayer, outclass = TanhLayer, outputbias = False, recurrent = True)

# Create a trainer for backprop and train the net.
#trainer = BackpropTrainer(fnn, DStrain, learningrate = 0.005)

trainer = RPropMinusTrainer(fnn, dataset = DS, verbose = True)
        
for i in range(4):
    # train the network for 1 epoch
    trainer.trainEpochs(1)
    print trainer.train()

fnn.reset()
summed = numpy.zeros(nClasses)
for sample in snd_18768:
    summed += fnn.activate([sample])
print summed / len(snd_18768)

fnn.reset()
summed = numpy.zeros(nClasses)
for sample in snd_21649:
    summed += fnn.activate([sample])
print summed / len(snd_21649)

fnn.reset()
summed = numpy.zeros(nClasses)
for sample in snd_20043:
    summed += fnn.activate([sample])
print summed / len(snd_20043)
