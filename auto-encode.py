import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import scipy.misc.pilutil as pilutil
import numpy
from pybrain.structure import *
from pybrain.datasets import *
import cPickle as pickle
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer, LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.utilities           import percentError
import scipy.io.wavfile as wavfile
import random
import sys
import data

nHidden = int(sys.argv[1])
nEpoch = int(sys.argv[2])
segments = 100
nClasses = 8

print 'Running with %d segments, %d hidden units and for %d epochs' % (segments, nHidden, nEpoch)
    
# Create the dataset
def append2DS(DS, cochleo):
    DS.newSequence()
    nFrames = cochleo.shape[1]
    for i in range(nFrames):
        DS.appendLinked(cochleo[:, i].T.tolist()[0], cochleo[:, i].T.tolist()[0])

DS = SequentialDataSet(segments, segments)
    
for id in data.train_set:
    cls = data.which_class(id)
    cochleogram = data.load_cochleogram(id)
    append2DS(DS, cochleogram)

# Build or load the network
if len(sys.argv) == 4:
    file = open(sys.argv[3], 'r')
    fnn = pickle.load(file)
    file.close()
    fnn.sorted = False
    fnn.sortModules()
else:
    fnn = buildNetwork(segments, nHidden, segments, hiddenclass = LSTMLayer, outclass = TanhLayer, outputbias = True, recurrent = True, peepholes = False, fast = False)

trainer = RPropMinusTrainer(fnn, dataset = DS, verbose = True)

print 'Begin training...'
        
# Train the network
trainer.trainEpochs(nEpoch)

# Store the encoder
file = open('autoencoder-%i.xml' % (nHidden), 'w')
pickle.dump(fnn, file)
file.close()

# Show an example 
id = 86464
cochleogram = data.load_cochleogram(id)
fnn.reset()
haha = numpy.zeros(cochleogram.shape, dtype=numpy.float32)
for i in range(cochleogram.shape[1]):
    output = fnn.activate(cochleogram[:, i].T.tolist()[0])
    haha[:, i] = output

plt.imshow(cochleogram)
plt.savefig('example.png')
    
plt.imshow(haha)
plt.savefig('encoded-%i.png' % (nHidden))
