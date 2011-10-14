import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import numpy
from pybrain.structure import *
from pybrain.datasets import *
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer, LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.utilities           import percentError
import scipy.io.wavfile as wavfile
import pickle
import random

# 6 classes
dog_train = [66587, 47926, 68389, 41918, 74495, 63292, 54545, 11722, 7913, 55005]
cat_train = [19278, 88475, 70907, 49784, 39989, 39990, 29676, 28112, 1730, 67448]
bird_train = [34690, 35133, 19242, 9328, 34207, 69575, 75073, 71857, 69574, 56234]
bell_train = [30602, 72227, 19573, 85797, 24102, 77693, 30599, 38229, 19459, 85798]
piano_train = [29843, 61228, 47793, 85383, 47875, 14520, 64805, 64804, 64803, 29821]
talking_train = [59772, 86466, 45115, 86477, 62951, 45118, 86473, 62950, 86384, 60563]
training_set = dog_train + cat_train + bird_train + bell_train + piano_train + talking_train
random.shuffle(training_set)

dog_test = [55365, 85362, 72579, 86279, 84317]
cat_test = [31505, 77404, 33657, 26769, 56380]
bird_test = [69527, 983, 67030, 74825, 85401]
bell_test = [32320, 85799, 85802, 49596, 34381]
piano_test = [63393, 47790, 85385, 64811, 85380]
talking_test = [86457, 86456, 86452, 60566, 86478]
test_set = dog_test + cat_test + bird_test + bell_test + piano_test + talking_test

def which_class(n):
    if n in dog_train or n in dog_test:
        return 0
    if n in cat_train or n in cat_test:
        return 1
    if n in bird_train or n in bird_test:
        return 2
    if n in bell_train or n in bell_test:
        return 3
    if n in piano_train or n in piano_test:
        return 4
    if n in talking_train or n in talking_test:
        return 5
    
def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()

def load_cochleogram(id):
    file = 'data-8/pickle/cochleogram_%i.pkl' % id
    coch = pickle.load(open(file))
    return coch

nClasses = 6

def append2DS(DS, cochleo, cls, nClasses):
    DS.newSequence()
    out = numpy.zeros(nClasses)
    out[cls] = 1
    out = out.tolist()
    for i in range(cochleo.shape[1]):
        DS.appendLinked(cochleo[:, i].T.tolist()[0], out)

DS = SequentialDataSet(100, nClasses)
    
for id in training_set:
    cls = which_class(id)
    data = load_cochleogram(id)
    append2DS(DS, data, cls, nClasses)

# fnn = buildNetwork(1, 15, 5, hiddenclass = LSTMLayer, outclass = SoftmaxLayer, outputbias = False, recurrent = True)

fnn = buildNetwork(100, 18, nClasses, hiddenclass = LSTMLayer, outclass = SoftmaxLayer, outputbias = False, recurrent = True)

# Create a trainer for backprop and train the net.
#trainer = BackpropTrainer(fnn, DStrain, learningrate = 0.005)

trainer = RPropMinusTrainer(fnn, dataset = DS, verbose = True)
        
for i in range(4):
    # train the network for 1 epoch
    trainer.trainEpochs(1)
    print trainer.train()
    
total = 0.0
correct_mode = 0.0
correct_mean = 0
for id in test_set:
    cls = which_class(id)
    data = load_cochleogram(id)
    fnn.reset()
    classifications_list = []
    summed = numpy.zeros(nClasses)
    for i in range(data.shape[1]):
        output = fnn.activate(data[:, i].T.tolist()[0])
        summed += output 
        classifications_list.append(numpy.argmax(output))
    result = summed / data.shape[1]
    sample_mode = stats.mode(classifications_list)[0][0]
    sample_mean = numpy.argmax(result)
    print cls, sample_mode
    total += 1
    if cls == sample_mean:
        correct_mean += 1
    if cls == sample_mode:
        correct_mode += 1

print correct_mode / total * 100.0, ' percent correct! (statistical mode)'
print correct_mean / total * 100.0, ' percent correct! (statistical mean)'

#fileNet = open('network.pkl', 'wb')
#fileTrainer = open('trainer.pkl', 'wb')
#pickle.dump(fnn, fileNet)
#pickle.dump(trainer, fileTrainer)
#fileNet.close()
#fileTrainer.close()
#
#print 'everything quite dandy'
