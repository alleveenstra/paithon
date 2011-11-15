import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
import scipy.misc.pilutil as pilutil
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
import sys

nHidden = int(sys.argv[1])
nEpoch = int(sys.argv[2])
segments = 100
nClasses = 8

print 'Running with %d segments, %d hidden units and for %d epochs' % (segments, nHidden, nEpoch)

talking_train = [86384, 45639, 62445, 86465, 60567, 45115, 60566, 86452, 86385, 62948, 86451, 86459, 45638, 86478, 86457, 86476, 59772, 86463, 86456, 45118]
sax_train = [665, 761, 673, 1236, 659, 72469, 734, 731, 667, 767, 736, 44639, 44635, 769, 44631, 764, 733, 735, 762, 677]
piano_train = [85382, 85386, 47911, 85383, 29822, 64802, 64795, 63394, 47793, 85387, 63392, 49292, 85380, 29825, 47833, 64811, 47875, 63391, 64819, 63393]
footsteps_train = [51629, 87519, 60614, 29600, 73102, 72737, 25005, 87520, 51150, 73105, 87522, 26372, 85110, 78369, 25077, 51149, 84873, 87521, 56354, 67145]
dog_train = [58382, 4918, 15792, 74495, 55365, 43787, 66587, 30226, 77674, 67049, 63292, 34872, 4912, 38121, 43786, 85663, 62048, 17414, 32318, 47926]
car_train = [27819, 85517, 68663, 84625, 38682, 19979, 44770, 19981, 50898, 82308, 2933, 68662, 38683, 35498, 68660, 21741, 81575, 89567, 55042, 74969]
bird_train = [20221, 69575, 43036, 72892, 34694, 85405, 9328, 56549, 75073, 44526, 31384, 983, 69357, 15425, 56234, 35133, 30446, 66524, 28244, 34690]
bell_train = [38229, 78403, 24102, 30599, 27759, 15402, 77694, 66445, 13722, 72227, 30600, 30157, 26874, 2539, 30153, 77692, 49596, 32320, 9218, 69385]

train_set = talking_train + sax_train + piano_train + footsteps_train + dog_train + car_train + bird_train + bell_train

talking_test = [86461, 86450, 86464, 60563, 62949, 86475, 83521, 62946, 86466, 62951]
sax_test = [32236, 72468, 44650, 1244, 44632, 737, 44628, 763, 1242, 662]
piano_test = [47832, 47790, 64805, 14520, 85385, 29821, 61228, 64803, 64825, 64827]
footsteps_test = [70024, 25208, 71887, 71891, 73101, 25262, 71888, 83969, 67144, 85560]
dog_test = [41918, 73373, 68389, 30344, 36902, 63261, 74723, 7913, 11722, 33849]
car_test = [44767, 18592, 19980, 76807, 38685, 58548, 89944, 50920, 59668, 89572]
bird_test = [68646, 22916, 54523, 22919, 28327, 69527, 57906, 28328, 69574, 34687]
bell_test = [19459, 19460, 77693, 87286, 59434, 26373, 85802, 80341, 15401, 19458]

test_set = talking_test + sax_test + piano_test + footsteps_test + dog_test + car_test + bird_test + bell_test

talking_validation = [86473, 50497, 86470, 62950, 86477, 86454, 86453, 86468, 70986, 60560]
sax_validation = [44640, 765, 72462, 738, 44651, 660, 770, 732, 675, 44645]
piano_validation = [29843, 64804, 47791, 29824, 85384, 61227, 64829, 64796, 47789, 32158]
footsteps_validation = [71777, 85132, 77094, 71889, 85208, 71925, 73104, 29601, 73103, 57801]
dog_validation = [86279, 55013, 24965, 54545, 72579, 14506, 85362, 78324, 84317, 55005]
car_validation = [38687, 75106, 54271, 39011, 36837, 7803, 38690, 79261, 44765, 50864]
bird_validation = [74825, 50335, 34207, 9327, 67439, 85401, 7802, 15543, 67030, 80706]
bell_validation = [87287, 1471, 19457, 30160, 54277, 19599, 85795, 30602, 29623, 26875]

validation_set = talking_validation + sax_validation + piano_validation + footsteps_validation + dog_validation + car_validation + bird_validation + bell_validation

talking = talking_train + talking_test + talking_validation
sax = sax_train + sax_test + sax_validation
piano = piano_train + piano_test + piano_validation
footsteps = footsteps_train + footsteps_test + footsteps_validation
dog = dog_train + dog_test + dog_validation
car = car_train + car_test + car_validation
bird = bird_train + bird_test + bird_validation
bell = bell_train + bell_test + bell_validation

everything = talking + sax + piano + footsteps + dog + car + bird + bell

random.shuffle(train_set)

def which_class(n):
    if n in talking:
        return 0
    if n in sax:
        return 1
    if n in piano:
        return 2
    if n in footsteps:
        return 3
    if n in dog:
        return 4
    if n in car:
        return 5
    if n in bird:
        return 6
    if n in bell:
        return 7
    
def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()
    
def resampleMatrix(input, factor):
    new_x = int(input.shape[0] / factor)
    output = numpy.zeros((new_x, input.shape[1]))
    for x in range(new_x):
        x_start = x * factor
        x_end = (x + 1) * factor
        output[x, :] = numpy.mean(input[x_start : x_end, :], 0)
    return output

def load_cochleogram(id):
    file = 'data-8/pickle/cochleogram_%i.pkl' % id
    coch = pickle.load(open(file))
    return coch

def append2DS(DS, cochleo):
    DS.newSequence()
    nFrames = cochleo.shape[1]
    for i in range(nFrames - 1):
        DS.appendLinked(cochleo[:, i].T.tolist()[0], cochleo[:, i + 1].T.tolist()[0])

DS = SequentialDataSet(segments, segments)
    
for id in train_set:
    cls = which_class(id)
    data = load_cochleogram(id)
    append2DS(DS, data)

fnn = buildNetwork(segments, nHidden, segments, hiddenclass = LSTMLayer, outclass = SoftmaxLayer, outputbias = True, recurrent = True, peepholes = True)

trainer = RPropMinusTrainer(fnn, dataset = DS, verbose = True)

print 'Begin training...'
        
# train the network
for i in range(nEpoch):
    trainer.trainEpochs(1)
    print trainer.train()

# evaluate on the test set
def extractActivation():
    for id in test_set:
        cls = which_class(id)
        data = load_cochleogram(id, factor)
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
        total += 1
        print 'class: ', cls, ' recognition: ', sample_mean
        if cls == sample_mean:
            correct_mean += 1
        if cls == sample_mode:
            correct_mode += 1
