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

# 18 classes
train_whistle = [16111, 31512, 67048, 79675, 68281, 79684, 48982, 88425, 11343, 18789, 79681, 57729, 79685, 35719, 16980, 34604]
train_water = [13397, 22866, 67436, 79220, 68332, 80758, 58413, 85599, 13374, 20433, 79226, 60909, 82403, 41634, 17295, 35623]
train_telephone = [27963, 36896, 66290, 75828, 66292, 80341, 64091, 85343, 17895, 30054, 79640, 65514, 85061, 51169, 29621, 44182]
train_speech = [35219, 72880, 86445, 86450, 86446, 87644, 86434, 87709, 31683, 43001, 86459, 86436, 87686, 86394, 42348, 84605]
train_piano = [21649, 31902, 49212, 64816, 49218, 68439, 39198, 83122, 21640, 28309, 68437, 39203, 72046, 39173, 25484, 39163] 
train_horn = [ 2819, 37915, 45930, 58017, 45931, 68121, 42313, 71287, 26177, 2937, 63754, 42315, 68657, 42292, 28915, 42290]
train_guitar = [ 1406, 3479, 58029, 64359, 58973, 71410, 54232, 75320, 12000, 24088, 68968, 54245, 74196, 44298, 15847, 41984]
train_fire = [17748, 39015, 46332, 51452, 47836, 55090, 46317, 73146, 17559, 18614, 53294, 46327, 564, 39036, 18613, 39032]
train_engine = [19106, 2932, 66650, 70117, 69759, 70121, 55025, 77122, 17910, 26972, 70118, 55055, 70123, 36433, 21187, 31970]  
train_door = [17961, 30712, 52265, 57023, 52290, 69386, 43427, 76736, 17905, 20188, 65585, 43429, 69945, 36924, 19853, 35618] 
train_dog = [32794, 50612, 77674, 84653, 84650, 84656, 66546, 84662, 24731, 38121, 84654, 67401, 84660, 55156, 32795, 55154] 
train_cough = [17168, 26296, 51136, 58792, 52323, 65184, 45150, 81903, 16580, 19802, 63579, 46971, 77157, 34303, 19023, 32662] 
train_clock = [28113, 47041, 80298, 80302, 80299, 80304, 75323, 81972, 22830, 41728, 80303, 75710, 80340, 56342, 39146, 51826] 
train_child = [43852, 47315, 77407, 77419, 77412, 77432, 63333, 77451, 31491, 44704, 77427, 63337, 77439, 5013, 44702, 5006]
train_cat = [24734, 53254, 64018, 66513, 66510, 66517, 64012, 77071, 18272, 33657, 66514, 64014, 66518, 60960, 28305, 60959] 
train_breath = [15132, 19865, 75522, 84272, 75523, 84582, 54962, 84594, 15129, 16570, 84376, 62503, 84587, 42918, 16112, 31360]
train_bird = [20043, 34594, 59117, 62158, 59186, 66784, 54083, 72461, 15425, 20684, 64063, 57667, 67439, 43036, 20221, 36913]
train_bell = [18768, 30599, 61439, 66217, 61447, 69511, 53272, 80440, 15086, 26385, 66717, 61031, 70057, 41381, 26034, 34804]
training_set = train_whistle + train_water + train_telephone + train_speech + train_piano + train_horn + train_guitar + train_fire + train_engine + train_door + train_dog + train_cough + train_clock + train_child + train_cat + train_breath + train_bird + train_bell
random.shuffle(training_set)

def which_class(n):
    if n in train_whistle:
        return 0
    if n in train_water:
        return 1
    if n in train_telephone:
        return 2
    if n in train_speech:
        return 3
    if n in train_piano:
        return 4
    if n in train_horn:
        return 5
    if n in train_guitar:
        return 6
    if n in train_fire:
        return 7
    if n in train_engine:
        return 8
    if n in train_door:
        return 9
    if n in train_dog:
        return 10
    if n in train_cough:
        return 11
    if n in train_clock:
        return 12
    if n in train_child:
        return 13
    if n in train_cat:
        return 14
    if n in train_breath:
        return 15
    if n in train_bird:
        return 16
    if n in train_bell:
        return 17
    
def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()

def load_cochleogram(id):
    file = 'data/cochleogram_%i.pkl' % id
    coch = pickle.load(open(file))
    return coch

nClasses = 18

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
        
for i in range(3):
    # train the network for 1 epoch
    trainer.trainEpochs(1)
    print trainer.train()
    
total = 0.0
correct_mode = 0.0
correct_mean = 0
for id in training_set:
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
