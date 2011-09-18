#-*- encoding: utf8 -*- 

# A python script to test and understand pybrain basics
# Based in Martin Felder's FNN pybrain example

from pybrain.tools.shortcuts     import buildNetwork
from pybrain.structure.modules   import SoftmaxLayer, LSTMLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.utilities           import percentError
from pybrain.structure import *
from pybrain.datasets import *
import numpy

# whistle      = [16111, 31512, 67048, 79675, 68281, 79684, 48982, 88425, 11343, 18789, 79681, 57729, 79685, 35719, 16980, 34604]
# water        = [13397, 22866, 67436, 79220, 68332, 80758, 58413, 85599, 13374, 20433, 79226, 60909, 82403, 41634, 17295, 35623]
# telephone    = [27963, 36896, 66290, 75828, 66292, 80341, 64091, 85343, 17895, 30054, 79640, 65514, 85061, 51169, 29621, 44182]
# speech       = [35219, 72880, 86445, 86450, 86446, 87644, 86434, 87709, 31683, 43001, 86459, 86436, 87686, 86394, 42348, 84605]
# piano        = [21649, 31902, 49212, 64816, 49218, 68439, 39198, 83122, 21640, 28309, 68437, 39203, 72046, 39173, 25484, 39163] 
# horn         = [ 2819, 37915, 45930, 58017, 45931, 68121, 42313, 71287, 26177,  2937, 63754, 42315, 68657, 42292, 28915, 42290]
# guitar       = [ 1406,  3479, 58029, 64359, 58973, 71410, 54232, 75320, 12000, 24088, 68968, 54245, 74196, 44298, 15847, 41984]
# fire         = [17748, 39015, 46332, 51452, 47836, 55090, 46317, 73146, 17559, 18614, 53294, 46327,   564, 39036, 18613, 39032]
# engine       = [19106,  2932, 66650, 70117, 69759, 70121, 55025, 77122, 17910, 26972, 70118, 55055, 70123, 36433, 21187, 31970]  
# door         = [17961, 30712, 52265, 57023, 52290, 69386, 43427, 76736, 17905, 20188, 65585, 43429, 69945, 36924, 19853, 35618] 
# dog          = [32794, 50612, 77674, 84653, 84650, 84656, 66546, 84662, 24731, 38121, 84654, 67401, 84660, 55156, 32795, 55154] 
# cough        = [17168, 26296, 51136, 58792, 52323, 65184, 45150, 81903, 16580, 19802, 63579, 46971, 77157, 34303, 19023, 32662] 
# clock        = [28113, 47041, 80298, 80302, 80299, 80304, 75323, 81972, 22830, 41728, 80303, 75710, 80340, 56342, 39146, 51826] 
# child        = [43852, 47315, 77407, 77419, 77412, 77432, 63333, 77451, 31491, 44704, 77427, 63337, 77439,  5013, 44702,  5006]
# cat          = [24734, 53254, 64018, 66513, 66510, 66517, 64012, 77071, 18272, 33657, 66514, 64014, 66518, 60960, 28305, 60959] 
# breath       = [15132, 19865, 75522, 84272, 75523, 84582, 54962, 84594, 15129, 16570, 84376, 62503, 84587, 42918, 16112, 31360]
# bird         = [20043, 34594, 59117, 62158, 59186, 66784, 54083, 72461, 15425, 20684, 64063, 57667, 67439, 43036, 20221, 36913]
# bell         = [18768, 30599, 61439, 66217, 61447, 69511, 53272, 80440, 15086, 26385, 66717, 61031, 70057, 41381, 26034, 34804]


def generate_DS():
    factor = 8.0
    input = numpy.matrix(range(200)).astype(numpy.float32) / factor
    input_sin = numpy.sin(input) * 0.5
    output_sin = numpy.sin(input + 0.1) * 0.5
    DS = SequentialDataSet(1, 1)
    DS.newSequence()
    for i in range(len(input_sin)):
        print i, input_sin[0, i], output_sin[0, i]
        DS.appendLinked([input_sin[0, i]], [output_sin[0, i]])
    return DS

DS = generate_DS()

# build a feed-forward network with 20 hidden units, plus 
# a corresponding trainer

fnn = buildNetwork(1, 8, 1, hiddenclass = LSTMLayer, outclass = TanhLayer, outputbias = False, recurrent = True)

net = RecurrentNetwork()
net.addInputModule(LinearLayer(1, name = 'in'))
net.addModule(SigmoidLayer(3, name = 'hidden'))
net.addOutputModule(LinearLayer(1, name = 'out'))
net.addConnection(FullConnection(net['in'], net['hidden'], name = 'c1'))
net.addConnection(FullConnection(net['hidden'], net['out'], name = 'c2'))
net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name = 'c3'))
net.sortModules()

#trainer = RPropMinusTrainer(fnn, dataset = DS, verbose = True)
trainer = BackpropTrainer(fnn, dataset = DS, verbose = True)

trainer.trainEpochs(1000)
