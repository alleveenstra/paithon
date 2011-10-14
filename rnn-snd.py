# -*- coding: utf-8 -*-
"""
   _____                   
  /  _  \  ** paithon: machine learning framework **
 /  / \  \ 
/ ,,\  \  \
\___/  /  / @author: alle.veenstra@gmail.com
  s    \/
"""

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

import scipy.io
import matplotlib.pyplot as plt
import numpy
import scipy.io.wavfile as wavfile
import recurrent

def plot_vector(x):
    plt.plot(range(len(x)), x)
    plt.show()

def load_snd(id):
    w = wavfile.read('data/%i_update.wav' % id)
    channel = w[1]
    channel = channel / numpy.std(channel) * 0.3
    return channel.tolist()

def generate_output(nClasses, length, cls):
    out = numpy.ones((nClasses, length)) * -0.3
    out[cls, :] = 0.3
    return out.T.tolist()

inputs = {}
outputs = {}

# bell
snd18768 = load_snd(18768)
inputs[0] = snd18768
outputs[0] = generate_output(3, len(snd18768), 0)
# piano
snd21649 = load_snd(21649)
inputs[1] = snd21649
outputs[1] = generate_output(3, len(snd21649), 1)
# clock
snd28113 = load_snd(20043)
inputs[2] = snd28113
outputs[2] = generate_output(3, len(snd28113), 2)

network = recurrent.RecurrentNetwork(1, [8], 3)
trainer = recurrent.RecurrentTrainer(network)
trainer.eta = 0.0001
errors = trainer.train(inputs, outputs, 10)
plot_vector(errors)