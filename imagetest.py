# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 21:01:41 2011

@author: -
"""

import numpy
import scipy
import pylab
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

a = numpy.reshape(pylab.imread('suspect8.pgm'), 64 * 64)
a = ((a - numpy.mean(a)) / numpy.std(a)) * 0.3
a = numpy.reshape(a, (64, 64))

plt.imshow(a, origin='lower')
plt.gray()
plt.show()