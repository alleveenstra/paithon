# -*-

import pickle
import numpy
import loader

def dictSort(d):
    """ returns a dictionary sorted by keys """
    our_list = d.items()
    our_list.sort()
    k = {}
    for item in our_list:
        k[item[0]] = item[1]
    return k

id = 0

pkl_file = open('factorization.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

A = data['A'].T
B = data['B'].T

movie = A[id, :]
movie /= numpy.linalg.norm(movie)

print movie.shape

titles = loader.loadTitles()

values = numpy.zeros((A.shape[0], 2))
for m in range(A.shape[0]):
    other = A[m, :] 
    other /= numpy.linalg.norm(other)
    distance = 1.0 - numpy.inner(movie , other)[0, 0]
    values[m, 0] = numpy.abs(distance)
    values[m, 1] = m

sortIndex = numpy.argsort(values[:, 0])

for movie in values[sortIndex[0:10], 1]:
    print titles[movie]
