import numpy
import backprop
import pylab
import matplotlib.pyplot as plt

def readImage(filename, dist_width = 0.3):
    image = numpy.reshape(pylab.imread(filename), 64 * 64)
    image = ((image - numpy.mean(image)) / numpy.std(image)) * dist_width
    return image
    
def showImages(before, after, n_images = 1, n_image = 1):
    before = numpy.matrix(numpy.reshape(before, (64, 64)))
    after = numpy.matrix(numpy.reshape(after, (64, 64)))    
    if n_image == 1:
        plt.figure(1)
    plt.subplot(n_images / 2, 4, 1 + (n_image - 1) * 2)
    plt.imshow(before, origin = 'lower')
    plt.gray()
    plt.subplot(n_images / 2, 4, 1 + (n_image - 1) * 2 + 1)
    plt.imshow(after, origin = 'lower')
    plt.gray()
    if n_images == n_image:
        plt.show()

def testImage():
    bp = backprop.MultiLayerPerceptron(64 * 64, 4, 64 * 64, 0.08)
    bp.noiser = backprop.SaltPepperNoiser()
    
    c1 = readImage('lfwcrop_grey/faces/Alejandro_Toledo_0003.pgm')
    c2 = readImage('lfwcrop_grey/faces/Arminio_Fraga_0005.pgm')
    c3 = readImage('lfwcrop_grey/faces/Bill_Graham_0008.pgm')
    c4 = readImage('lfwcrop_grey/faces/Costas_Simitis_0006.pgm')
    c5 = readImage('lfwcrop_grey/faces/Dennis_Kucinich_0004.pgm')
    c6 = readImage('lfwcrop_grey/faces/Ernie_Grunfeld_0001.pgm')
    c7 = readImage('lfwcrop_grey/faces/Harry_Schmidt_0001.pgm')
    c8 = readImage('lfwcrop_grey/faces/James_Kelly_0004.pgm')

    examples = numpy.matrix([c1, c2, c3, c4, c5, c6, c7, c8])
    
    errors = bp.train(examples, examples, 400)
        
    index = 1
    for image in (c1, c2, c3, c4, c5, c6, c7, c8):
        image = bp.noiser.addNoise(image)
        showImages(image, bp.evaluateNetwork(image), 8, index)
        index += 1
 
    plt.figure(2)
    plt.plot(range(len(errors)), errors)
    plt.show()
    
testImage()
