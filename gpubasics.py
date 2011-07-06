'''
Created on Jul 4, 2011

@author: alleveenstra
'''

import pyopencl as cl
import numpy
import numpy.linalg as la
import backprop
import matplotlib.pyplot as plt
import pylab

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
    __kernel void evaluate( int inputSize,
                            int outputSize, 
                            __global const float *input, 
                            __global const float *weight,
                            __global const float *bias,
                            __global float       *output)
    {
      int gid = get_global_id(0);
      int i;
      float sigma = 0;
      for (i = 0; i < inputSize; i++) {
        sigma = sigma + weight[gid + outputSize * i] * input[i];
      }
      output[gid] = tanh(sigma + bias[gid]);
    }
    """).build()

class OpenCLNetworkEvaluator(backprop.DefaultNetworkEvaluator):

    def __init__(self, perceptron):
        self.perceptron = perceptron
    
    def evaluateNetwork(self, inputs):
        perceptron = self.perceptron
        
        self.evaluateOnOpenCL(perceptron.inputActivation, perceptron.hiddenWeight, perceptron.hiddenBias, perceptron.hiddenActivation)
        self.evaluateOnOpenCL(perceptron.hiddenActivation, perceptron.outputWeight, perceptron.outputBias, perceptron.outputActivation)
        return perceptron.outputActivation

    def evaluateOnOpenCL(self, input, weight,  bias, output):
        input_size = numpy.int32(input.size)
        output_size = numpy.int32(output.size)
        
        input_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = input)
        weight_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = weight)
        bias_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = bias)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
        prg.evaluate(queue, output.transpose().shape, None, input_size, output_size, input_buf, weight_buf, bias_buf, output_buf)
        cl.enqueue_read_buffer(queue, output_buf, output).wait()

def testSimpleXor():
    bp = backprop.MultiLayerPerceptron(2, 4, 1, 0.08, 0, 0)
    bp.evaluationFunction = OpenCLNetworkEvaluator(bp)
    
    examples = numpy.matrix([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(numpy.float32)
    classes = numpy.matrix([ [1], [-1], [-1], [1] ]).astype(numpy.float32)
    errors = bp.train(examples, classes, 400)
    
    print '[0,0] -> %.2f' % bp.evaluateNetwork([0, 0])[0, 0]
    print '[1,0] -> %.2f' % bp.evaluateNetwork([1, 0])[0, 0]
    print '[0,1] -> %.2f' % bp.evaluateNetwork([0, 1])[0, 0]
    print '[1,1] -> %.2f' % bp.evaluateNetwork([1, 1])[0, 0]
    
    plt.plot(range(len(errors)), errors)
    plt.show()
    
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
    bp.evaluationFunction = OpenCLNetworkEvaluator(bp)
    
    #print bp.hiddenWeight[0:2, :]
    #print "--------"

    c1e1 = readImage('class1_example1.pgm')
    c1e2 = readImage('class1_example2.pgm')
    c2e1 = readImage('class2_example1.pgm')
    c2e2 = readImage('class2_example2.pgm')
    c3e1 = readImage('class3_example1.pgm')
    c3e2 = readImage('class3_example2.pgm')
    c4e1 = readImage('class4_example1.pgm')
    c4e2 = readImage('class4_example2.pgm')
    examples = numpy.matrix([c1e1, c1e2, c2e1, c2e2, c3e1, c3e2, c4e1, c4e2])
    
    errors = bp.train(examples, examples, 100)
        
    index = 1
    for image in (c1e1, c1e2, c2e1, c2e2, c3e1, c3e2, c4e1, c4e2):
        image = bp.noiser.addNoise(image)
        showImages(image, bp.evaluateNetwork(image), 8, index)
        index += 1
 
    plt.figure(2)
    plt.plot(range(len(errors)), errors)
    plt.show()
    
#testSimpleXor()
testImage()
