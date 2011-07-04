'''
Created on Jul 4, 2011

@author: alleveenstra
'''

import pyopencl as cl
import numpy
import numpy.linalg as la

a = numpy.random.rand(50000).astype(numpy.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = a)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

prg = cl.Program(ctx, """
    float something();

    __kernel void sum(__global const float *a, __global float *dest)
    {
      int gid = get_global_id(0);
      dest[gid] = something();
    }
    
    float something() {
      return 42;
    }
    """).build()

prg.sum(queue, a.shape, None, a_buf, dest_buf)
print a.shape

dest = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, dest_buf, dest).wait()

print dest
