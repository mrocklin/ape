from ape.theano_gpu_util import *
import numpy as np
import theano

def test_cpu_to_gpu_graph():
    if theano.config.device != 'gpu':
        return
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x,y)
    inputs = (x,y)
    outputs = (z,)
    gpu_inputs, gpu_outputs = cpu_to_gpu_graph(inputs, outputs)
    f = theano.function(gpu_inputs, gpu_outputs)
    assert all(isinstance(inp, theano.sandbox.cuda.CudaNdarrayVariable)
                for inp in (gpu_inputs+gpu_outputs))

def test_togpu_tocpu_data():
    if theano.config.device != 'gpu':
        return
    x = np.ones((5,5), dtype='float32')
    gx = togpu_data(x)
    assert isinstance(gx, theano.sandbox.cuda.CudaNdarray)
    xx = tocpu_data(gx)
    assert (x==xx).all()


def test_cpu_to_gpu_var():
    if theano.config.device != 'gpu':
        return
    x = theano.tensor.matrix('x')
    gx, cx = cpu_to_gpu_var(x)
    f = theano.function((gx,), cx)
    xx = np.ones((5,5), dtype='float32')
    gxx = togpu_data(xx)
    assert (f(gxx) == xx).all()
