from ape.theano_gpu_util import *

def test_cpu_to_gpu_graph():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x,y)
    inputs = (x,y)
    outputs = (z,)
    gpu_inputs, gpu_outputs = cpu_to_gpu_graph(inputs, outputs)
    f = theano.function(gpu_inputs, gpu_outputs)
    assert all(isinstance(inp, theano.sandbox.cuda.CudaNdarrayVariable)
                for inp in (gpu_inputs+gpu_outputs))
