# Utility functions for dealing with GPU variables in Theano
from theano.sandbox import cuda
import theano

def cpu_to_gpu_var(x):
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'gpu_%s'%x.name
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var

def cpu_to_gpu_graph(inputs, outputs):
    """ Converts a cpu-only subgraph into a gpu-only subgraph

    >>> x, y = theano.tensor.matrix('x'), theano.tensor.matrix('y')
    >>> z = theano.tensor.dot(x, y)
    >>> gpu_inputs, gpu_outputs = cpu_to_gpu_graph((x,y), (z,))
    >>> f = theano.function(gpu_inputs, gpu_outputs)
    >>> theano.printing.debugprint(f)
    GpuDot22 [@A] ''   0
     |gpu_x [@B]
     |gpu_y [@C]
    """

    gpu_inputs, cpu_inputs = zip(*map(cpu_to_gpu_var, inputs))
    outputs2 = theano.clone(outputs, replace=dict(zip(inputs, cpu_inputs)))
    gpu_outputs = map(theano.sandbox.cuda.basic_ops.gpu_from_host, outputs2)
    final_outputs = map(lambda o: theano.Out(o, borrow=True), gpu_outputs)

    return tuple(gpu_inputs), tuple(gpu_outputs)
