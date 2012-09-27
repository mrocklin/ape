# Utility functions for dealing with GPU variables in Theano
from theano.sandbox import cuda
import numpy as np
import theano

math_opt = theano.compile.optdb.query('-inplace', '+fast_run', '-gpu')
gpu_opt  = cuda.opt.gpu_optimizer.query('+gpu', '-inplace', '-async')
gpu_comm = cuda.opt.gpu_cut_copies.query('+gpu')

def cpu_to_gpu_var(x):
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'gpu_%s'%x.name
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var

def gpu_to_cpu_var(x):
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'cpu_%s'%x.name
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
    for go, co in zip(gpu_outputs, outputs2):
        if co.name:
            go.name = "gpu_"+co.name

    fgraph = theano.FunctionGraph(gpu_inputs, gpu_outputs)
    math_opt.optimize(fgraph)
    gpu_opt.optimize(fgraph)
    gpu_comm.optimize(fgraph)
    fgraph.disown()

    return tuple(gpu_inputs), tuple(gpu_outputs)

def togpu_data(x, copy=True):
    """ Sends a cpu variable to the GPU

    Works for numpy.ndarrays

    >>> x = np.ones((5,5), dtype='float32')
    >>> togpu_data(x)
    <CudaNdarray at 0x2ece870>

    See also
    --------
        tocpu_data
    """

    if isinstance(x, np.ndarray):
        return theano.sandbox.cuda.shared_constructor(x).get_value(
                borrow=True, return_internal_type=True)
    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return xg.get_value(return_internal_type=True, borrow=copy)
    assert False

def tocpu_data(x, copy=True):
    """ Sends a GPU variable to the CPU

    Works for theano.sandbox.cuda.CudaNdarrays

    >>> x  = np.ones((5,5), dtype='float32')
    >>> gx = togpu_data(x)
    >>> xx = tocpu_data(gx)
    >>> xx
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]], dtype=float32)

    See also
    --------
        tocpu_data
    """

    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return x.get_value(return_internal_type=False)
    assert False
