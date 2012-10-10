from ape.theano_gpu_util import *
from theano.sandbox.cuda import GpuOp, HostFromGpu, GpuFromHost
import numpy as np
import theano

def test_cpu_to_gpu_graph():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x,y); z.name = 'z'
    _test_cpu_to_gpu_graph((x, y), (z,))

def test_cpu_to_gpu_graph2():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = x + y; z.name = 'z'
    _test_cpu_to_gpu_graph((x, y), (z,))

def _test_cpu_to_gpu_graph(i, o):
    if theano.config.device != 'gpu':
        return
    gi, go = cpu_to_gpu_graph(i, o)

    # Everything is a CudaNdarrayVariable
    assert all(isinstance(inp, theano.sandbox.cuda.CudaNdarrayVariable)
                for inp in (gi+go))
    # Everything is a GpuOp
    assert all(isinstance(n.op, GpuOp)
            for n in theano.gof.graph.list_of_nodes(gi, go))
    # And we didn't need any communication
    assert not any(isinstance(n.op, (HostFromGpu, GpuFromHost))
            for n in theano.gof.graph.list_of_nodes(gi, go))

    cpu_names = [v.name for v in i + o]
    gpu_names = [v.name for v in gi+go]
    assert gpu_names == map(lambda n: "gpu_"+n, cpu_names)


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
