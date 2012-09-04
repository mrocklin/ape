from ape.timings.comptime_run import time_computation
import theano
import numpy as np

def test_time_computaiton():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, y)

    xx = np.ones((100, 100), dtype=x.dtype)
    yy = np.ones((100, 100), dtype=y.dtype)

    time = time_computation((x, y), (z,), (xx, yy), 5)
    assert isinstance(time, float)
