from ape.timings.comptime_run import time_computation, comptime_run
import theano
import numpy as np
from ape.util import dearrayify
from theano.tensor.utils import shape_of_variables
from ape.env_manip import variables_with_names, fgraph_iter

def test_time_computaiton():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, y)

    xx = np.ones((100, 100), dtype=x.dtype)
    yy = np.ones((100, 100), dtype=y.dtype)

    time = time_computation((x, y), (z,), (xx, yy), 5)
    assert isinstance(time, float)

def test_comptime_run():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, y)
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))

    input_shapes = {x: (10, 10), y: (10, 10)}
    known_shapes = dearrayify(shape_of_variables(fgraph, input_shapes))
    known_shapes = {str(k): v for k,v in known_shapes.items()}

    time_comp_fn = lambda ins, outs, num_ins, niter: 1

    fgraphs = list(fgraph_iter(fgraph))
    niter = 3

    results = comptime_run(known_shapes, niter, fgraphs, time_comp_fn)
    assert results == [1]*len(fgraphs)

