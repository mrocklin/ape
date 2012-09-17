from ape.timings.computation.run import comptime_run
import theano
from ape.util import dearrayify
from theano.tensor.utils import shape_of_variables
from ape.env_manip import clean_variables, fgraph_iter

def test_comptime_run():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, y)
    theano.gof.utils.give_variables_names((x,y), (z,))
    clean_variables((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))

    input_shapes = {x: (10, 10), y: (10, 10)}
    known_shapes = dearrayify(shape_of_variables(fgraph, input_shapes))
    known_shapes = {str(k): v for k,v in known_shapes.items()}

    time_comp_fn = lambda ins, outs, num_ins, niter: 1

    fgraphs = list(fgraph_iter(fgraph))
    niter = 3

    results = comptime_run(known_shapes, niter, fgraphs, time_comp_fn)
    assert results == [1]*len(fgraphs)

