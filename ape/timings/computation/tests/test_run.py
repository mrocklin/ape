from ape.timings.computation.run import comptime_run
import theano
from ape.util import dearrayify
from ape.theano_util import shape_of_variables
from ape.timings.util import graph_iter
from ape.env_manip import clean_variable

def test_comptime_run():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, y)
    inputs, outputs = (x, y), (z,)
    variables = theano.gof.graph.variables(inputs, outputs)

    nodes = theano.gof.graph.list_of_nodes(inputs, outputs)

    theano.gof.utils.give_variables_names(variables)
    map(clean_variable, variables)

    input_shapes = {x: (10, 10), y: (10, 10)}
    known_shapes = shape_of_variables(inputs, outputs, input_shapes)
    known_shapes = {str(k): v for k,v in known_shapes.items()}

    time_comp_fn = lambda ins, outs, num_ins, niter: 1

    fgraphs = list(graph_iter(nodes))
    niter = 3

    results = comptime_run(known_shapes, niter, fgraphs, time_comp_fn)
    assert results == [1]*len(fgraphs)

