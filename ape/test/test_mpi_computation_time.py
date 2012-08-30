from ape.mpi_computation_time import compute_time_on_machine
from ape.env_manip import fgraph_iter, variables_with_names
import theano

def test_compute_time_on_machine():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x+1, 2*y)
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))
    input_shapes = {x:(1000,1000), y:(1000,1000)}
    niter = 3
    machine = 'ankaa.cs.uchicago.edu'

    times = compute_time_on_machine(fgraph, input_shapes, machine, niter)
    assert isinstance(times, dict)
    assert set(fgraph.nodes) == set(times.keys())
    assert all(isinstance(val, float) for val in times.values())
