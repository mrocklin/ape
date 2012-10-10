from ape.timings.computation.mpi import _compute_time_on_machine
from ape.env_manip import clean_variable
import theano

def _test_compute_time_on_machine(machine):
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x+1, 2*y)
    i, o = (x, y), (z,)

    apply_nodes = theano.gof.graph.list_of_nodes(i, o)
    variables = theano.gof.graph.variables(i, o)

    theano.gof.utils.give_variables_names(variables)
    map(clean_variable, variables)

    input_shapes = {x:(1000,1000), y:(1000,1000)}
    niter = 3

    times = _compute_time_on_machine('ape/timings/computation/run_cpu.py',
                                     i, o, input_shapes, machine, niter)
    assert isinstance(times, dict)
    assert set(map(str, apply_nodes)) == set(times.keys())
    assert all(isinstance(val, float) for val in times.values())

def test_nfs():
    _test_compute_time_on_machine('ankaa.cs.uchicago.edu')

def test_remote():
    _test_compute_time_on_machine('baconost.cs.uchicago.edu')
