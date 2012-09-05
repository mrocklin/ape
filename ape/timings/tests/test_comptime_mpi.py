from ape.timings.comptime_mpi import _compute_time_on_machine
from ape.env_manip import variables_with_names
import theano

def _test_compute_time_on_machine(machine):
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x+1, 2*y)
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))
    input_shapes = {x:(1000,1000), y:(1000,1000)}
    niter = 3

    times = _compute_time_on_machine('ape/timings/comptime_run_cpu.py',
                                     fgraph, input_shapes, machine, niter)
    assert isinstance(times, dict)
    assert set(map(str, fgraph.nodes)) == set(times.keys())
    assert all(isinstance(val, float) for val in times.values())

def test_nfs():
    _test_compute_time_on_machine('ankaa.cs.uchicago.edu')

def test_remote():
    _test_compute_time_on_machine('baconost.cs.uchicago.edu')
