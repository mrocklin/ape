from ape.timings.comptime_mpi import compute_time_on_machine
from ape.env_manip import fgraph_iter, variables_with_names
import theano
from ape.timings.comptime_mpi import comptime_dict
from ape.timings.comptime import make_runtime_function

def _test_compute_time_on_machine(machine):
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x+1, 2*y)
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))
    input_shapes = {x:(1000,1000), y:(1000,1000)}
    niter = 3

    times = compute_time_on_machine(fgraph, input_shapes, machine, niter)
    assert isinstance(times, dict)
    assert set(map(str, fgraph.nodes)) == set(times.keys())
    assert all(isinstance(val, float) for val in times.values())

def test_nfs():
    _test_compute_time_on_machine('ankaa.cs.uchicago.edu')

def test_remote():
    _test_compute_time_on_machine('baconost.cs.uchicago.edu')

def test_comptime_dict():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))
    machine_groups = (('ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu'),
                      ('milkweed.cs.uchicago.edu',))

    times = comptime_dict(fgraph, {x:(1000,1000), y:(1000,1000)}, 10,
                                 machine_groups)
    assert isinstance(times, dict)
    assert set(times.keys()) == set(machine_groups)
    # The keys of the subdicts are apply nodes
    assert all(all(key in map(str, fgraph.nodes) for key in d)
               for d in times.values())
    # The values of the subdicts are floats
    assert all(all(isinstance(val, float) for val in d.values())
               for d in times.values())

    runtime = make_runtime_function(times)
    assert all(isinstance(runtime(n, 'ankaa.cs.uchicago.edu'), float)
                for n in fgraph.nodes)

