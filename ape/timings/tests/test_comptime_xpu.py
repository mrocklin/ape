import theano
from ape.env_manip import fgraph_iter, variables_with_names
from ape.timings.comptime import make_runtime_function

def _test_comptime_dict_xpu(machines, machine_groups, comptime_dict_fn):
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    variables_with_names((x,y), (z,))
    fgraph = theano.FunctionGraph((x,y), (z,))

    times = comptime_dict_fn(fgraph, {x:(1000,1000), y:(1000,1000)}, 10,
                              machines, machine_groups)
    assert isinstance(times, dict)
    assert set(times.keys()) == set(machine_groups)
    # The keys of the subdicts are apply nodes
    assert all(all(key in map(str, fgraph.nodes) for key in d)
               for d in times.values())
    # The values of the subdicts are floats
    assert all(all(isinstance(val, float) for val in d.values())
               for d in times.values())

    runtime = make_runtime_function(times)
    assert all(isinstance(runtime(n, machine), float)
                for n in fgraph.nodes
                for machine in machines)