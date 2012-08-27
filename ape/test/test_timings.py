from ape.timings import (make_runtime_fn, compute_runtimes,
        make_runtime_function, make_commtime_function)
import theano

def test_make_runtime_fn():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    runtime_fn = make_runtime_fn([x,y], [z], {x:(1000,1000), y:(1000,1000)},
            lambda id: True)
    env = theano.FunctionGraph([x, y], [z])

    assert all(isinstance(runtime_fn(an, 'ankaa'), float)
                                                for an in env.toposort())

def test_make_runtime_function():
    data = {('A', 'B'): {'gemm':1, 'sum':2},
            ('C',)    : {'gemm':3, 'sum':1}}
    fn = make_runtime_function(data)
    assert fn('gemm', 'A') == 1
    assert fn('sum', 'C') == 1

def test_compute_runtimes():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    times = compute_runtimes([x,y], [z], {x:(1000,1000), y:(1000,1000)}, 10)
    assert isinstance(times, dict)
    assert all(isinstance(key, str)     for key in times)
    assert all(isinstance(val, float)   for val in times.itervalues())
    assert 'dot(x, x)' in times

def test_make_commtime_function():
    data = {('a','b'): (1, 1) , ('b','a'): (0, 10)}

    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y

    env = theano.FunctionGraph([x,y], [z])

    dot = env.toposort()[2]

    from theano.tensor.utils import shape_of_variables
    known_shapes = shape_of_variables(env, {x:(100,100), y:(100,100)})

    commtime = make_commtime_function(data, known_shapes)

    assert commtime(dot, 'a', 'b') == 4*100*100 * 1  + 1
    assert commtime(dot, 'b', 'a') == 4*100*100 * 10 + 0





