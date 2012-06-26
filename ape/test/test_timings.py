from ape.timings import (make_runtime_fn, compute_runtimes,
        make_runtime_function, save_dict, load_dict)
import theano

def test_make_runtime_fn():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    runtime_fn = make_runtime_fn([x,y], [z], {x:(1000,1000), y:(1000,1000)},
            lambda id: True)
    env = theano.Env([x, y], [z])

    assert all(isinstance(runtime_fn(an, 'ankaa'), float)
                                                for an in env.toposort())

def test_make_runtime_function():
    data = {('A', 'B'): {'gemm':1, 'sum':2},
            ('C',)    : {'gemm':3, 'sum':1}}
    fn = make_runtime_function(data)
    assert fn('gemm', 'A') == 1
    assert fn('sum', 'C') == 1

def test_save_dict():
    data = {('A', 'B'): {'gemm':1, 'sum':2},
            ('C',)    : {'gemm':3, 'sum':1}}
    save_dict('_temp.tmp', data)
    data2 = load_dict('_temp.tmp')
    assert data == data2

def test_compute_runtimes():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    times = compute_runtimes([x,y], [z], {x:(1000,1000), y:(1000,1000)}, 10)
    assert isinstance(times, dict)
    assert all(isinstance(key, str)     for key in times)
    assert all(isinstance(val, float)   for val in times.itervalues())
    assert 'dot(x, x)' in times
