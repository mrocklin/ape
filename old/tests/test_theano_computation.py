from theano_computation import TheanoComputation, give_names_to_function
import theano
import theano.tensor as T

def test_theano_computation():
    x = T.matrix('x')
    y = T.dot(x, x.T); y.name = 'y'
    f = theano.function([x], y)
    tc = TheanoComputation(f, [(20,10)])
    assert tc.known_shapes['x'] == (20, 10)
    assert tc.known_shapes['y'] == (20, 20)
    assert tc.f is f

def test_give_names_to_function():
    x = T.matrix('x')
    y = T.dot(x, x.T)
    f = theano.function([x], y)
    give_names_to_function(f)
    assert all(f.maker.env.variables)
