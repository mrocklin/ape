from theano_computation import TheanoComputation
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




