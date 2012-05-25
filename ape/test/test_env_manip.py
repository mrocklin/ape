from ape.env_manip import *

def test_simple_env():
    x = T.matrix('x')
    y = x+x*x
    env = theano.Env([x], [y])

    assert type(pack(env)) is str
    assert isinstance(math_optimize(env), theano.Env)
    assert str(unpack(pack(env))) == str(env)
    assert str(math_optimize(env)) == \
            "[Elemwise{Composite{[add(i0, sqr(i0))]}}(x)]"

def test_complex_env():
    x = T.matrix('x')
    y = T.matrix('y')

    z = T.dot(x, y) * T.sin(x) - y.sum(0)
    env = theano.Env([x, y], [z])

    assert type(pack(env)) is str
    assert isinstance(math_optimize(env), theano.Env)
    assert str(unpack(pack(env))) == str(env)
