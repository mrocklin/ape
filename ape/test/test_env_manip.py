from ape.env_manip import *

x = T.matrix('x')
y = x+x*x
simple_env = theano.Env([x], [y])

x = T.matrix('x')
y = T.matrix('y')
z = T.dot(x, y) * T.sin(x) - y.sum(0)
complex_env = theano.Env([x, y], [z])

def test_pack():
    assert type(pack(simple_env)) is str
    assert str(unpack(pack(simple_env))) == str(simple_env)
    assert str(unpack(pack(complex_env))) == str(complex_env)

def test_pack_file():
    f = open('_temp.dat', 'w')
    pack(simple_env, f)
    pack(complex_env, f)
    f.close()
    f = open('_temp.dat', 'r')
    simple_env2 = unpack(f)
    complex_env2 = unpack(f)

    assert str(simple_env) == str(simple_env2)
    assert str(complex_env) == str(complex_env2)

def test_math_optimize():
    assert isinstance(math_optimize(simple_env), theano.Env)
    assert str(math_optimize(simple_env)) == \
            "[Elemwise{Composite{[add(i0, sqr(i0))]}}(x)]"

def test_shape_of_variable():
    x = T.matrix('x')
    y = x+x
    env = theano.Env([x], [y])
    assert shape_of_variables(env, {x: (5, 5)}) == {x: (5, 5), y: (5, 5)}

    x = T.matrix('x')
    y = T.dot(x, x.T)
    env = theano.Env([x], [y])
    shapes = shape_of_variables(env, {x: (5, 1)})
    assert shapes[x] == (5, 1)
    assert shapes[y] == (5, 5)

def test_precedes():
    x = theano.tensor.matrix('x')
    y = x+x
    z = y*y
    assert precedes(y.owner, z.owner)

def test_env_with_names():
    x = T.matrix('x')
    y = x+x*x
    env = theano.Env([x], [y])
    env = env_with_names(env)
    assert set(("x", "var_0", "var_1", "var_2")).issuperset(
            {var.name for var in env.variables})
