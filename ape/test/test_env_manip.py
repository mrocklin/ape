from ape.env_manip import *

x = T.matrix('x')
y = x+x*x
simple_env = theano.FunctionGraph([x], [y])

x = T.matrix('x')
y = T.matrix('y')
z = T.dot(x, y) * T.sin(x) - y.sum(0)
complex_env = theano.FunctionGraph([x, y], [z])

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
    f.close()

    assert str(simple_env) == str(simple_env2)
    assert str(complex_env) == str(complex_env2)

def test_pack_many():
    f = open('_temp.dat', 'w')
    pack_many((simple_env, complex_env), f)
    f.close()
    f = open('_temp.dat', 'r')
    (simple_env2, complex_env2) = unpack_many(f)

    assert str(simple_env) == str(simple_env2)
    assert str(complex_env) == str(complex_env2)


def test_math_optimize():
    assert isinstance(math_optimize(simple_env), theano.FunctionGraph)
    assert str(math_optimize(simple_env)) == \
            "[Elemwise{Composite{[add(i0, sqr(i0))]}}(x)]"

def test_shape_of_variables():
    x = T.matrix('x')
    y = x+x
    env = theano.FunctionGraph([x], [y])
    assert shape_of_variables(env, {x: (5, 5)}) == {x: (5, 5), y: (5, 5)}

    x = T.matrix('x')
    y = T.dot(x, x.T)
    env = theano.FunctionGraph([x], [y])
    shapes = shape_of_variables(env, {x: (5, 1)})
    assert shapes[x] == (5, 1)
    assert shapes[y] == (5, 5)

def test_shape_of_subtensor():
    x = theano.tensor.matrix('x')
    subx = x[1:]
    env = theano.FunctionGraph([x], [subx])
    shapes = shape_of_variables(env, {x: (10, 10)})
    assert shapes[subx] == (9, 10)

def test_precedes():
    x = theano.tensor.matrix('x')
    y = x+x
    z = y*y
    assert precedes(y.owner, z.owner)

def test_variables_with_names():
    x = T.matrix('x')
    y = x+x*x
    variables_with_names([x], [y]) # change state
    variables = theano.gof.graph.variables([x],[y])

    assert set(("x", "var_0", "var_1", "var_2")).issuperset(
            {var.name for var in variables})

def test_env_with_names():
    x = T.matrix('x')
    y = x+x*x
    env = theano.FunctionGraph([x], [y])
    env = env_with_names(env)
    assert set(("x", "var_0", "var_1", "var_2")).issuperset(
            {var.name for var in env.variables})
