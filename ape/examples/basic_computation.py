import theano
a = theano.tensor.fmatrix('a')
b = theano.tensor.fmatrix('b')

c = a + b; c.name = 'c'
d = theano.tensor.dot(a, a); d.name = 'd'

e = c + d; e.name = 'e'

inputs = (a,b)
outputs = (e,)

theano.gof.graph.utils.give_variables_names(
        theano.gof.graph.variables(inputs, outputs))

n = 2
input_shapes = {a: (n, n),
                b: (n, n)}

# a -- dot -- d \
#    \           + -- e
# b --  +  -- c /
