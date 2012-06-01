import theano
from theano import tensor as T
from computation_graph import *
x = T.matrix('x')
y = T.matrix('y')
z = (T.dot(x,y)+y)/x.sum()
f = theano.function([x,y], z)
an = f.maker.outputs[0].variable.owner
j = TheanoJob(an)
G = j.to_network()

