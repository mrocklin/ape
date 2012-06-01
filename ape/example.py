import theano
from theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)
from master import compile

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
env = theano.Env([x, y], [z])

input_shapes ={x:(5,5), y:(5,5)}
machine_ids = ["ankaa.cs.uchicago.edu", "mimosa.cs.uchicago.edu"]

ss = compile(env, machine_ids, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, input_shapes, 100)

f = open('results.py', 'w'); f.write(ss); f.close()

