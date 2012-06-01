import theano
from theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)
from env_manip import env_with_names, unpack, unpack_many, shape_of_variables
from schedule import gen_code

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
env = theano.Env([x, y], [z])
env = env_with_names(env)
shapes = shape_of_variables(env, {var:(5,5) for var in env.inputs})
shapes = {k.name : v for k,v in shapes.items()}

machine_ids = ["ankaa.cs.uchicago.edu", "mimosa.cs.uchicago.edu"]

sched = compute_schedule(*make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100))

d = gen_code(sched, 'env.dat', shapes)
f = open('template.py'); s = f.read(); f.close()

ss = s%d

f = open('results.py', 'w'); f.write(ss); f.close()

