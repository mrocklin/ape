from ape.theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)
from ape.env_manip import (env_with_names, unpack, unpack_many,
        shape_of_variables)
from ape.schedule import gen_code, machine_dict_to_code, is_output

import theano

def test_gen_code_simple():
    x = theano.tensor.matrix('x')
    y = x+x*x
    env = theano.FunctionGraph([x], [y])
    env = env_with_names(env)

    machine_ids = ["ankaa", "mimosa"]
    _test_gen_code(env, machine_ids)

def test_gen_code_complex():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    env = theano.FunctionGraph([x, y], [z])
    env = env_with_names(env)
    machine_ids = ["ankaa", "mimosa"]

    _test_gen_code(env, machine_ids)

def _test_gen_code(env, machine_ids):

    sched = compute_schedule(*make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100))

    shapes = shape_of_variables(env, {var:(5,5) for var in env.inputs})
    shapes = {k.name : v for k,v in shapes.items()}

    d = gen_code(sched, 'envs.dat', shapes)

    env_filename = d['env_filename']
    assert env_filename == 'envs.dat'
    envs = unpack_many(env_filename)
    assert all(type(x) == theano.FunctionGraph for x in envs)

    varinit = d['variable_initialization']
    print [var.name for var in env.variables]
    print varinit
    assert all(var.name in varinit for var in env.variables
            if not is_output(var))

    recv_code = d['recv']
    host_code = d['compute']

def test_machine_dict_to_code():
    d = {'cpu':['x = 1', 'y = 2'], 'gpu':['z = 3']}
    s = machine_dict_to_code(d)
    print s
    assert s == (
"""if host == 'gpu':
    z = 3
if host == 'cpu':
    x = 1
    y = 2""")
def test_empty_machine_dict_to_code():
    d = {'cpu':[]}
    s = machine_dict_to_code(d)
    assert s=="if host == 'cpu':\n    pass"
