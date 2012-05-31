from ape.theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)
from ape.env_manip import env_with_names, unpack, unpack_many
from ape.schedule import gen_code

import theano

def test_gen_code_simple():
    x = theano.tensor.matrix('x')
    y = x+x*x
    env = theano.Env([x], [y])
    env = env_with_names(env)

    machine_ids = ["ankaa", "arroyitos"]
    _test_gen_code(env, machine_ids)

def test_gen_code_complex():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    env = theano.Env([x, y], [z])
    env = env_with_names(env)
    machine_ids = ["ankaa", "arroyitos"]

    _test_gen_code(env, machine_ids)

def _test_gen_code(env, machine_ids):

    sched = compute_schedule(*make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100))

    d = gen_code(sched, 'envs.dat')

    env_filename = d['env_filename']
    assert env_filename == 'envs.dat'
    envs = unpack_many(env_filename)
    assert all(type(x) == theano.Env for x in envs)

    compile_str = d['compile']
    assert len(compile_str.split('\n')) == len(envs)*2 # each env is compiled

    var_tags = eval(d['variable_tags'].split('= ')[1])
    # all variables accounted for
    assert all(var.name in var_tags for var in env.variables)
    assert len(set(var_tags.values())) == len(var_tags.values()) # no repeats

    host_code = d['host_code']
