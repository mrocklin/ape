from ape.theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability)

import theano

def test_make_ilp():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    env = theano.Env([x, y], [z])
    machine_ids = ["ankaa", "arroyitos"]

    prob, X, S, Cmax = make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100)

    prob.solve()
    assert Cmax.value() == 13


