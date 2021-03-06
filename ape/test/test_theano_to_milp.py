from ape.theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)

import theano

def test_make_ilp():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    env = theano.FunctionGraph([x, y], [z])
    machine_ids = ["ankaa", "arroyitos"]

    prob, X, S, Cmax = make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100)

    prob.solve()
    assert Cmax.value() == 5

def test_compute_schedule():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    env = theano.FunctionGraph([x, y], [z])
    machine_ids = ["ankaa", "arroyitos"]

    sched = compute_schedule(*make_ilp(env, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100))

    # nodes are the jobs
    assert env.apply_nodes == set([job for job, (time, id) in sched])
    times = [time for job, (time, id) in sched]
    # jobs are sorted by time
    assert list(sorted(times)) == times
    # the machine ids match what we put in
    assert all(id in machine_ids for job, (time, id) in sched)

