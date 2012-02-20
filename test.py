from theano_to_milp import make_ilp, compute_schedule

from mul_sum_computation import make_computation
from three_node_system import system, A

computation = make_computation(2)

prob, X, S, Cmax, runtimes, commtimes = make_ilp(computation, system, A, M=10, niter=3)
sched = compute_schedule(prob, X, S, Cmax)
print sched


def test_single_wire():
    an = f.maker.env.outputs[0].owner
    job = TheanoJob(an)
    V = TheanoArrayVariable(job.inputs[0]._variable, (1000,1000))
    w = MPIWire(A, B)
    A.instantiate_random_variable(V)
    A.compile(job)
    A.run(job)
    for output in job.outputs:
        w.transmit(output)

    assert all(output in B for output in job.outputs)

