
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

def test_computation_local(schedule):
    import numpy as np
    from mpi4py import MPI
    d = locals()

    jobs = schedule.system.jobs
    inputs = schedule.computation.inputs

    # Compile functions locally
    for job in jobs:
        fn = job.function(gpu=False)
        d[A.local_name(job)] = fn

    # Push inputs into namespace
    for var in computation.inputs:
        d[A.local_name(var)] = np.ones(var.shape).astype(var.dtype)





