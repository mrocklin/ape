from collections import defaultdict
import theano
import numpy as np
import theano.tensor as T
#from Job import *
from util import set_union
from tompkins import schedule
from pulp import value


tdp = theano.printing.debugprint
fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_cpu_only = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude=['gpu']))
cpu_mode = theano.Mode(optimizer=fast_run_cpu_only, linker='py')

def name(x):
    if x in name._names:
        return name._names[x]
    s = '%s_%d'%(str(x), name._count)
    name._count += 1
    name._names[x] = s
    return s
name._names = {}
name._count = 0


def all_applys(outputs):
    applies = set()
    variables = list(outputs)
    for v in variables:
        if v.owner and v.owner not in applies:
            applies.add(v.owner)
            if v.owner.inputs:
                for input in v.owner.inputs:
                    if input not in variables:
                        variables.append(input)
    return applies

def intermediate_shapes(inputs, outputs, shapes):
    numeric_inputs = [np.ones(shape).astype(np.float32) for shape in shapes]

    apply_nodes = all_applys(outputs)

    intermediate_inputs = [i for an in apply_nodes for i in an.inputs]

    shapes = theano.function(inputs,
            [var.shape for var in intermediate_inputs+outputs])

    iinput_shape_dict = dict(zip(intermediate_inputs+outputs,
                                 shapes(*numeric_inputs)))
    return iinput_shape_dict

def compute_runtimes(computation, system, niter=5):
    """
    Compute runtimes of a computation on a system

    Returns a dict mapping (job, machine) -> runtime of job on machine
    """
    jobs = computation.jobs
    machines = system.machines
    N = system.comm

    runtimes = {} # Maps job, machine -> runtime
    for machine in machines:
        for job in jobs:
            runtimes[name(job), machine] = machine.predict_runtime(job, niter)

    return runtimes

def compute_commtimes2(computation, system, **kwargs):
    """
    Compute communication times - inactive

    Returns Dict mapping (from_job, to_job,worker, worker) to communication time
    This function takes into accoount some jobs may require an incomplete set
    of variables from another

    This function is more correct but not what is required for the tompkins
    algorithm
    """
    jobs = computation.jobs
    machines = system.machines
    N = system.comm

    commtimes = defaultdict(lambda : 1e10) # Maps job1,job2,m1,m2 -> comm time
    for job in jobs:
        for child in job.children:
            variables = set(job.outputs).intersection(child.inputs)
            for A in machines:
                for B in machines:
                    total_time = 0
                    for V in variables:
                        total_time += N.predict_transfer_time(A,B,V, **kwargs)
                    commtimes[job,child,A,B] = total_time
    return commtimes

def compute_commtimes(computation, system, **kwargs):
    """
    Compute communication times

    Returns Dict mapping (job, worker, worker) to communication time
    """
    jobs = computation.jobs
    machines = system.machines
    N = system.comm

    commtimes = defaultdict(lambda : 1e10) # Maps job1,job2,m1,m2 -> comm time
    for job in jobs:
        variables = job.outputs
        for A in machines:
            for B in machines:
                total_time = 0
                for V in variables:
                    total_time += N.predict_transfer_time(A,B,V, **kwargs)
                commtimes[name(job),A,B] = total_time

    return commtimes

def B_machine_ability(computation, system, startmachine=None, endmachine=None):
    if not endmachine:
        endmachine = startmachine
    B = {}
    for job in computation.jobs:
        for machine in system.machines:
            if machine.can_compute(job):
                B[name(job), machine] = 1
            else:
                B[name(job), machine] = 0
    if startmachine:
        for job in computation.start_jobs:
            for machine in system.machines:
                if machine == startmachine:
                    B[name(job), machine] = 1
                else:
                    B[name(job), machine] = 0

    if endmachine:
        for job in computation.end_jobs:
            for machine in system.machines:
                if machine == endmachine:
                    B[name(job), machine] = 1
                else:
                    B[name(job), machine] = 0

    return B

value = lambda z:z.value()
def make_ilp(computation, system, startmachine, **kwargs):
    machines = system.machines
    network = system.comm

    Jobs = map(name, computation.jobs)
    Agents = machines
    PP = computation.precedence()
    P = defaultdict(lambda:0)
    for a,b in PP:
        P[name(a), name(b)] = PP[a,b]
    B = B_machine_ability(computation, system, startmachine)
    runtimes  = compute_runtimes( computation, system, **kwargs)
    commtimes = compute_commtimes(computation, system, **kwargs)
    D = runtimes
    C = commtimes
    R = defaultdict(lambda:0)
    M = 100 # TODO
    prob, X, S, Cmax = schedule(Jobs, Agents, D, C, R, B, P, M)

    return prob, X, S, Cmax

def compute_schedule(prob, X, S, Cmax):
    prob.solve()

    print "Makespan: ", value(Cmax)
    def runs_on(job, X):
        return [k for k,v in X[job].items() if value(v)==1][0]

    sched = [(job,(value(time), runs_on(job,X))) for job, time in S.items()]
    sched.sort(key=lambda x:x[1])
    return sched



