from env_manip import precedes
from collections import defaultdict
from tompkins.ilp import schedule
import pulp

def dummy_compute_cost(an, id):
    return 1
def dummy_comm_cost(an, from_id, to_id):
    return 0 if from_id == to_id else 2

def dummy_ability(an, machine):
    if an == None: # input node!
        pass # we'll probably want special behavior here
    return 1

def make_ilp(env, machine_ids, compute_time, comm_time, ability, max_time):
    """
    Take FunctionGraph, a list of machines, and functions to tell us compute and
    communication times.

    inputs
    ------
    env :: theano.FunctionGraph object - represents computational graph
    machine_ids  :: [machine identifier] (iterable)
    compute_time :: theano.Apply, id     -> time (float)
    comm_time    :: theano.Apply, id, id -> time (float)
    ability      :: theano.Apply, id     -> 1 or 0
    max_time     :: time (float)

    Returns a pulp integer linear program.

    See Also:
        compute_schedule

    """

    jobs = list(env.nodes)

    P = {(job1, job2): precedes(job1, job2) for job1 in jobs for job2 in jobs}

    R = { job : 0                           for job in jobs}

    D = {(job, id) : compute_time(job, id)  for job in jobs
                                            for id in machine_ids}

    C = {(job, id1, id2) : comm_time(job, id1, id2)
                                            for job in jobs
                                            for id1 in machine_ids
                                            for id2 in machine_ids}

    B = {(job, id) : ability(job, id)       for job in jobs
                                            for id in machine_ids}

    prob, X, S, Cmax = schedule(jobs, machine_ids, D, C, R, B, P, max_time)
    prob.solver = pulp.LpSolverDefault
    prob.solver.maxSeconds = max_time

    return prob, X, S, Cmax

def compute_schedule(prob, X, S, Cmax):
    """
    Take the outputs of make_ilp and produce a list of the form
    [job, (start_time, machine_id),
     job, (start_time, machine_id),
     ... ]

    sorted by start_time.

    >>> sched = compute_schedule(*make_ilp(env, ... ))
    """
    prob.solve()

    def runs_on(job, X):
        return [k for k,v in X[job].items() if v.value()==1][0]

    sched = [(job,(time.value(), runs_on(job,X))) for job, time in S.items()]
    return list(sorted(sched, key=lambda x:x[1]))
