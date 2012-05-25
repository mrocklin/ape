from env_manip import precedes
from collections import defaultdict
from tompkins import schedule
import pulp

def dummy_compute_cost(an, id):
    return 1
def dummy_comm_cost(an, from_id, to_id):
    return 2

def dummy_ability(an, machine):
    if an == None: # input node!
        pass # we'll probably want special behavior here
    return 1

def make_ilp(env, machine_ids, compute_time, comm_time, ability, max_time):
    jobs = list(env.nodes)

    P = {(a, b): precedes(a, b) for a in jobs for b in jobs}
    R = {job : 0 for job in jobs}
    D = {(job, id) : compute_time(job, id) for job in jobs
                                           for id in machine_ids}
    C = {(job, id1, id2) : comm_time(job, id1, id2)
                    for job in jobs
                    for id1 in machine_ids
                    for id2 in machine_ids}
    B = {(job, id) : ability(job, id) for job in jobs
                                      for id in machine_ids}
    prob, X, S, Cmax = schedule(jobs, machine_ids, D, C, R, B, P, max_time)
    prob.solver = pulp.LpSolverDefault
    prob.solver.maxSeconds = max_time

    return prob, X, S, Cmax
