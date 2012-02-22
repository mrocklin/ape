"""
Do the following
cd ~/workspace/ape
ipcontroller --profile=mpi # in one terminal
mpiexec --np 3 ipengine --profile=mpi # in another terminal
python test.py # in a third terminal
"""

from theano_to_milp import make_ilp
from theano_to_milp import go_schedule as tompkins_schedule
#from heft import theano_heft_schedule as heft_schedule
from heft import schedule as heft_schedule
from timings import make_runtime_fn, make_commtime_fn, make_commtime_fn_tompkins

from mul_sum_computation import make_computation
from three_node_system import system, A

computation = make_computation(2, (10, 10))

runtime = make_runtime_fn(computation, system)
commtime = make_commtime_fn(computation, system)

sched_heft = heft_schedule(computation.jobs, system.machines,
                computation.start_jobs, computation.end_jobs,
                runtime, commtime, cache=False)

commtime_tompkins = make_commtime_fn_tompkins(computation, system)
sched_milp = tompkins_schedule(computation, system, A, M=1,
        runtime=runtime, commtime=commtime_tompkins)


#print sched_milp
print [(x.worker, x.task_list) for x in sched_heft]
