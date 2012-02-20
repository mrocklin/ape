from theano_to_milp import make_ilp
from theano_to_milp import go_schedule as tompkins_schedule
from heft import theano_heft_schedule as heft_schedule

from mul_sum_computation import make_computation
from three_node_system import system, A

computation = make_computation(2)

sched_milp = tompkins_schedule(computation, system, A, M=1)
sched_heft = heft_schedule(computation, system, A)
print sched_milp
print [(x.worker, x.task_list) for x in sched_heft]
