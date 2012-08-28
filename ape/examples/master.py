from theano.tensor.utils import shape_of_variables
from theano.gof.fg import FunctionGraph as FunctionGraph
from ape.timings import compute_runtimes, compute_commtimes
from ape.env_manip import variables_with_names, math_optimize
from ape.util import save_dict, load_dict
from ape.timings import (make_runtime_function, make_commtime_function)
from ape.theano_to_milp import dummy_ability
from ape.master import compile

from kalman import inputs, outputs, input_shapes, machines

variables_with_names(inputs, outputs) # give identifiers to all variables
env = FunctionGraph(inputs, outputs)
env2 = math_optimize(env)
env2_var_dict = {str(var): var for var in env.variables}
input_shapes2 = {env2_var_dict[str(var)]:input_shapes[var]
                 for var in input_shapes}


all_shapes = shape_of_variables(env, input_shapes)

# Compute Cost
compute_times = compute_runtimes(inputs, outputs, input_shapes)
compute_times = {machines: compute_times}
save_dict('compute_times.dat', compute_times)
# compute_times = load_dict('compute_times.dat')
compute_cost = make_runtime_function(compute_times)

# Communication Cost
comm_dict = compute_commtimes([10,100,1000,2000]*5,
            set(machines))
save_dict('comm_times.dat', comm_dict)
# comm_dict = load_dict('comm_times.dat')
comm_cost = make_commtime_function(comm_dict, all_shapes)


code = compile(env, machines, compute_cost, comm_cost,
        dummy_ability, input_shapes, 100)

f = open('results.py', 'w'); f.write(code); f.close()

