from theano.tensor.utils import shape_of_variables
from theano.gof.fg import FunctionGraph as FunctionGraph
from ape.timings.comptime import comptime_dict
from ape.timings.commtime import commtime_dict
from ape.env_manip import variables_with_names, math_optimize
from ape.util import save_dict, load_dict
from ape.timings.comptime import make_runtime_function
from ape.timings.commtime import make_commtime_function
from ape.theano_to_milp import dummy_ability
from ape.master import compile

from nfs_triple import machine_groups, network, machines
from kalman import inputs, outputs, input_shapes

variables_with_names(inputs, outputs) # give identifiers to all variables
fgraph = FunctionGraph(inputs, outputs)
fgraph2 = math_optimize(fgraph)
fgraph2_var_dict = {str(var): var for var in fgraph.variables}
input_shapes2 = {fgraph2_var_dict[str(var)]:input_shapes[var]
                 for var in input_shapes}


all_shapes = shape_of_variables(fgraph, input_shapes)

# Compute Cost
compute_times = comptime_dict(fgraph, input_shapes, 10, machines,
                              machine_groups)
save_dict('compute_times.dat', compute_times)
# compute_times = load_dict('compute_times.dat')
compute_cost = make_runtime_function(compute_times)

# Communication Cost
comm_dict = commtime_dict(network, [10,100,1000,2000]*5)

save_dict('comm_times.dat', comm_dict)
# comm_dict = load_dict('comm_times.dat')
comm_cost = make_commtime_function(comm_dict, all_shapes)


code = compile(fgraph, machines, compute_cost, comm_cost,
        dummy_ability, input_shapes, 100)

f = open('results.py', 'w'); f.write(code); f.close()

