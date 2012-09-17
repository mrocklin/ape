from theano.tensor.utils import shape_of_variables
from ape.codegen.schedule import gen_code
from ape.theano_to_milp import compute_schedule, make_ilp
from ape import ape_dir
from ape.util import dearrayify
import theano

def compile(env, machine_ids, compute_cost, comm_cost, ability,
        input_shapes, max_makespan):
    """ Master function that takes

    inputs
    ------
    env :: theano.FunctionGraph object - represents computational graph
    machine_ids  :: [machine identifier] (iterable)
    compute_time :: theano.Apply, id     -> time (float)
    comm_time    :: theano.Apply, id, id -> time (float)
    ability      :: theano.Apply, id     -> 1 or 0
    input_shapes :: dict mapping input variable to shape
    max_time     :: time (float)

    returns code to optimally compute env on the given machines
    """
    # Massage input FunctionGraph
    fgraph = fgraph.clone()
    theano.gof.graph.give_variables_names(fgraph.variables)

    # TODO - this should be put somewhere else, perhaps in shape_of_variables?
    # Re-key input-shapes to the new vars in the env with names
    name_to_var = {var.name : var for var in env.inputs}
    input_shapes = {name_to_var[var.name] : input_shapes[var]
                        for var in input_shapes}
    shapes = dearrayify(shape_of_variables(env, input_shapes))
    shapes = {k.name : v for k,v in shapes.items()}

    # Compute schedule using Integer programming
    sched = compute_schedule(*make_ilp(env, machine_ids, compute_cost,
            comm_cost, ability, 100))

    # Turn this schedule into python code
    specifics = gen_code(sched, 'env.dat', shapes)

    # Read code template from template.py
    f = open(ape_dir+'ape/codegen/template.py'); code = f.read(); f.close()

    return code%specifics
