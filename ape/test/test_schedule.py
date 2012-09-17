from ape.theano_to_milp import (make_ilp, dummy_compute_cost, dummy_comm_cost,
        dummy_ability, compute_schedule)
from ape.env_manip import unpack, unpack_many
from theano.tensor.utils import  shape_of_variables
from ape.schedule import gen_code, machine_dict_to_code, is_output

import theano

def test_gen_code_simple():
    x = theano.tensor.matrix('x')
    y = x+x*x
    fgraph = theano.FunctionGraph([x], [y])
    theano.gof.graph.utils.give_variables_names(fgraph.variables)

    machine_ids = ["ankaa", "mimosa"]
    _test_gen_code(fgraph, machine_ids)

def test_gen_code_complex():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
    fgraph= theano.FunctionGraph([x, y], [z])
    theano.gof.graph.utils.give_variables_names(fgraph.variables)
    machine_ids = ["ankaa", "mimosa"]

    _test_gen_code(fgraph, machine_ids)

def _test_gen_code(fgraph, machine_ids):

    sched = compute_schedule(*make_ilp(fgraph, machine_ids, dummy_compute_cost,
            dummy_comm_cost, dummy_ability, 100))

    shapes = shape_of_variables(fgraph, {var:(5,5) for var in fgraph.inputs})
    shapes = {k.name : v for k,v in shapes.items()}

    d = gen_code(sched, 'envs.dat', shapes)

    fgraph_filename = d['env_filename']
    assert fgraph_filename == 'envs.dat'
    fgraphs = unpack_many(fgraph_filename)
    assert all(type(x) == theano.FunctionGraph for x in fgraphs)

    varinit = d['variable_initialization']
    print [var.name for var in fgraph.variables]
    print varinit
    assert all(var.name in varinit for var in fgraph.variables
            if not is_output(var))

    recv_code = d['recv']
    host_code = d['compute']

def test_machine_dict_to_code():
    d = {'cpu':['x = 1', 'y = 2'], 'gpu':['z = 3']}
    s = machine_dict_to_code(d)
    print s
    assert s == (
"""if host == 'gpu':
    z = 3
if host == 'cpu':
    x = 1
    y = 2""")
def test_empty_machine_dict_to_code():
    d = {'cpu':[]}
    s = machine_dict_to_code(d)
    assert s=="if host == 'cpu':\n    pass"
