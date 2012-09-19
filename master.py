from ape.examples.kalman import *
from ape.examples.nfs_triple import machines, machine_groups, network
import ape

fgraph = theano.FunctionGraph(inputs, outputs)
# theano.gof.graph.utils.give_variables_names(fgraph.variables)
map(ape.env_manip.clean_variable, fgraph.variables)

# Timings
from ape import timings
from theano.tensor.utils import shape_of_variables
from ape.util import save_dict, load_dict
recompute = False
if recompute:
    comps = timings.comptime_dict(fgraph, input_shapes, 5, machines,
            machine_groups)
    comms = timings.commtime_dict(network)
    save_dict('tmp/comps.dat', comps)
    save_dict('tmp/comms.dat', comms)
else:
    comps = load_dict('tmp/comps.dat')
    comms = load_dict('tmp/comms.dat')

comptime = timings.make_runtime_function(comps)
known_shapes = shape_of_variables(fgraph, input_shapes)
known_shape_strings = {str(k): v for k, v in known_shapes.items()}
commtime = timings.make_commtime_function(comms, known_shapes)

# DicDag conversion
import dicdag

dag, dinputs, doutputs = dicdag.theano.fgraph_to_dag(fgraph)
unidag = dicdag.unidag.dag_to_unidag(dag)

def makeapply(inputs, op, output):
    inputs = map(lambda x: x.clone(), inputs)
    outputs = (output.clone(), )
    return theano.Apply(op, inputs, outputs)

def dag_commtime(job, a, b):
    inputs, op, output = job
    return commtime(output, a, b)
    # return sum(commtime(output, a, b) for output in outputs)
def dag_comptime(job, a):
    if job==dicdag.index:
        return 0
    return comptime(makeapply(*job), a)

# Tompkins
import tompkins
dags, sched, makespan = tompkins.schedule(
        unidag, machines, dag_comptime, dag_commtime,
        lambda j:0, lambda j,a:1, 10)

def intersection(a, b):
    return set(a).intersection(set(b))

def replace_send_recvs(dag):
    return tompkins.dag.replace_send_recv(dag,
        lambda A, B, (a,b,fout), c : ((fout,), ("send", B), ("t_"+fout.name,)),
        lambda A, B, (a,b,fout), c : ((), ("recv", A), fout))

cleaner_dags = {machine: replace_send_recvs(dag)
                    for machine, dag in dags.items()}

rankfile = {machine: i for i, machine in enumerate(dags)}
tagfile  = {var: i for i, var in enumerate(map(str, fgraph.variables))}

# TODO: Sends to MPI
# TODO needs unit test
def ith_output(fn, inputs, idx, old_var):
    from tompkins.dag import issend, isrecv
    if issend(fn):
        assert len(inputs) == 1 and idx == 0
        _, machine = fn
        var = theano.tensor.io.send(inputs[0],
                                    rankfile[machine],
                                    tagfile[str(inputs[0])])
        var.name = old_var[2:]
        return var

    if isrecv(fn):
        assert len(inputs) == 0 and idx == 0
        _, machine = fn
        var = theano.tensor.io.recv(known_shape_strings[str(old_var)],
                                    old_var.dtype,
                                    rankfile[machine],
                                    tagfile[str(old_var)])
        var.name = old_var.name
        return var

    return dicdag.theano.theano_dag.ith_output(fn, inputs, idx, old_var)

full_dags  = {m: dicdag.unidag.unidag_to_dag(dag)
                        for m, dag in cleaner_dags.items()}

def dag_to_fgraph(dag):
    tdag = dicdag.remove_index_entries(dicdag.insert_single_indices(dag))
    inputs = dicdag.inputs_of(tdag)
    outputs = dicdag.outputs_of(tdag)
    tins, touts = dicdag.tuple_dag_to_graph(tdag, inputs, outputs, ith_output)
    tins, touts = theano.gof.graph.clone(tins, touts)
    return theano.FunctionGraph(tins, touts)

fgraphs= {machine: dag_to_fgraph(dag) for machine, dag in full_dags.items()}
