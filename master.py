import os
import dicdag
import ape
import tompkins
from ape.codegen import (write_inputs, write_rankfile, write_fgraph,
        write_hostfile, write_sched)

# Timings
from ape import timings
from ape.theano_util import shape_of_variables
from ape.util import save_dict, load_dict, dearrayify

# DicDag conversion
def makeapply(inputs, op, output):
    inputs = map(lambda x: x.clone(), inputs)
    outputs = (output.clone(), )
    return theano.Apply(op, inputs, outputs)

def replace_send_recvs(dag):
    return tompkins.dag.replace_send_recv(dag,
        lambda A, B, (a,b,fout), c : ((fout,), ("send", B), ("t_"+fout.name,)),
        lambda A, B, (a,b,fout), c : ((), ("recv", A), fout))

def make_ith_output(rankfile, tagfile, known_shapes):
    known_shape_strings = {str(k): v for k, v in known_shapes.items()}
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
    return ith_output

def dag_to_fgraph(dag, ith_output):
    tdag = dicdag.dag_to_tdag(dag)
    inputs = dicdag.inputs_of(tdag)
    outputs = dicdag.outputs_of(tdag)
    tins, touts = dicdag.tuple_dag_to_graph(tdag, inputs, outputs, ith_output)
    tins, touts = theano.gof.graph.clone(tins, touts)
    return theano.FunctionGraph(tins, touts)

def write(fgraphs, scheds, rankfile, rootdir, known_shapes):
    known_shape_strings = {str(k): v for k, v in known_shapes.items()}
    write_rankfile(rankfile, rootdir+"rankfile")
    write_hostfile(rankfile, rootdir+"hostfile")

    for machine, fgraph in fgraphs.items():
        write_fgraph(fgraph, rootdir+machine+".fgraph")
        write_inputs(fgraph, rootdir+machine+".inputs", known_shape_strings)
    for machine, sched  in  scheds.items():
        write_sched( sched,  rootdir+machine+".sched")

def run_command(rankfile,  rootdir):
    return (
        "mpiexec -np %(num_hosts)d -hostfile %(rootdir)shostfile "
        "-rankfile %(rootdir)srankfile python ape/codegen/run.py")%{
            'num_hosts': len(rankfile), 'rootdir': rootdir}

def blah(inputs, outputs, commtime, comptime, input_shapes):
    known_shapes = shape_of_variables(inputs, outputs, input_shapes)
    variables = theano.gof.graph.variables(inputs, outputs)

    assert all(var.name for var in variables)
    map(ape.env_manip.clean_variable, variables)

    dag, dinputs, doutputs = dicdag.theano.fgraph_to_dag(fgraph)
    unidag = dicdag.unidag.dag_to_unidag(dag)

    def dag_commtime(job, a, b):
        inputs, op, output = job
        return commtime(output, a, b)
    def dag_comptime(job, a):
        if job==dicdag.index:
            return 0
        return comptime(makeapply(*job), a)

    # Compute Schedule
    dags, sched, makespan = tompkins.schedule(
            unidag, machines, dag_comptime, dag_commtime,
            lambda j:0, lambda j,a:1, 10)

    cleaner_dags = {machine: replace_send_recvs(dag)
                        for machine, dag in dags.items()}
    full_dags  = {m: dicdag.unidag.unidag_to_dag(dag)
                            for m, dag in cleaner_dags.items()}

    scheds = {machine: tuple(makeapply(*job) for job, time, m in sched
                                             if m == machine)
                        for _, _, machine in sched}

    rankfile = {machine: i for i, machine in enumerate(dags)}
    tagfile  = {var: i for i, var in enumerate(map(str, fgraph.variables))}

    ith_output = make_ith_output(rankfile, tagfile, known_shapes)

    fgraphs = {machine: dag_to_fgraph(dag, ith_output)
                        for machine, dag in full_dags.items()}

    return fgraphs, scheds, rankfile

if __name__ == '__main__':
    from ape.examples.kalman import *
    from ape.examples.triple import machines, machine_groups, network
    rootdir = 'tmp/'
    os.system('mkdir -p %s'%rootdir)

    fgraph = theano.FunctionGraph(*theano.gof.graph.clone(inputs, outputs))
    recompute = True
    if recompute:
        comps = timings.comptime_dict(fgraph, input_shapes, 5, machines,
                machine_groups)
        comms = timings.commtime_dict(network)
        save_dict(rootdir+'comps.dat', comps)
        save_dict(rootdir+'comms.dat', comms)
    else:
        comps = load_dict(rootdir+'comps.dat')
        comms = load_dict(rootdir+'comms.dat')

    comptime = timings.make_runtime_function(comps)
    commtime = timings.make_commtime_function(comms, known_shapes)

    fgraphs, scheds, rankfile = blah(fgraph, commtime, comptime, input_shapes)
