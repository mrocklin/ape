import os
import dicdag
import ape
import theano
import tompkins
from ape.codegen import (write_inputs, write_rankfile, write_graph,
        write_hostfile, write_sched)

from ape import timings
from ape.theano_util import shape_of_variables
from ape.util import (save_dict, load_dict, dearrayify, merge, iterable,
        intersection, remove, fmap, unique)
from ape.dag_manip import merge_cpu_gpu_dags, gpu_job
from tompkins.dag import issend, isrecv

def sanitize(inputs, outputs):
    """ Ensure that all variables have valid names """
    variables = theano.gof.graph.variables(inputs, outputs)
    theano.gof.graph.utils.give_variables_names(variables)
    map(ape.env_manip.clean_variable, variables)
    assert all(var.name for var in variables)

def make_apply(inputs, op, outputs):
    """ Turn a unidag job (from tompkins) into a theano apply node """
    if not iterable(outputs):
        outputs = (outputs, )
    inputs  = tuple(map(lambda x: x.clone(),  inputs))
    outputs = tuple(map(lambda x: x.clone(), outputs))
    return theano.Apply(op, inputs, outputs)

def replace_send_recvs(dag):
    return tompkins.dag.replace_send_recv(dag,
        lambda A, B, (a,b,fout), c : ((fout,), ("send", B), "t_"+fout.name),
        lambda A, B, (a,b,fout), c : ((), ("recv", A), fout))

def make_ith_output(rankfile, tagfn, known_shapes, thismachine):
    known_shape_strings = {str(k): v for k, v in known_shapes.items()}
    def ith_output(fn, inputs, idx, old_var):
        if issend(fn):
            assert len(inputs) == 1 and idx == 0
            frommachine = thismachine
            _, tomachine = fn
            var = inputs[0]
            var = theano.tensor.io.send(var,
                                      rankfile[tomachine],
                                      tagfn(frommachine, str(var), tomachine))
            var.name = "mpi_token_"+old_var[2:]
            return var

        if isrecv(fn):
            assert len(inputs) == 0 and idx == 0
            tomachine = thismachine
            _, frommachine = fn
            var = theano.tensor.io.recv(known_shape_strings[str(old_var)],
                                        old_var.dtype,
                                        rankfile[frommachine],
                                        tagfn(frommachine, str(old_var),
                                               tomachine))

            var.name = "mpi_token_" + old_var.name
            return var

        return dicdag.theano.theano_dag.ith_output(fn, inputs, idx, old_var)
    return ith_output

def dag_to_theano_graph(dag, ith_output):
    tdag = dicdag.dag_to_tdag(dag)
    inputs = dicdag.inputs_of(tdag)
    outputs = dicdag.outputs_of(tdag)
    tins, touts = dicdag.tuple_dag_to_graph(tdag, inputs, outputs, ith_output)
    return tins, touts

def write(graphs, scheds, rankfile, rootdir, known_shapes):
    known_shape_strings = {str(k): v for k, v in known_shapes.items()}
    write_rankfile(rankfile, rootdir+"rankfile")
    write_hostfile(rankfile, rootdir+"hostfile")

    for machine, graph in graphs.items():
        write_graph(graph, rootdir+machine+".fgraph")
        write_inputs(graph, rootdir+machine+".inputs", known_shape_strings)
    for machine, sched  in  scheds.items():
        write_sched( sched,  rootdir+machine+".sched")

def run_command(rankfile,  rootdir):
    return (
        "mpiexec -np %(num_hosts)d -hostfile %(rootdir)shostfile "
        "-rankfile %(rootdir)srankfile python ape/codegen/run.py %(rootdir)s")%{
            'num_hosts': len(rankfile), 'rootdir': rootdir}

def convert_gpu_scheds(scheds, machines):
    """ Convert the jobs which correspond to gpu machines to gpu-jobs

    See also:
        group_sched_by_machine - produces input to this function
    """
    gjob = lambda i, op, o: gpu_job(i, op, (o,))
    return {m: tuple(map(gjob, *zip(*jobs))) if machines[m]['type'] == 'gpu'
                                         else jobs
                                for m, jobs in scheds.items()}

def group_sched_by_machine(sched):
    """
    inputs: a sched variable as returned by tompkins.schedule

    outputs: a dict mapping {machine: (i, op, o)} where the list is the in
             order schedule of nodes
    """
    return {machine: tuple(job for job, time, m in sched if m == machine)
                               for _, _, machine in sched}

def tompkins_to_theano_scheds(sched, machines):
    """
    inputs: a sched variable as returned by tompkins.schedule

    outputs: a dict mapping {machine: [apply_nodes]} where the list is the in
             order schedule of nodes
    """
    scheds = group_sched_by_machine(sched)
    scheds = convert_gpu_scheds(scheds, machines)
    return {m: tuple(map(make_apply, *zip(*jobs))) for m, jobs in scheds.items()
                                                   if     jobs}


def merge_gpu_dags(dags, machines):
    is_gpu = lambda m       : machines[m]['type'] == 'gpu'
    host   = lambda gpu_name: machines[gpu_name]['host']

    gpu_dags = {m for m in dags if is_gpu(m)}

    merge_dags = {host(g): merge_cpu_gpu_dags(host(g), dags[host(g)], g,dags[g])
                        for g in intersection(machines, dags) if is_gpu(g)}
    old_dags = {m: dags[m] for m in dags
                           if  not is_gpu(m)
                           and not m in merge_dags}
    new_dags = merge(merge_dags, old_dags)

    return new_dags

start = "start"
def start_jobs(inputs):
    return {i: {'fn': start, 'args': ()} for i in inputs}
end = "end"
def end_jobs(outputs):
    return {'output_'+o.name: {'fn': end, 'args': (o,)} for o in outputs}

def is_start_job((i,op,o)):
    return op==start
def is_end_job((i,op,o)):
    return op==end

def remove_start_jobs(udag):
    return {job: udag[job]
            for job in remove(is_start_job, udag)}
def remove_end_jobs(udag):
    return {job: tuple(remove(is_end_job, udag[job]))
            for job in remove(is_end_job, udag)}

def remove_jobs_from_sched(sched):
    """ Remove some silly jobs from a schedule """
    pred = lambda x: is_start_job(x) or is_end_job(x)
    return tuple(remove(pred, sched))

def tagof(A, var, B):
    key = (A, var, B)
    if key in tagof.cache:
        tagof.counts[key]+=1
        return tagof.cache[key]
    else:
        tag = tagof.nexttag
        tagof.nexttag += 1
        tagof.cache[key] = tag
        tagof.counts[key] = 1
        return tag
tagof.cache = {}
tagof.nexttag = 0
tagof.counts = {}

def check_send_recv(dags):
    """ Check result returned by manydags

    Check that the from/to jobs exist where the sends say they will
    Check that each send/recv has a partner
    """
    for m, dag in dags.items():
        for outvar, job in dag.items():
            if issend(job['fn']) or isrecv(job['fn']):
                if issend(job['fn']):
                    var = job['args'][0]
                    from_machine = m
                    to_machine = job['fn'][1]
                    print "send", to_machine, var, from_machine
                    if (str(dags[to_machine][var]) !=
                        str({'fn':('recv', from_machine), 'args':()})):
                        assert False
                        return False
                if isrecv(job['fn']):
                    var = outvar
                    to_machine = m
                    from_machine = job['fn'][1]
                    print "recv", to_machine, var, from_machine
                    if (str({'fn':('send', to_machine), 'args':(var,)}) not in
                        map(str, dags[from_machine].values())):
                        assert False
                        return False
    return True

def distribute(inputs, outputs, input_shapes, machines, commtime, comptime, makespan=100):
    known_shapes = shape_of_variables(inputs, outputs, input_shapes)
    variables = theano.gof.graph.variables(inputs, outputs)

    dag, dinputs, doutputs = dicdag.theano.theano_graph_to_dag(inputs, outputs)
    vars = set.union(set(dinputs), set(doutputs), set(dag.keys()),
            {v for value in dag.values() for v in value['args']})
    assert len(vars) == len(map(str, vars))

    unidag = dicdag.unidag.dag_to_unidag(dag)

    # TODO: This should be an input
    is_gpu       = lambda    m: machines[m]['type'] == 'gpu'
    can_start_on = lambda v, m: not is_gpu(m)
    can_end_on   = lambda v, m: not is_gpu(m)

    def dag_commtime(job, a, b):
        inputs, op, output = job
        return commtime(output, a, b)
    def dag_comptime(job, a):
        if job[1]==dicdag.index:
            return 0
        return comptime(make_apply(*job), a)

    # Compute Schedule
    dags, sched, makespan = tompkins.schedule(
            unidag, machines, dag_comptime, dag_commtime,
            lambda j:0, lambda j,a:1, makespan)

    cleaner_dags = fmap(replace_send_recvs, dags)

    full_dags  = fmap(dicdag.unidag.unidag_to_dag, cleaner_dags)
    check_send_recv(full_dags)

    merge_dags = merge_gpu_dags(full_dags, machines)
    check_send_recv(merge_dags)

    rankfile = {machine: i for i, machine in enumerate(sorted(merge_dags))}

    theano_graphs = {machine: dag_to_theano_graph(dag,
                     make_ith_output(rankfile, tagof, known_shapes, machine))
                            for machine, dag in merge_dags.items()}
    if not all(count == 2 for count in tagof.counts.values()):
        print "issue with tag counts"
        for x in tagof.counts:          print x
        for x in tagof.cache:           print x

    scheds = tompkins_to_theano_scheds(sched, machines)

    return theano_graphs, scheds, rankfile, makespan

if __name__ == '__main__':
    from ape.examples.kalman import inputs, outputs, input_shapes
    from ape.examples.triple import machines, machine_groups, network
    rootdir = 'tmp/'
    os.system('mkdir -p %s'%rootdir)

    # sanitize
    sanitize(inputs, outputs)

    # do timings if necessary
    recompute = False
    if recompute:
        comps = timings.comptime_dict(inputs, outputs, input_shapes, 5,
                                      machines, machine_groups)
        comms = timings.commtime_dict(network)
        save_dict(rootdir+'comps.dat', comps)
        save_dict(rootdir+'comms.dat', comms)
    else:
        comps = load_dict(rootdir+'comps.dat')
        comms = load_dict(rootdir+'comms.dat')

    known_shapes = shape_of_variables(inputs, outputs, input_shapes)
    comptime = timings.make_runtime_function(comps)
    commtime = timings.make_commtime_function(comms, known_shapes)

    # Break up graph
    graphs, scheds, rankfile, make = distribute(inputs, outputs, input_shapes,
                                                machines, commtime, comptime)

    # Write to disk
    write(graphs, scheds, rankfile, rootdir, known_shapes)

    # Print out fgraphs as pdfs
    fgraphs = {m: theano.FunctionGraph(*theano.gof.graph.clone(i, o))
                            for m, (i, o) in graphs.items()}
    for m, g in fgraphs.items():
        theano.printing.pydotprint(g, outfile="%s%s.pdf"%(rootdir,m),
                                      format="pdf")

    print "Makespan: %f"%make
    print run_command(rankfile,  rootdir)
