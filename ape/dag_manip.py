import dicdag
import ape
from ape.theano_gpu_util import cpu_to_gpu_graph, cpu_to_gpu_var, gpu_name
from theano.gof.graph import list_of_nodes
from theano.sandbox.cuda.basic_ops import GpuFromHost, HostFromGpu
from ape.util import merge

def is_gpu_machine(m):
    return m[-4:] == '-gpu'

def non_comm_dag(dag):
    """ Returns the computational core of dicdag. Cuts off send/recvs.

    returns core-of-dicdag, {sent_variables}, {recved_variables}"""
    from tompkins.dag import issend, isrecv
    non_comm = {k:v for k,v in dag.items()
            if not issend(v['fn']) and not isrecv(v['fn'])}
    sent_variables  = {v['args'][0] for k, v in dag.items() if issend(v['fn'])}
    recvd_variables = {k for k, v in dag.items() if isrecv(v['fn'])}
    return non_comm, sent_variables, recvd_variables

inputs_of  = lambda dag: dicdag.inputs_of( dicdag.dag_to_tdag(dag))
outputs_of = lambda dag: dicdag.outputs_of(dicdag.dag_to_tdag(dag))

def internal_gpu_theano_graph(dag):
    """

    inputs - a dicdag with send/recvs
    outputs -
        gins/gots - a theano i/o graph with gpu variables and ops
        # sents - tuple of sent gpu variables
        # recvs - a tuple of recved gpu variables
        """

    non_comm, sent, recved = non_comm_dag(dag)
    inputs  = inputs_of( non_comm)
    outputs = outputs_of(non_comm)

    ins, outs = dicdag.theano.dag_to_theano_graph(non_comm, inputs, outputs)
    gins, gouts = cpu_to_gpu_graph(ins, outs)

    return gins, gouts

def gpu_job(i, op, o):
    """ Convert a cpu job to a gpu job

    inputs:
        i  - iterable of input variables
        op - a cpu op
        o  - iterable of output variables
    outputs:
        gi  - iterable of input variables
        gop - a gpu op
        go  - iterable of output variables
    """
    node = op.make_node(*i)
    for v,nv in zip(o, node.outputs):
        nv.name = v.name

    gii, goo = cpu_to_gpu_graph(node.inputs, node.outputs)
    if len(list_of_nodes(gii, goo)) != 1:
        raise ValueError("We have assumed that translations of single cpu-ops "
                         "would result in single gpu-ops. This computation "
                         "has invalidated that assumption")
    op = goo[0].owner.op
    go = map(lambda x: x.clone(), goo)
    gi = map(lambda x: x.clone(), gii)

    for v, gv in zip(i+o, gi+go):
        gv.name = gpu_name(v.name)

    return gi, op, go

def gpu_dag(dag):
    """ The GPU version of a CPU dag - including gpu communication """
    i, o = inputs_of(dag), outputs_of(dag)
    recvs = {cpu_to_gpu_var(inp)[0].clone(): {'fn': GpuFromHost(), 'args': (inp,)}
                for inp in i}
    sends = {out: {'fn': HostFromGpu(), 'args': (cpu_to_gpu_var(out)[0].clone(),)}
                for out in o}
    def gpu_item((k, v)):
        i, op, o = v['args'], v['fn'], (k,)
        gi, gop, go = gpu_job(i, op, o)
        return (go[0], {'fn': gop, 'args': gi})
    gdag = dict(map(gpu_item, dag.items()))

    return merge(gdag, recvs, sends)

def unify_variables(dag, fn):
    """
    Create a new dag where all variables that equate under fn are the same
    """
    cache = {}
    def new(v):
        if v.name in cache:
            return cache[v.name]
        else:
            cache[v.name] = v
            return v
    return {new(k) : {'fn': v['fn'], 'args': map(new, v['args'])}
                        for k, v in dag.items()}

def unify_by_name(dag):
    return unify_variables(dag, lambda v: v.name)

def merge_dags(dags):
    """ Merge dags - remove send/recvs between them

    input:
        dags - dict mapping {machine: dag}
    output
        Just a single dag
    """
    from tompkins.dag import issend, isrecv

    dag = merge(*dags.values())
    return {k: v for k,v in dag.items()
            if  not (issend(v['fn']) and v['fn'][1] in dags)
            and not (isrecv(v['fn']) and v['fn'][1] in dags)}
