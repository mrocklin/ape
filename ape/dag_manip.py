import dicdag
import ape
from ape.theano_gpu_util import cpu_to_gpu_graph, cpu_to_gpu_var, gpu_name
from theano.gof.graph import list_of_nodes
try:
    from theano.sandbox.cuda.basic_ops import GpuFromHost, HostFromGpu
except ImportError:
    pass
from ape.util import merge
from tompkins.dag import issend, isrecv

def is_gpu_machine(m):
    return m[-4:] == '-gpu'

def non_comm_dag(dag):
    """ Returns the computational core of dicdag. Cuts off send/recvs.

    returns core-of-dicdag, {sent_variables}, {recved_variables}"""
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
    nc_dag, sent, recvd = non_comm_dag(dag)

    recvs = {cpu_to_gpu_var(var)[0].clone(): {'fn': GpuFromHost(),
                                              'args': (var,)}
                for var, it in dag.items()
                if isrecv(it['fn'])}

    sends = {it['args'][0]: {'fn': HostFromGpu(),
                             'args':(cpu_to_gpu_var(it['args'][0])[0].clone(),)}
                for _, it in dag.items()
                if issend(it['fn'])}

    def gpu_item((k, v)):
        i, op, o = v['args'], v['fn'], (k,)
        gi, gop, go = gpu_job(i, op, o)
        return (go[0], {'fn': gop, 'args': gi})
    gdag = dict(map(gpu_item, nc_dag.items()))

    return merge(gdag, recvs, sends)

def unify_variables(dag, fn, seed=None):
    """
    Create a new dag where all variables that equate under fn are the same
    """
    if seed:
        cache = dict(zip(map(fn, seed), seed))
    else:
        cache = {}
    def new(v):
        if fn(v) in cache:
            return cache[fn(v)]
        else:
            cache[fn(v)] = v
            return v
    return {new(k) : {'fn': v['fn'], 'args': tuple(map(new, v['args']))}
                        for k, v in dag.items()}

def unify_by_name(dag, seed=None):
    return unify_variables(dag, str, seed)

def merge_dags(dags):
    """ Merge dags - remove send/recvs between them

    input:
        dags - dict mapping {machine: dag}
    output
        Just a single dag
    """

    dag = merge(*dags.values())
    return {k: v for k,v in dag.items()
            if  not (issend(v['fn']) and v['fn'][1] in dags)
            and not (isrecv(v['fn']) and v['fn'][1] in dags)}

def variables(dag):
    """ All variables of a dicdag """
    return set.union({a for v in dag.values() for a in v['args']}, dag.keys())

def merge_cpu_gpu_dags(cpu_name, cdag, gpu_name, gdag):
    """ Merge a cpu and gpu dag - convert the gpu dag first """
    if any((issend(v['fn']) or isrecv(v['fn'])) and v['fn'][1] != cpu_name
            for v in gdag.values()):
        raise Exception("The GPU wants to communicate to someone who isn't the"
                        " host. We haven't yet built this functionality. TODO")

    dag = merge_dags({cpu_name: cdag, gpu_name: gpu_dag(gdag)})
    return unify_by_name(dag, tuple(variables(merge(cdag,
                                                    non_comm_dag(gdag)[0]))))

def is_sendrecv((k, job), s_or_r):
    assert s_or_r in ['send', 'recv']
    return isinstance(job['fn'], tuple) and job['fn'][0] == s_or_r

def flatten_values(d):
    """ Flatten out the values of a dict

    >>> flatten_values({1: (2, 3), 4: (5,)})
    [(1, 2), (1, 3), (4, 5)]
    """
    return [(k,v) for k, values in d.items() for v in values]

def sends(dag):
    sendjobs = [job for job in dag.items() if is_sendrecv(job, 'send')]
    return [(v['args'][0], v['fn'][1]) for (k, v) in sendjobs]
def recvs(dag):
    recvjobs = [job for job in dag.items() if is_sendrecv(job, 'recv')]
    return [(k, v['fn'][1]) for (k, v) in recvjobs]

def allsendrecvs(dags, s_or_r):
    return sum([sendrecvs(dag, s_or_r) for dag in dags], [])

def consistent_send_recvs(dags):
    allsends = flatten_values(fmap(sends, dags))
    allrecvs = flatten_values(fmap(recvs, dags))

