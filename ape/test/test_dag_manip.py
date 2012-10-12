from ape.dag_manip import *
from ape.util import merge
from theano.sandbox.cuda.basic_ops import GpuFromHost, GpuOp
import theano

def test_is_gpu_machine():
    assert is_gpu_machine('baconost.cs.uchicago.edu-gpu')
    assert not is_gpu_machine('baconost.cs.uchicago.edu')

def _comm_dag():
    x = theano.tensor.matrix('x')
    y = x + x; y.name = 'y'
    dag, inputs, outputs = dicdag.theano.theano_graph_to_dag((x,), (y,))
    recv = {x: {'fn': ("recv", "A"), 'args':()}}
    send = {'t_y': {'fn': ("send", "A"), 'args': (y,)}}
    comm_dag = merge(dag, send, recv)
    return dag, comm_dag, inputs, outputs

def test_non_comm_dag_real():
    dag, comm_dag, inputs, outputs = _comm_dag()
    dag2, sent, recvd = non_comm_dag(comm_dag)
    assert dag == dag2

def test_non_comm_dag():
    a,b,c,d,e = 'abcde'
    A,B,C,D,E = 'ABCDE'
    dag = {a: {'fn': ("recv", A), 'args':()},
           b: {'fn': "add", 'args': (a, c)},
           d: {'fn': ("send", A), 'args': (b,)}}

    non_comm, sent, recvd = non_comm_dag(dag)
    assert sent  == {b}
    assert recvd == {a}
    assert non_comm == {b: {'fn': "add", 'args': (a, c)}}

def test_internal_gpu_theano_graph():
    dag, comm_dag, inputs, outputs = _comm_dag()

    gins, gouts = internal_gpu_theano_graph(comm_dag)
    assert all(isinstance(var, theano.sandbox.cuda.var.CudaNdarrayVariable)
            for var in gins+gouts)
    assert all(isinstance(n.op, GpuOp)
                    for n in theano.gof.graph.list_of_nodes(gins, gouts))

    assert gins[0].name == 'gpu_x'
    assert gouts[0].name == 'gpu_y'

def test_merge_dags():
    from theano.tensor.basic import dot
    a,b,c,d,e,f = theano.tensor.matrices('abcdef')
    gdag = {b:  {'fn': dot, 'args': (a, a)},
         't_b': {'fn': ('send', 'cpu'), 'args': (b,)}}
    cdag = {d:  {'fn': dot, 'args': (c, c)},
            e:  {'fn': ('recv', 'gpu'), 'args': ()},
            f:  {'fn': dot, 'args': (d, e)}}

    assert merge_dags({'cpu': cdag, 'gpu':gdag}) == \
         {b:  {'fn': dot, 'args': (a, a)},
          d:  {'fn': dot, 'args': (c, c)},
          f:  {'fn': dot, 'args': (d, e)}}

def test_gpu_job():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.matrix('z')
    op = theano.tensor.add

    gi, gop, go = gpu_job((x,y), op, (z,))

    assert go[0].name == gpu_name(z.name)
    assert gi[0].name == gpu_name(x.name)
    assert gi[1].name == gpu_name(y.name)
    assert gop.__class__ == theano.sandbox.cuda.basic_ops.GpuElemwise

def test_gpu_dag():
    from theano.tensor.basic import dot
    a,b,c,d,e,f = theano.tensor.matrices('abcdef')

    dag = {c    : {'fn': dot, 'args': (a, b)},
           't_c': {'fn': ("send", "cpu"), 'args': (c,)}}
    gdag = gpu_dag(dag)
    assert all(isinstance(v['fn'], GpuOp) for v in gdag.values())
    assert c in gdag
    assert isinstance(gdag[c]['fn'], HostFromGpu)

def test_gpu_dag_transfers():
    from theano.tensor.basic import dot
    a,b,c = theano.tensor.matrices('abc')

    dag = {c    : {'fn': dot, 'args': (a, b)}}
    gdag = gpu_dag_transfers(dag)

    print gdag

    assert c in gdag
    assert isinstance(gdag[c]['fn'], HostFromGpu)
    assert all(isinstance(v['fn'], (HostFromGpu, GpuFromHost))
                            for v in gdag.values())
    assert all(any(inp in v['args'] for v in gdag.values())
                                    for inp in (a,b))
    assert len(gdag) == 3

def test_gpu_dag_sendrecvs():
    from theano.tensor.basic import dot
    a,b,c,d,e,f = theano.tensor.matrices('abcdef')

    dag = {c    : {'fn': dot, 'args': (a, b)},
           't_a': {'fn': ("send", "B"), 'args':(a,)},
           't_c': {'fn': ("send", "B"), 'args':(c,)}}
    gdag = gpu_dag(dag)
    assert a in gdag
    assert isinstance(gdag[a]['fn'], HostFromGpu)
    assert gdag[a]['args'][0].name == gpu_name(a.name)


def test_unify_by_name():
    from theano.tensor.basic import dot
    a,b,c = theano.tensor.matrices('abc')
    aa,bb,cc = theano.tensor.matrices('abc')

    dag = {c: {'fn': dot, 'args': (a, b)},
           a: {'fn': dot, 'args': (bb,)}}

    dag2 = unify_by_name(dag)

    a2 = filter(lambda v: v.name == 'a', dag2)[0]
    c2 = filter(lambda v: v.name == 'c', dag2)[0]
    # the bb and b above have been unified
    assert dag2[a2]['args'][0] in dag2[c2]['args']

def test_unify_by_name_with_seed():
    from theano.tensor.basic import dot
    a,b,c = theano.tensor.matrices('abc')
    aa,bb,cc = theano.tensor.matrices('abc')

    dag = {c: {'fn': dot, 'args': (a, b)},
           a: {'fn': dot, 'args': (bb,)}}

    dag2 = unify_by_name(dag, (a,b,c))

    # the bb and b above have been unified
    assert dag2[a]['args'] == (b, )

def test_variables():
    assert (variables({'a': {'fn': 1, 'args': ('b', 'c')}, 'c': {'fn': 2, 'args': ()}})
            ==
            set('abc'))

def test_merge_cpu_gpu_dags():
    from theano.tensor.basic import dot
    a,b,c,d,e,f = theano.tensor.matrices('abcdef')
    gdag = {b:  {'fn': dot, 'args': (a, a)},
         't_b': {'fn': ('send', 'cpu'), 'args': (b,)}}
    cdag = {d:  {'fn': dot, 'args': (c, c)},
            e:  {'fn': ('recv', 'gpu'), 'args': ()},
            f:  {'fn': dot, 'args': (d, e)}}

    dag = merge_cpu_gpu_dags('cpu', cdag, 'gpu', gdag)

    assert isinstance(dag[b]['fn'], HostFromGpu)

def test_flatten_values():
    assert flatten_values({1: (2, 3), 4: (5,)}) == [(1, 2), (1, 3), (4, 5)]

def test_sendrecvs():
    a,b,c,d,e = 'abcde'
    A,B,C,D,E = 'ABCDE'
    dag = {a: {'fn': ("recv", A), 'args':()},
           b: {'fn': "add", 'args': (a, c)},
           d: {'fn': ("send", A), 'args': (b,)}}

    assert sends(dag) == [(b, A)]
    assert recvs(dag) == [(a, A)]
