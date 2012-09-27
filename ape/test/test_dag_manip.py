from ape.dag_manip import *
from ape.util import merge
from theano.sandbox.cuda.basic_ops import gpu_from_host
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
    assert all(isinstance(n.op, theano.sandbox.cuda.GpuOp)
                    for n in theano.gof.graph.list_of_nodes(gins, gouts))

    assert gins[0].name == 'gpu_x'
    assert gouts[0].name == 'gpu_y'

def test_merge_cpu_gpu():
    from theano.tensor.basic import dot
    a,b,c,d,e,f = theano.tensor.matrices('abcdef')
    gdag = {b:  {'fn': dot, 'args': (a, a)},
         't_b': {'fn': ('send', 'cpu'), 'args': (b,)}}
    cdag = {d:  {'fn': dot, 'args': (c, c)},
            e:  {'fn': ('recv', 'gpu'), 'args': ()},
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
