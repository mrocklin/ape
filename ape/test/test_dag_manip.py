from ape.dag_manip import *
from ape.util import merge
from theano.sandbox.cuda.basic_ops import gpu_from_host
import theano

def test_is_gpu_machine():
    assert is_gpu_machine('baconost.cs.uchicago.edu-gpu')
    assert not is_gpu_machine('baconost.cs.uchicago.edu')

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

def test_internal_gpu_fgraph():
    x = theano.tensor.matrix('x')
    y = x + x; y.name = 'y'
    dag, inputs, outputs = dicdag.theano.fgraph_to_dag(
            theano.FunctionGraph(
                *theano.gof.graph.clone((x,), (y,))))
    recv = {x: {'fn': ("recv", "A"), 'args':()}}
    send = {'t_y': {'fn': ("send", "A"), 'args': (y,)}}
    comm_dag = merge(dag, send, recv)
    gins, gouts = internal_gpu_fgraph(comm_dag)
    assert all(isinstance(var, theano.sandbox.cuda.var.CudaNdarrayVariable)
            for var in gins+gouts)

    assert gins[0].name == 'gpu_x'
    assert gouts[0].name == 'gpu_y'

