from ape.dag_manip import *

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
