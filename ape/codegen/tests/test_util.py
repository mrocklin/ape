import theano
from ape.codegen.util import (write_inputs, write_rankfile, read_inputs,
        write_graph, read_graph, write_sched, read_sched, sched_to_cmp,
        make_scheduler)
import os
from ape import ape_dir

input_filename = 'testinputs'
testdir = 'tmp/'
os.system('mkdir -p %s'%testdir)
def test_write_inputs():
    fname = testdir + input_filename
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    z = x + y
    write_inputs(((x,y), (z,)), fname, {'x': (10, 10), 'y':(10, 10)})
    file = open(fname); s = file.read(); file.close()
    assert s.strip() == ("import numpy as np\n"
    "x = np.random.rand(*(10, 10)).astype('float32')\n"
    "y = np.random.rand(*(10, 10)).astype('float32')\n"
    "inputs = (x, y)").strip()

def test_write_inputs_no_inputs():
    fname = testdir + input_filename
    write_inputs(((), ()), fname, {})
    file = open(fname); s = file.read(); file.close()
    assert s.strip() == "import numpy as np\ninputs = ()".strip()

def test_read_inputs():
    test_write_inputs()
    inputs = read_inputs(testdir+input_filename)
    assert len(inputs) == 2
    assert [i.shape for i in inputs] == [(10, 10), (10, 10)]

def test_write_rankfile():
    fname = testdir + "test_rankfile"
    rankfile = {"a": 0, "b": 2, "c": 1}
    write_rankfile(rankfile, fname)
    file = open(fname); s = file.read(); file.close()
    assert s == (
            "rank 0=a slot=0\n"
            "rank 1=c slot=0\n"
            "rank 2=b slot=0\n")

def test_read_write_graph():
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    z = x + y

    fname = testdir + "test_read_write_fgraph"
    write_graph(((x,y), (z,)), fname)
    fgraph = theano.FunctionGraph((x,y), (z,))
    fgraph2 = read_graph(fname)
    assert str(fgraph) == str(fgraph2)
    assert isinstance(fgraph2, theano.FunctionGraph)

def _test_sched():
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    a = x + y
    b = x * y
    c = x ** y
    an, bn, cn = a.owner, b.owner, c.owner
    sched = [an, bn, cn]
    return sched

def test_sched_to_cmp():
    sched = an, bn, cn = _test_sched()
    cmp = sched_to_cmp(sched)
    assert cmp(an, bn) < 0 and cmp(cn, an) > 0
    d = theano.tensor.matrix('d')
    dn = (d+d).owner
    assert cmp(dn, an) == 0

def test_write_sched():
    sched = _test_sched()
    fname = testdir + "test_write_sched"
    write_sched(sched, fname)
    file = open(fname);
    assert len(file.readlines()) == 3
    file.close()

def test_make_scheduler():
    fgraph      = read_graph(ape_dir+'ape/codegen/tests/test.fgraph')
    sched       = read_sched(ape_dir+'ape/codegen/tests/test.sched')
    sched_cmp   = sched_to_cmp(sched)
    scheduler   = make_scheduler(sched_cmp)
    nodes       = scheduler(fgraph)
    nodestrings = map(str, nodes)


    print set(sched) - set(nodestrings)
    assert set(sched).issubset(set(nodestrings))

    indices = map(lambda line: nodestrings.index(line), sched)
    assert sorted(indices) == indices
