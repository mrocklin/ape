import theano
from ape.codegen.util import (write_inputs, write_rankfile, read_inputs,
        write_fgraph, read_fgraph)
import os

input_filename = 'testinputs'
testdir = 'tmp/'
os.system('mkdir -p %s'%testdir)
def test_write_inputs():
    fname = testdir + input_filename
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    z = x + y
    fgraph = theano.FunctionGraph((x,y), (z,))
    write_inputs(fgraph, fname, {'x': (10, 10), 'y':(10, 10)})
    file = open(fname); s = file.read(); file.close()
    assert s == ("import numpy as np\n"
    "x = np.random.rand(*(10, 10)).astype('float32')\n"
    "y = np.random.rand(*(10, 10)).astype('float32')\n"
    "inputs = (x, y)\n")

def test_read_inputs():
    test_write_inputs()
    inputs = read_inputs(testdir+input_filename)
    assert len(inputs) == 2
    assert [i.shape for i in inputs] == [(10, 10), (10, 10)]

def test_write_rankfile():
    fname = testdir + "test_rankfile"
    rankfile = {"A": 0, "B": 2, "C": 1}
    write_rankfile(rankfile, fname)
    file = open(fname); s = file.read(); file.close()
    assert s == (
            "rank 0=A slot=0\n"
            "rank 1=C slot=0\n"
            "rank 2=B slot=0\n")

def test_read_write_fgraph():
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    z = x + y
    fgraph = theano.FunctionGraph((x,y), (z,))

    fname = testdir + "test_read_write_fgraph"
    write_fgraph(fgraph, fname)
    fgraph2 = read_fgraph(fname)
    assert str(fgraph) == str(fgraph2)
    assert isinstance(fgraph2, theano.FunctionGraph)
