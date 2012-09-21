import theano
from ape.codegen.util import write_input_file, write_rankfile

def test_write_input_file():
    x = theano.tensor.matrix('x', dtype='float32')
    y = theano.tensor.matrix('y', dtype='float32')
    z = x + y
    fgraph = theano.FunctionGraph((x,y), (z,))
    write_input_file(fgraph, '/tmp/.testinput', {'x': (10, 10), 'y':(10, 10)})
    file = open('/tmp/.testinput'); s = file.read(); file.close()
    assert s == ("import numpy as np\n"
    "x = np.random.rand(*(10, 10)).astype(float32)\n"
    "y = np.random.rand(*(10, 10)).astype(float32)\n")

def test_write_rankfile():
    rankfile = {"A": 0, "B": 2, "C": 1}
    write_rankfile(rankfile, "/tmp/.testrankfile")
    file = open('/tmp/.testrankfile'); s = file.read(); file.close()
    assert s == (
            "rank 0=A slot=0\n"
            "rank 1=C slot=0\n"
            "rank 2=B slot=0\n")
