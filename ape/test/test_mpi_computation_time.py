from ape.mpi_computation_time import compute_time_on_machine
import theano

def test_compute_time_on_machine():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x+1, 2*y)
    fgraph = theano.FunctionGraph((x,y), (z,))
    input_shapes = {x:(1000,1000), y:(1000,1000)}
    niter = 3
    machine = 'ankaa.cs.uchicago.edu'

    time = compute_time_on_machine(fgraph, input_shapes, machine, niter)
    assert isinstance(time, float)
