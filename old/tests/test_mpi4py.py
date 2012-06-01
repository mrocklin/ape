import mpi4py
from mpi4py import MPI
import numpy

def test_nonblocking_unordered_transfer():
    import mpi4py
    from mpi4py import MPI
    import numpy
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = numpy.arange(1000, dtype='i')
    data2 = numpy.arange(100, dtype='i')
    datarecv = numpy.empty(1000, dtype='i')
    data2recv = numpy.empty(100, dtype='i')

    # these are handled out of order
    comm.Isend([data, MPI.INT], dest=0, tag=1)
    comm.Isend([data2, MPI.INT], dest=0, tag=2)
    comm.Irecv([data2recv, MPI.INT], source=0, tag=2)
    comm.Irecv([datarecv, MPI.INT], source=0, tag=1)

    assert ((data-datarecv)==0).all()
    assert ((data2-data2recv)==0).all()


