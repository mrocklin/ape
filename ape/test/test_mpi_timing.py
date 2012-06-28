import ast
import os
from ape.mpi_timing import comm_times

def test_mpi_timing():
    values = comm_times([10, 100, 1000], 'ankaa.cs.uchicago.edu',
            'mimosa.cs.uchicago.edu')
    assert all(isinstance(n, int) for n,time in values)
    assert all(isinstance(time, float) for n,time in values)
