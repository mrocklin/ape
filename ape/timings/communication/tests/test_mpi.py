import ast
import os
from ape.timings.communication.mpi import (comm_times_single,
        comm_times_group, commtime_dict_mpi)

def test_comm_times_single():
    values = comm_times_single([10, 100, 1000], 'ankaa.cs.uchicago.edu',
            'mimosa.cs.uchicago.edu')
    assert all(isinstance(n, int) for n,time in values)
    assert all(isinstance(time, float) for n,time in values)

def test_comm_times_group():
    values = comm_times_group([10, 100, 1000],
            {'ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu'})
    d = {(a,b,c):d for a,b,c,d in values}
    assert ('ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu', 400) in d

def test_commtime_dict_mpi():
    machines = {'ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu',
                'milkweed.cs.uchicago.edu'}
    network = {(a,b): {'type':'mpi'} for a in machines
                                     for b in machines
                                     if a!=b}
    results = commtime_dict_mpi(network)
    result = results['ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu']
    assert set(result.keys()).issuperset(set(('intercept', 'slope')))
    assert isinstance(result['intercept'], float)
    assert set(results.keys()) == set(network.keys())
