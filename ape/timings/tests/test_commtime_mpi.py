import ast
import os
from ape.timings.commtime_mpi import (comm_times_single, model,
        comm_times_group, model_dict_group, function_from_group_dict,
        commtime_dict_mpi)

def test_mpi_timing_single():
    values = comm_times_single([10, 100, 1000], 'ankaa.cs.uchicago.edu',
            'mimosa.cs.uchicago.edu')
    assert all(isinstance(n, int) for n,time in values)
    assert all(isinstance(time, float) for n,time in values)

def test_mpi_timing_group():
    values = comm_times_group([10, 100, 1000],
            {'ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu'})
    d = {(a,b,c):d for a,b,c,d in values}
    assert ('ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu', 400) in d

def test_model_dict_group():
    data = [('a','b',10., 1.), ('a','b',100., 1.),
            ('b','a',10., 1.), ('b','a',100., 1.)]
    d = model_dict_group(data)
    mod_dict = d['a', 'b']
    int, slope = mod_dict['intercept'], mod_dict['slope']

    assert abs(int-1)<.000001 and abs(slope - 0) < .000001

def test_function_from_group_dict():
    data = {('a','b'): (1, 1) , ('b','a'): (0, 10)}
    fn = function_from_group_dict(data)
    assert fn(0, 'a', 'b') == 1
    assert fn(0, 'b', 'a') == 0
    assert fn(1000, 'a', 'b') == 1001
    assert fn(1000, 'b', 'a') == 10000

def test_model():
    intercept, slope = model([10, 100, 1000, 10000], 'ankaa.cs.uchicago.edu',
            'mimosa.cs.uchicago.edu')
    assert isinstance(intercept, float)
    assert isinstance(slope, float)
    assert 0 < intercept < .01
    assert 0 < slope < .00001

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
