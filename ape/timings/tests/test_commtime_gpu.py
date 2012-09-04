import ast
import os
from ape import ape_dir
import theano
from ape.timings.commtime_gpu import commtime_dict_togpu, commtime_dict_fromgpu

def test_commtime_tocpu_run():
    if not theano.sandbox.cuda.cuda_available:
        return
    ns = [10, 100]
    results = os.popen('python %s/ape/timings/commtime_togpu_run.py "%s"'%(ape_dir, str(ns))).read()
    results = ast.literal_eval(results)
    assert isinstance(results, list)
    assert all(len(result) == 2 for result in results)
    assert len(results) == len(ns)

def test_commtime_togpu_run():
    if not theano.sandbox.cuda.cuda_available:
        return
    ns = [10, 100]
    results = os.popen('python %s/ape/timings/commtime_tocpu_run.py "%s"'%(ape_dir, str(ns))).read()
    results = ast.literal_eval(results)
    assert isinstance(results, list)
    assert all(len(result) == 2 for result in results)
    assert len(results) == len(ns)

def test_commtime_dict_togpu():
    network = {('baconost.cs.uchicago.edu', 'baconost.cs.uchicago.edu_gpu'):
                     {'type':'togpu'}}
    result = commtime_dict_togpu(network)
    assert isinstance(result[network.keys()[0]]['intercept'], float)

def test_commtime_dict_fromgpu():
    network = {('baconost.cs.uchicago.edu_gpu', 'baconost.cs.uchicago.edu'):
                     {'type':'fromgpu'}}
    result = commtime_dict_fromgpu(network)
    assert isinstance(result[network.keys()[0]]['intercept'], float)

