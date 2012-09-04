import ast
import os
from ape import ape_dir
import theano

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
