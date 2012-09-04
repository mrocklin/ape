from ape.timings.commtime_util import ( model_dict_group,
        function_from_group_dict)
import numpy as np

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
