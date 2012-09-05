from ape.timings.computation.gpu import comptime_dict_gpu
from test_xpu import _test_comptime_dict_xpu

def test_comptime_dict_gpu():
    machines = {'baconostgpu': {'type':'gpu',
                                'host':'baconost.cs.uchicago.edu'}}
    machine_groups = (('baconostgpu', ),)
    _test_comptime_dict_xpu(machines, machine_groups, comptime_dict_gpu)
