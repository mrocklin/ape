from ape.timings.computation.cpu import comptime_dict_cpu
from test_xpu import _test_comptime_dict_xpu

def test_comptime_dict_cpu():
    from ape.examples.nfs_triple import machines, machine_groups
    _test_comptime_dict_xpu(machines, machine_groups, comptime_dict_cpu)
