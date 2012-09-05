from ape.timings.computation.mpi import _compute_time_on_machine

def comptime_dict_cpu(fgraph, input_shapes, niter, machines,
                        machine_groups=None):
    if not machine_groups:
        machine_groups = tuple((machine,) for machine in machines)

    return {mg: _compute_time_on_machine('ape/timings/computation/run_cpu.py',
                                         fgraph, input_shapes, mg[0], niter)
            for mg in machine_groups
            if  machines[mg[0]]['type'] == 'cpu'}
