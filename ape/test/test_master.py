from ape.util import unique, load_dict, merge
from ape.master import (sanitize, make_apply, distribute,
        tompkins_to_theano_scheds, group_sched_by_machine, convert_gpu_scheds,
        start_jobs, is_start_job, remove_start_jobs, remove_end_jobs, end_jobs,
        remove_jobs_from_sched)
from ape.theano_util import shape_of_variables
import theano
import os
import dicdag

def test_sanitize():
    x = theano.tensor.matrix('x')
    y = x.T
    z = y + 1
    sanitize((x,), (z,))
    assert all(v.name and '.' not in v.name for v in (x,y,z))
    assert unique((x,y,z))
    print x, y, z

def test_make_apply():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    op = theano.tensor.elemwise.Sum()
    job = ((x,), op, y)
    apply = make_apply(*job)
    assert isinstance(apply, theano.Apply)
    assert apply.op == op
    assert apply.inputs[0].name == x.name
    assert apply.outputs[0].name == y.name

def make_job():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    op = theano.tensor.elemwise.Sum()
    return ((x,), op, y)

def test_group_sched_by_machine():
    sched = [(make_job(), 1, "A"),
             (make_job(), 1, "B"),
             (make_job(), 2, "A")]
    scheds = group_sched_by_machine(sched)
    assert set(scheds.keys()) == set('AB')
    assert all(isinstance(v, tuple) for vs in scheds.values()
                                    for v  in vs)

def test_convert_gpu_scheds():
    sched = [(make_job(), 1, "A"),
             (make_job(), 1, "B"),
             (make_job(), 2, "A")]
    machines = {"A": {'type': 'cpu'}, "B": {'type': 'gpu'}}
    scheds = group_sched_by_machine(sched)
    gpu_scheds = convert_gpu_scheds(scheds, machines)
    assert gpu_scheds["A"] == scheds["A"]
    assert gpu_scheds["B"] != scheds["B"]
    assert isinstance(gpu_scheds["B"][0][1], theano.sandbox.cuda.GpuOp)

def test_tompkins_to_theano_scheds():
    machines = {"A": {'type': 'cpu'}, "B": {'type': 'gpu'}}
    sched = [(make_job(), 1, "A"),
             (make_job(), 1, "B"),
             (make_job(), 2, "A")]
    scheds = tompkins_to_theano_scheds(sched, machines)
    assert set(scheds.keys()) == set('AB')
    assert all(isinstance(v, theano.Apply) for vs in scheds.values()
                                           for v  in vs)

def test_integration():
    from ape.examples.basic_computation import inputs, outputs, input_shapes
    from ape.examples.basic_computation import a,b,c,d,e
    from ape.examples.basic_network import machines, A, B
    from ape import timings
    comm_dict = load_dict("ape/test/integration_test_comm_dict.dat")
    comp_dict = load_dict("ape/test/integration_test_comp_dict.dat")

    rootdir = '_test/'
    os.system('mkdir -p %s'%rootdir)
    sanitize(inputs, outputs)

    known_shapes = shape_of_variables(inputs, outputs, input_shapes)
    comptime = timings.make_runtime_function(comp_dict)
    commtime = timings.make_commtime_function(comm_dict, known_shapes)

    assert isinstance(commtime(a, A, B), (int, float))
    assert commtime(a, A, B) == commtime(a, B, A)

    elemwise = e.owner
    dot = d.owner
    assert comptime(elemwise, A) == 1
    assert comptime(elemwise, B) == 100
    assert comptime(dot, A) == 100
    assert comptime(dot, B) == 1

    graphs, scheds, rankfile = distribute(inputs, outputs, input_shapes,
                                          machines, commtime, comptime, 50)

    # graphs == "{'A': ([b, a], [e]), 'B': ([a], [d])}"
    ais, [ao]  = graphs[A]
    [bi], [bo] = graphs[B]
    assert set(map(str, ais)) == set("ab")
    assert ao.name == e.name
    assert bi.name == a.name
    assert bo.name == d.name

    assert rankfile[A] != rankfile[B]
    assert str(scheds['B'][0]) == str(dot)
    assert map(str, scheds['A']) == map(str, (c.owner, e.owner))

    # test graphs, scheds, rankfile
    # test that inputs and outputs are untouched

    # Write to disk
    # write(graphs, scheds, rankfile, rootdir, known_shapes)

    # test files created
    # test that can read them correctly

def test_start_jobs():
    assert all(is_start_job(j)
                for j in dicdag.unidag.dag_to_unidag(start_jobs('abcd')))

def test_remove_start_jobs():
    udag = {((), 'start', ('a',)): ((('a',), 'add', ('b',)),),
            (('a',), 'add', ('b',)): ((('b',), 'end', ('output_b',)),),
            (('b',), 'end', ('output_b',)): ()}
    assert remove_end_jobs(remove_start_jobs(udag)) == {
            (('a',), 'add', ('b',)): ()}

def test_start_end_jobs():
    x = theano.tensor.matrix('x')
    y = theano.tensor.dot(x, x); y.name = 'y'
    dag, dinputs, doutputs = dicdag.theano.theano_graph_to_dag((x,), (y,))
    (dx,) = dinputs
    (dy,) = doutputs

    assert dx.name == x.name
    assert dy.name == y.name

    dag2 = merge(start_jobs(dinputs), end_jobs(doutputs), dag)
    assert dy in dag2
    assert any(len(v['args'])==1 and v['args'][0] == dy for v in dag2.values())

    unidag = dicdag.unidag.dag_to_unidag(dag2)

def test_remove_jobs_from_sched():
    sched = (((), 'start', 'a'),
             (('a',), theano.tensor.dot, 'b'),
             (('b',), 'end', 'output_b'))
    assert remove_jobs_from_sched(sched) == ((('a',), theano.tensor.dot, 'b'),)
