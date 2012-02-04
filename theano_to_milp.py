from collections import defaultdict
import theano
import numpy as np
import theano.tensor as T
#from Job import *
from util import set_union

tdp = theano.printing.debugprint
fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_cpu_only = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude=['gpu']))
cpu_mode = theano.Mode(optimizer=fast_run_cpu_only, linker='py')

def precedence_dict(job):
    u = defaultdict(lambda:0)
    def add_precedence_info(j):
        for child in j.children:
            P[child,j] = 1 # child comes before j
            add_precedence_info(child)

    add_precedence_info(job)
    return P

def all_applys(outputs):
    applies = set()
    variables = list(outputs)
    for v in variables:
        if v.owner and v.owner not in applies:
            applies.add(v.owner)
            if v.owner.inputs:
                for input in v.owner.inputs:
                    if input not in variables:
                        variables.append(input)
    return applies

def intermediate_shapes(inputs, outputs, shapes):
    numeric_inputs = [np.ones(shape).astype(np.float32) for shape in shapes]

    apply_nodes = all_applys(outputs)

    intermediate_inputs = [i for an in apply_nodes for i in an.inputs]

    shapes = theano.function(inputs,
            [var.shape for var in intermediate_inputs+outputs])

    iinput_shape_dict = dict(zip(intermediate_inputs+outputs,
                                 shapes(*numeric_inputs)))
    return iinput_shape_dict

def togpu_data(x, copy=True):
    if isinstance(x, np.ndarray):
        return theano.sandbox.cuda.shared_constructor(x).get_value(
                borrow=True, return_internal_type=True)
    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return xg.get_value(return_internal_type=True, borrow=copy)
    assert False

def tocpu_data(x, copy=True):
    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return x.get_value(return_internal_type=False)
    assert False
