import theano
import numpy as np

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
