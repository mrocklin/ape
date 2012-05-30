import theano
import theano.tensor as T
import cPickle

def math_optimize(env):
    """
    Creates a new Env that has been optimized using no gpu or inplace operations

    Does not overwrite input

    >>> env2 = math_optimize(env)
    """

    fast_run_no_inplace = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude = ['inplace', 'gpu']))
    optimized_env = env.clone()
    fast_run_no_inplace.optimize(optimized_env) # this is inplace
    return optimized_env

def pack(env):
    """
    Pickles a theano.Env into a string

    >>> import theano
    >>> import cPickle
    >>> from ape.env_manip import pack, unpack
    >>> x = theano.tensor.matrix('x')
    >>> y = x+x
    >>> env = theano.Env([x], [y])
    >>> type(pack(env))
    str
    >>> str(unpack(pack(env))) == str(env)
    True

    See Also:
        unpack
    """

    ins, outs  = theano.gof.graph.clone(env.inputs, env.outputs)
    s = cPickle.dumps((ins, outs))
    return s

def unpack(s):
    """
    Unpickle a packed string into an Env

    See Also:
        pack
    """
    ins, outs = cPickle.loads(s)
    env = theano.Env(ins, outs)
    return env

def shape_of_variables(env, input_shapes):
    """

    Inputs:
        env - the theano.Env in question
        input_shapes - a dict mapping input to shape

    Outputs:
        shapes - a dict mapping variable to shape

    WARNING : This modifies the env! Not pure!

    """
    if not hasattr(env, 'shape_feature'):
        env.extend(theano.tensor.opt.ShapeFeature())

    sym_to_num_dict = {sym: num
                        for input in input_shapes
                        for sym, num in zip(env.shape_feature.shape_of[input],
                                            input_shapes[input])}
    def sym_to_num(sym):
        """ sym to num dict doesn't hold theano constants - add a case """
        if sym in sym_to_num_dict:
            return sym_to_num_dict[sym]
        if isinstance(sym, theano.Constant):
            return sym.value

    return {var: tuple(map(sym_to_num, env.shape_feature.shape_of[var]))
            for var in env.shape_feature.shape_of}

def precedes(a, b):
    """ does a directly precede b ? """
    return len(set(a.outputs).intersection(b.inputs)) != 0

def env_with_names(env):
    ins, outs  = theano.gof.graph.clone(env.inputs, env.outputs)
    env = theano.Env(ins, outs)

    for i, var in enumerate(env.variables):
        var.name = var.name or "var_%d"%i

    return env
