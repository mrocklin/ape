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

def pack(env, file=None):
    """
    Pickles a theano.Env into a string

    Inputs
    env - a theano.Env object
    file - a file into which we should dump the result
         - if left empty we return a string

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
    if file:
        cPickle.dump((ins, outs), file)
        return file
    else:
        s = cPickle.dumps((ins, outs))
        return s
def pack_many(envs, target_file):
    """ Pack many envs into a file """
    if isinstance(target_file, str):
        target_file = open(target_file, 'r')
    for env in envs:
        pack(env, target_file)
    return target_file

def unpack(source):
    """
    Unpickle a packed string into an Env

    source can be a
        string
        file

    if the source is a file then one can call pack and unpack many times on the
    same file

    See Also:
        pack
    """
    if isinstance(source, str):
        ins, outs = cPickle.loads(source)
    elif isinstance(source, file):
        ins, outs = cPickle.load(source)

    env = theano.Env(ins, outs)
    return env
def unpack_many(target_file):
    """ Pack many envs into a file """
    if isinstance(target_file, str):
        target_file = open(target_file, 'r')
    envs = []
    while(True):
        try:
            envs.append(unpack(target_file))
        except EOFError:
           return envs
    assert False

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

    input_dims  = [dimension for inp in env.inputs
                             for dimension in env.shape_feature.shape_of[inp]]

    output_dims = [dimension for shape in env.shape_feature.shape_of.values()
                             for dimension in shape]

    compute_shapes = theano.function(input_dims, output_dims)

    numeric_input_dims  = [dim for inp in env.inputs
                               for dim in input_shapes[inp]]
    numeric_output_dims = compute_shapes(*numeric_input_dims)

    sym_to_num_dict = dict(zip(output_dims, numeric_output_dims))

    return {var: tuple(sym_to_num_dict[dim]
                             for dim in env.shape_feature.shape_of[var])
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
