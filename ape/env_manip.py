import theano
import theano.tensor as T
import cPickle

def math_optimize(env):
    """
    Creates new FunctionGraph optimized using no gpu or inplace operations

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
    Pickles a theano.FunctionGraph into a string

    Inputs
    env - a theano.FunctionGraph object
    file - a file into which we should dump the result
         - if left empty we return a string

    >>> import theano
    >>> import cPickle
    >>> from ape.env_manip import pack, unpack
    >>> x = theano.tensor.matrix('x')
    >>> y = x+x
    >>> env = theano.FunctionGraph([x], [y])
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
    Unpickle a packed string into a FunctionGraph

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

    env = theano.FunctionGraph(ins, outs)
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

def precedes(a, b):
    """ does a directly precede b ? """
    return len(set(a.outputs).intersection(b.inputs)) != 0

def variables_of(env):
    """ Returns the variables of a FunctionGraph in a more predictable manner

    This is an alternative to env.variables() which produces a set """
    variables = [var for node in env.toposort()
                     for var in node.inputs + node.outputs]
    #variables = [var for node in env.toposort()
    #                 for var in node.inputs] + env.outputs
    out_variables = []
    for var in variables:
        if var not in out_variables:
            out_variables.append(var)
    return out_variables

def _clean(name):
    return name.replace('.', '_dot_')
def variables_with_names(inputs, outputs):
    """
    Name all variables between inputs and outputs

    Warning : Changes state! Not Pure!
    """
    all_variables = theano.gof.graph.variables(inputs, outputs)
    for i, var in enumerate(all_variables):
        name = (var.name or "") + "_var_%d"%i
        var.name = _clean(name)
    return all_variables

def env_with_names(env):
    ins, outs  = theano.gof.graph.clone(env.inputs, env.outputs)
    env = theano.FunctionGraph(ins, outs)

    for i, var in enumerate(variables_of(env)):
        var.name = var.name or "var_%d"%i

    return env
