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
