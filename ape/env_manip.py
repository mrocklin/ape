import theano
import theano.tensor as T
import cPickle

def math_optimize(env):
    fast_run_no_inplace = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude = ['inplace', 'gpu']))
    optimized_env = env.clone()
    fast_run_no_inplace.optimize(optimized_env) # this is inplace
    return optimized_env

def pack(env):
    ins, outs  = theano.gof.graph.clone(env.inputs, env.outputs)
    s = cPickle.dumps((ins, outs))
    return s

def unpack(s):
    ins, outs = cPickle.loads(s)
    env = theano.Env(ins, outs)
    return env
