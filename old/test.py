import theano
import theano.tensor as T
import numpy
from Job import Job

def mmul(a,b, **kwargs):
    #return T.tensordot(a,b,axes=[1,0], **kwargs)
    return T.dot(a,b, **kwargs)

fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_no_inplace = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude = ['inplace']))

#fast_run_no_inplace = fast_run.excluding('inplace')


x = T.matrix('x')
y = mmul(x,x)
z = y.sum()

dx = T.matrix('dx')
dz = T.Rop(z,x,dx)

f = theano.function([x], z)
fp = theano.function([x,dx], dz)
ffp = theano.function([x,dx], [z,dz])


As = [T.matrix('A%d'%i) for i in range(5)]
Bs = [mmul(A,A) for A in As]
C = sum(Bs)
g = theano.function(As, C)

someAs = [numpy.eye(3, dtype=numpy.float32) for i in range(5)]

myenv = theano.FunctionGraph(As, [C])
myoptimizer = f.maker.mode.optimizer

def show_optimization_path(env, filename='opt_path.txt', optimizer=myoptimizer):
    file = open(filename, 'w')
    for opt in myoptimizer:
        file.write('\n\n')
        file.write(str(opt)+'\n')
        opt.optimize(env)
        theano.printing.debugprint(env, file=file)
    file.flush()
    file.close()


# can now do
# someAs = someAs = [numpy.eye(3, dtype=numpy.float32) for i in range(5)]
# d = intermediate_shapes(As, [C], someAs)
# aps = list(all_applys([C]))
# d[aps[i].inputs[j]]

env = theano.FunctionGraph([x], [z])
an = env.outputs[0].owner
job = Job(an)

f = job.function()
#g = job.function(gpu=True)
