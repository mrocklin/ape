import theano
from timings import compute_runtimes, compute_commtimes
from env_manip import variables_with_names
from util import save_dict

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y
variables_with_names([x,y], [z]) # give names to all variables between x,y and z


an_times = compute_runtimes([x,y], [z], {x:(1000,1000), y:(1000,1000)})
times = {('ankaa.cs.uchicago.edu','mimosa.cs.uchicago.edu'):an_times}
save_dict('compute_times.dat', times)

comm_dict = compute_commtimes([10,100,1000,2000]*5,
            {'ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu'}))
save_dict('comm_times.dat', comm_dict)
