import theano
import theano.sandbox.linalg as linalg
from theano.tensor.utils import shape_of_variables
from theano.gof.fg import FunctionGraph as FunctionGraph
from ape.timings import compute_runtimes, compute_commtimes
from ape.env_manip import variables_with_names
from ape.util import save_dict, load_dict
from ape.timings import (make_runtime_function, make_commtime_function)
from ape.theano_to_milp import dummy_ability
from ape.master import compile

mu = theano.tensor.matrix('mu')
Sigma = theano.tensor.matrix('Sigma')
H = theano.tensor.matrix('H')
R = theano.tensor.matrix('R')
data = theano.tensor.matrix('data')

dot = theano.tensor.dot

A = dot(Sigma, H.T)
B = R + dot(H, dot(Sigma, H.T))

new_mu    = mu + dot(A, linalg.solve(B, dot(H, mu) - data))
new_Sigma = Sigma - dot(dot(A, linalg.solve(B, H)), Sigma)

inputs = [mu, Sigma, H, R, data]
outputs = [new_mu, new_Sigma]

variables_with_names(inputs, outputs) # give names to all variables between x,y and z


n = 1000
env = FunctionGraph(inputs, outputs)
input_shapes = {mu:     (n, 1),
                Sigma:  (n, n),
                H:      (n, n),
                R:      (n, n),
                data:   (n, 1)}

all_shapes = shape_of_variables(env, input_shapes)

machines = ('ankaa.cs.uchicago.edu','mimosa.cs.uchicago.edu')



# Compute Cost
compute_times = compute_runtimes(inputs, outputs, input_shapes)
compute_times = {machines: compute_times}
save_dict('compute_times.dat', compute_times)
# compute_times = load_dict('compute_times.dat')
compute_cost = make_runtime_function(compute_times)

# Communication Cost
comm_dict = compute_commtimes([10,100,1000,2000]*5,
            set(machines))
save_dict('comm_times.dat', comm_dict)
# comm_dict = load_dict('comm_times.dat')
comm_cost = make_commtime_function(comm_dict, all_shapes)


code = compile(env, machines, compute_cost, comm_cost,
        dummy_ability, input_shapes, 100)

f = open('results.py', 'w'); f.write(code); f.close()

