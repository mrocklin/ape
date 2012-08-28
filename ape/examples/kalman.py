import theano
import theano.sandbox.linalg as linalg

mu = theano.tensor.matrix('mu')
Sigma = theano.tensor.matrix('Sigma')
H = theano.tensor.matrix('H')
R = theano.tensor.matrix('R')
data = theano.tensor.matrix('data')

dot = theano.tensor.dot

A = dot(Sigma, H.T)
B = R + dot(H, dot(Sigma, H.T))

new_mu    = mu + dot(A, linalg.solve(B, dot(H, mu) - data))
new_mu.name = "updated_mu"
new_Sigma = Sigma - dot(dot(A, linalg.solve(B, H)), Sigma)
new_Sigma.name = "updated_Sigma"

inputs = [mu, Sigma, H, R, data]
outputs = [new_mu, new_Sigma]

n = 2000
input_shapes = {mu:     (n, 1),
                Sigma:  (n, n),
                H:      (n, n),
                R:      (n, n),
                data:   (n, 1)}

machines = ('ankaa.cs.uchicago.edu','mimosa.cs.uchicago.edu')
