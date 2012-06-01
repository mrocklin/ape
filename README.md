APE
===

This is a project to investigate the feasibility of static scheduling of array
primitive operations onto heterogeneous hardware.

Scheduling is hard. It is difficult to 

 * predict runtimes and communication times
 * predict runtimes and communication times on heterogeneous hardware
 * find an optimal schedule once these times have been obtained
 * write code to hetergeneous hardware

We pose that these problems become managable under a highly reduced and very
structured set of allowable operations. In particular array operations

 * Are highly predictable (once problem sizes exceed the cache)
 * Are easy to model
 * Can express many scientific programs with little complexity
 * Present a uniform interface to heterogeneous hardware

Technology
==========

Interface
---------

This project builds off of the 
[Theano project](http://deeplearning.net/software/theano/).
Theano presents a MatLab or NumPy style vectorized language to the user. I.e. 

    y = x[:,0].sum()

But instead of performing computations directly it builds up a graph. This
graph is what we choose to schedule.  

Scheduling
----------

Currently our scheduling backend is built off of integer programming and, if it
terminates, provides optimal solutions. You can read more about it
[here](http://github.com/mrocklin/Tompkins).

We also plan to build in a heuristic backend based around the HEFT algorithm. 

Communication
-------------

We communicate code using a network file system. 

We communicate at runtime using [mpi4py](http://mpi4py.scipy.org/).

Local Execution
---------------

For local execution we again depend on Theano. Theano provides implementations
for many array operations on both CPUs and GPUs, allowing us separate this
scheduling work from orthogonal work in many-core computing. 

What we produce
---------------
This project takes 
 * Theano input code like the example above (specifically we want a theano.Env)
 * Functions to estimate the cost of running and communicating operations on
   various machines in your network
 * Sizes of all inputs in your code

and produces
 * a file, env.dat which contains compilable theano objects
 * an orchestrating .py file that you can run with mpiexec/mpirun

Author
======
[Matthew Rocklin](http://matthewrocklin.com)

