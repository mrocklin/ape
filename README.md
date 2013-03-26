APE
===

*This project is defunct.  I came to the conclusion that it was trying to
accomplish too much.*

This is a project to investigate the feasibility of statically scheduling array
primitive operations onto heterogeneous hardware.

Scheduling is hard. It is difficult to 

 * predict runtimes 
 * predict communication times
 * support operations on heterogeneous hardware
 * find an optimal schedule (this is NP-Hard)

We pose that these problems become managable under a highly reduced and very
structured set of allowable operations. In particular we consider array operations as such a set. This is because they

 * are highly predictable (once problem sizes exceed the cache)
 * are easy to model
 * can express many scientific programs with little complexity (NP-Hard
   problems become feasible)
 * Present a uniform interface to heterogeneous hardware (I.e. we have both a
   CPU and GPU BLAS)

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
[here](http://github.com/mrocklin/Tompkins). APE depends on this repository.

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
===============

This project takes 
 * Theano input code like the example above (specifically we want a theano.   
   FunctionGraph)
 * Functions to estimate the cost of running and communicating operations on
   various machines in your network
 * Sizes of all inputs in your code

and produces
 * a file, env.dat which contains compilable theano objects
 * an orchestrating .py file that you can run with mpiexec/mpirun

Look at `ape/example.py` and `ape/master.py` for an example.

Status
======

This project is not functional. You should not use it. 

Author
======
[Matthew Rocklin](http://matthewrocklin.com)

