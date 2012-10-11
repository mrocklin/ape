Communication Computation Overlap
=================================

Consider the local MPI computation below 

![](mpi-po.pdf)

This graph specifies the dependencies between jobs. A job may not be run until all jobs to which it points are completed. Some jobs are communication jobs. `Recv` and `Send` are asynchronous and finish immediately, their corresponding `Wait` jobs block until the transmission is complete.

On a sequential machine we need to run these jobs in some order. A valid order can be found by performing a topological sort on the graph. Note that there are many valid orders. Two are produced below.

    Order 1: R1 RW1 R2 RW2 *2 +5 dot solve S1 SW1

    Order 2: R2 R1 RW2 *2 S1 RW1 +5 dot solve SW1

In a purely sequential environment the order is inconsequential as long as it satisfies the dependences of the DAG. Because we have asynchronous operations `S1, R1, R2` the order does matter. In order 1 notice that S1 is called just before SW1. We start and then wait on the transfer. In order 2 `S1` and `SW1` are separated by the operations `RW1, +5, dot, solve`. It is very likely that the communication time will will finish by the time these operations are finished and that `SW1` will complete immediately.

Order 1 is the naive ordering. Order 2 attempts to maximize communication computation overlap. How can we obtain orderings like 2 given a DAG?

Satisfying Multiple DAGs
========================

To maximize communication-computation overlap we want to order our nodes so
that we do all of our asynchronous Send/Recv starts, then all of our
computation, then finally all of our waits. In some sense we want to enforce a
DAG that looks like this. 

![](comp-comm.pdf)

while still satisfying our original dependence DAG

![](mpi-po.pdf)

In general this is impossible. For example notice that on the top `times-2` depends on `S1` but that in our original DAG `S1` depends on `times-2`. The desire that sends happen before computaiton must defer to the need to maintain data dependence (`Send` is sending the results of `times-2` so it must come afterwards). In general the graph on top defers to the graph on the bottom.

We could imagine adding a third DAG, for example one that prefers handling MPI Sends and Recvs with lower tags first. If all compute nodes adhere to this convention then we remove potential deadlocks.

The first DAG represents the data dependencies of the computation. It is essential. The other DAGs are desires. We desire to include as many edges as possible, assuming that these edges do not conflict with the first DAG.

Given a sequence of DAGs ordered by importance we construct a consensus DAG which contains as many edges as possible from all of the DAGs, preferring those earlier in the sequence to those later in the sequence. We perform a topological sort on the final conglomerate DAG to obtain an ordering. This DAG has far more edges/dependencies and so is far less ambiguous. There are relatively fewer correct orderings for the conglomerate DAG.

Partial Orders and Comparator functions
=======================================

*How do we input DAGs from scientific users?*

A DAG is equivalent to a partial order which can be completely described by a comparator function. Comparator functions are widely known in programming communities and serve as a good interface between scientific programmers and scheduling algorithms.

```Python
    def dependence(a, b):
        """ A cmp function for nodes in a graph - does a depend on b?

        Returns positive number if a depends on b
        Returns negative number if b depends on a
        Returns 0 otherwise
        """
        if depends((a, b)): return  1
        if depends((b, a)): return -1
        return 0
```

The communication/computation overlap dag shown above is highly symmetric. It consists of three groups. Each element within a group is ordered equivalently to the others in the group and has a clear ordering relative to the others. This pattern is indicative of a total order. Total orders are equivalent to sort-key functions. The sort-key function for the communication-computation overlap is defined below. 

```Python

def mpi_send_wait_key(a):
    """ Wait as long as possible on Waits, Start Send/Recvs early """
    if isinstance(a.op, (MPIRecvWait, MPISendWait)):
        return 1
    if isinstance(a.op, (MPIRecv, MPISend)):
        return -1
    return 0
```

and the key for "lower tags first" is here

``` python
def mpi_tag_key(a):
    """ Break MPI ties by using the variable tag - prefer lower tags first """
    if isinstance(a.op, (MPISend, MPIRecv, MPIRecvWait, MPISendWait)):
        return a.op.tag
    else:
        return 0
```

We have thus reduced the problem of ordering computations to maximize communication/computation overlap and eliminate deadlocks to these three functions, each of which is accessible to a programmer who is not familiar with DAG scheduling.

Extensions
==========

The computational core of these algorithms have nothing to do with scheduling. How else might they be used?

It is now simple to describe and execute different scheduling policies programmatically.
