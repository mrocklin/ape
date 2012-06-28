
if __name__ == '__main__':
    from mpi_prelude import *
    import time
    from sys import argv, stdout
    import ast

    ns, sender, receiver = argv[1:]
    ns = ast.literal_eval(ns)

    results = []

    for n in ns:
        x = np.ones(n, dtype=np.float32)

        if host == receiver:
            recv(x, n, sender)

        comm.barrier()
        if host == receiver:
            starttime = time.time()
            wait(n)
            endtime = time.time()
            duration = endtime - starttime
            results.append((x.nbytes, duration))

        if host == sender:
            send(x, n, receiver)

    if host == receiver:
        stdout.write("[%s]\n"%(',\n'.join(map(str, results))))

else:
    import ast
    import os
    import numpy as np
    def comm_times(ns, send_host, recv_host):
        """ Computes transit times between two hosts

        Returns a list of [(nbytes, transit-time)]

        >>> comm_times([10, 100, 1000], 'sender.univ.edu', 'receiver.univ.edu')
        [(4, 0.00013303756713867188),
         (40, 0.00015091896057128906),
         (400, 0.0002040863037109375),
         (4000, 0.0005209445953369141)]
        """

        hosts = (send_host, recv_host)
        file = open('_machinefile.txt', 'w')
        file.write('\n'.join(hosts))
        file.close()
        s = os.popen('''mpiexec -np 2 -machinefile _machinefile.txt python mpi_timing.py "%s" %s %s'''%(ns, hosts[0], hosts[1]))

        values = ast.literal_eval(s.read())
        return values

    def model(ns, send_host, recv_host):
        """ Computes the latency and inverse bandwidth between two hosts

        returns latency (intercept) and inverse bandwidth (slope) of the values
        returned by comm_times

        >>> model([10, 100, 1000], 'sender.univ.edu', 'receiver.univ.edu')
        (0.00017246120293471764, 8.8186875357188027e-08)

        time = .000172 + 8.818e-8*nbytes

        TODO - this function minimizes squared error. The larger values will
        dominate. Should weight nbytes = 10 similarly to nbytes = 1e9
        """

        values = comm_times(ns, send_host, recv_host)
        nbytes, times = zip(*values)
        slope, intercept = np.polyfit(nbytes, times, 1)
        return intercept, slope
