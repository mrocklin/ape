
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
    def comm_times(ns, send_host, recv_host):

        hosts = (send_host, recv_host)
        file = open('_machinefile.txt', 'w')
        file.write('\n'.join(hosts))
        file.close()
        s = os.popen('''mpiexec -np 2 -machinefile _machinefile.txt python mpi_timing.py "%s" %s %s'''%(ns, hosts[0], hosts[1]))

        values = ast.literal_eval(s.read())
        return values
