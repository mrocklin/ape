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

