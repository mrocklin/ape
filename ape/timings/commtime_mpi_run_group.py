if __name__ == '__main__':
    from mpi_prelude import *
    import time
    from sys import argv, stdout
    import ast

    ns, hosts = argv[1], argv[2:]
    ns = ast.literal_eval(ns)

    results = []

    for sender in hosts:
        for receiver in hosts:
            if sender == receiver: continue
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
                    results.append((sender, receiver, x.nbytes, duration))

                if host == sender:
                    send(x, n, receiver)

    comm.barrier()
    results_list = comm.gather(results)
    # Gather results back to head node
    if rank == 0:
        results = [res for results in results_list for res in results
                                                 if results] # flatten
        stdout.write("[%s]\n"%(',\n'.join(map(str, results))))
