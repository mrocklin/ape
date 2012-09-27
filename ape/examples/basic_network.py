A, B = 'AB'

machine_groups = ((A,), (B,))
machines = {a:{'type':'cpu'} for group in machine_groups for a in group}
network = {(a,b):{'type':'mpi'} for a in machines for b in machines if a!=b}
