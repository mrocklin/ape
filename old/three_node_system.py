from IPython.parallel import Client
from theano_infrastructure import *
from infrastructure import CommNetwork, ComputationalSystem

# Connect to machines
rc = Client(profile = 'mpi')
view = rc[:]
importall(view)
rcs = [rc[ident] for ident in rc.ids]
machines = map(CPUWorker, rcs)
A = machines[0]
try: # add a gpu if we can
    G = GPUWorker(A)
    machines.append(G)
except:  pass

# Set up communication network
wires = [MPIWire(a,b) for a in machines for b in machines if a!=b]
try:    wires += [CPUWireGPU(A,G), GPUWireCPU(G,A)]
except: pass
network = CommNetwork(wires)

system = ComputationalSystem(machines, network)
