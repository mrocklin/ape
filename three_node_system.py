from IPython.parallel import Client
from theano_infrastructure import *
from infrastructure import CommNetwork, ComputationalSystem

# Connect to machines
rc = Client(profile = 'mpi')
view = rc[:]
importall(view)
a,b,c = rc[0], rc[1], rc[2]
A,B,C = map(CPUWorker, (a,b,c))
machines = [A,B,C]
try: # add a gpu if we can
    G = GPUWorker(C)
    machines.append(G)
except:  pass

# Set up communication network
wires = [MPIWire(a,b) for a in [A,B,C] for b in [A,B,C] if a!=b]
try:    wires += [CPUWireGPU(C,G), GPUWireCPU(G,C)]
except: pass
network = CommNetwork(wires)

system = ComputationalSystem(machines, network)
