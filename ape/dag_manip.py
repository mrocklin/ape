import dicdag
import ape

def is_gpu_machine(m):
    return m[-4:] == '-gpu'

def non_comm_dag(dag):
    """ Returns the computational core of dicdag. Cuts off send/recvs.

    returns core-of-dicdag, {sent_variables}, {recved_variables}"""
    from tompkins.dag import issend, isrecv
    non_comm = {k:v for k,v in dag.items()
            if not issend(v['fn']) and not isrecv(v['fn'])}
    sent_variables  = {v['args'][0] for k, v in dag.items() if issend(v['fn'])}
    recvd_variables = {k for k, v in dag.items() if isrecv(v['fn'])}
    return non_comm, sent_variables, recvd_variables

inputs_of  = lambda dag: dicdag.inputs_of( dicdag.dag_to_tdag(dag))
outputs_of = lambda dag: dicdag.outputs_of(dicdag.dag_to_tdag(dag))

def internal_gpu_theano_graph(dag):
    """

    inputs - a dicdag with send/recvs
    outputs -
        gins/gots - a theano i/o graph with gpu variables and ops
        # sents - tuple of sent gpu variables
        # recvs - a tuple of recved gpu variables
        """

    non_comm, sent, recved = non_comm_dag(dag)
    inputs  = inputs_of( non_comm)
    outputs = outputs_of(non_comm)

    ins, outs = dicdag.theano.dag_to_theano_graph(non_comm, inputs, outputs)
    gins, gouts = ape.timings.theano_gpu_util.cpu_to_gpu_graph(ins, outs)

    return gins, gouts
