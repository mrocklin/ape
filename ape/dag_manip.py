def is_gpu_machine(m):
    return m[-4:] == '-gpu'

def non_comm_dag(dag):
    from tompkins.dag import issend, isrecv
    non_comm = {k:v for k,v in dag.items()
            if not issend(v['fn']) and not isrecv(v['fn'])}
    sent_variables  = {v['args'][0] for k, v in dag.items() if issend(v['fn'])}
    recvd_variables = {k for k, v in dag.items() if isrecv(v['fn'])}
    return non_comm, sent_variables, recvd_variables
