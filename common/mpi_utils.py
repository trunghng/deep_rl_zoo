'''
Taken with minor modification from OpenAI Spinning Up's github
Ref:
[1] https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py
[2] https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_pytorch.py
'''
import os, sys, subprocess
from mpi4py import MPI
import torch
import numpy as np


comm = MPI.COMM_WORLD


def mpi_fork(n, bind_to_core=False):
    '''
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.

    :param n: (int) Number of processes to split into
    :param bind_to_core: (bool) Bind each MPI process to a core
    '''
    if n <= 1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-n", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def proc_rank():
    '''
    Get process's rank/id
    '''
    return comm.Get_rank()


def n_procs():
    '''
    Get number of processes
    '''
    return comm.Get_size()


def mpi_op(x, op):
    '''
    Do :param op: with :param x: and distribute the result to all processes
    '''
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    comm.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def broadcast(x, root=0):
    '''
    Broadcast :param x: from process :param root: to all other MPI processes
    '''
    comm.Bcast(x, root=root)


def mpi_sum(x):
    '''
    Do a summation over MPI processes and distribute the result to all of them
    '''
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    '''
    Get an average a over MPI processes and distribute the result to all of them
    '''
    return mpi_sum(x) / n_procs()


def mpi_mean_std(x):
    '''
    Get mean, standard deviation over data :param x: collected over MPI processes using
        STD >= 0
        STD^2 = Var = E(X^2) - (EX)^2
    '''
    x = np.array(x, dtype=np.float32)
    global_sum = mpi_sum(np.sum(x))
    mean = global_sum / (mpi_sum(x.size))
    global_sum_squared = mpi_sum(np.sum(x ** 2))
    std = np.sqrt(global_sum_squared - mean ** 2)
    return mean, std


def setup_pytorch_for_mpi():
    '''
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    '''
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / n_procs()), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    '''
    Average contents of gradient buffers across all MPI processes
    '''
    if n_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def print_one(msg, rank=0):
    '''
    :param msg: (str) Messege to print
    :param rank: (int) Rank of the process that is proceeded to print the messege
    '''
    if proc_rank() == rank:
        print(msg)


def sync_params(module):
    '''
    Sync all parameters of module across all MPI processes
    '''
    if n_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)
