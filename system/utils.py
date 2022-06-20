import torch
from contextlib import contextmanager


@contextmanager
def torch_distributed_zero_first(rank):
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if rank == 0:
        torch.distributed.barrier()