import torch
import torch.distributed as dist

def reduce_mean(tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt