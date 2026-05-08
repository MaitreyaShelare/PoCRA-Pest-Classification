# """
# Synchronize metrics across all ranks in DDP.
# Compute global metrics from local metrics on each GPU.
# """

# import torch
# from typing import Dict, List
# from src.distributed.setup import is_initialized, get_world_size


# def reduce_dict(input_dict: Dict[str, float], average: bool = True) -> Dict[str, float]:
#     """
#     Reduce dictionary of metrics across all processes.
#     All processes get the same reduced dict.
    
#     Args:
#         input_dict: Dict of metric_name -> scalar_value
#         average: If True, compute mean; if False, sum
    
#     Returns:
#         Dict of reduced metrics
#     """
#     if not is_initialized():
#         return input_dict
    
#     world_size = get_world_size()
    
#     output_dict = {}
#     for key, value in input_dict.items():
#         # Convert to tensor for all_reduce
#         tensor = torch.tensor(value, dtype=torch.float32).cuda()
        
#         # All-reduce: sum across all processes
#         torch.distributed.all_reduce(tensor)
        
#         # Average if requested
#         if average:
#             tensor = tensor / world_size
        
#         output_dict[key] = tensor.item()
    
#     return output_dict


# def gather_list(object_list: List, max_size: int = 100) -> List:
#     """
#     Gather list from all processes to rank 0.
#     All processes contribute their lists; rank 0 gets concatenated result.
    
#     Args:
#         object_list: Local list of objects to gather
#         max_size: Maximum list size (for pre-allocation)
    
#     Returns:
#         Concatenated list (on rank 0 only; others get empty list)
#     """
#     if not is_initialized():
#         return object_list
    
#     world_size = get_world_size()
#     rank = torch.distributed.get_rank()
    
#     # Serialize lists of scalars
#     local_size = torch.tensor(len(object_list), dtype=torch.int64).cuda()
#     size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
#     torch.distributed.all_gather(size_list, local_size)
    
#     max_size = max(size.item() for size in size_list).int()
    
#     # Pad and gather
#     padded = object_list + [0] * (max_size - len(object_list))
#     gathered_list = [torch.zeros(max_size, dtype=torch.float32).cuda() 
#                      for _ in range(world_size)]
    
#     torch.distributed.all_gather(gathered_list, torch.tensor(padded[:max_size]).cuda())
    
#     # Reconstruct on rank 0
#     if rank == 0:
#         result = []
#         for i, size in enumerate(size_list):
#             result.extend(gathered_list[i][:size.item()].tolist())
#         return result
#     else:
#         return []


# def synchronize_between_processes() -> None:
#     """
#     Simple barrier: wait for all processes.
#     """
#     if is_initialized():
#         torch.distributed.barrier()

"""
Synchronize metrics across all ranks in DDP.
Compute global metrics from local metrics on each GPU.
"""

import torch
import torch.distributed as dist
from typing import Dict, List

from src.distributed.setup import (
    is_initialized,
    get_world_size,
)

def reduce_dict(
    input_dict: Dict[str, float],
    average: bool = True,
) -> Dict[str, float]:
    """
    Reduce dictionary of metrics across all processes.
    All processes get the same reduced dict.
    """

    if not is_initialized():
        return input_dict

    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():

        keys = sorted(input_dict.keys())

        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )

        values = torch.tensor(
            [input_dict[k] for k in keys],
            dtype=torch.float32,
            device=device,
        )

        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {
            k: v.item()
            for k, v in zip(keys, values)
        }

    return reduced_dict

def gather_list(
    object_list: List,
    max_size: int = 100,
) -> List:
    """
    Gather list from all processes to rank 0.

    All processes contribute their lists;
    rank 0 gets concatenated result.

    Args:
        object_list: Local list of objects to gather
        max_size: Maximum list size

    Returns:
        Concatenated list on rank 0;
        empty list on others
    """

    if not is_initialized():
        return object_list

    world_size = get_world_size()
    rank = dist.get_rank()

    device = torch.device(
        f"cuda:{torch.cuda.current_device()}"
    )

    # Local list size
    local_size = torch.tensor(
        len(object_list),
        dtype=torch.int64,
        device=device,
    )

    # Gather sizes
    size_list = [
        torch.zeros_like(local_size)
        for _ in range(world_size)
    ]

    dist.all_gather(size_list, local_size)

    max_size = int(
        max(size.item() for size in size_list)
    )

    # Pad list
    padded = object_list + [0] * (
        max_size - len(object_list)
    )

    padded_tensor = torch.tensor(
        padded[:max_size],
        dtype=torch.float32,
        device=device,
    )

    gathered_list = [
        torch.zeros(
            max_size,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(world_size)
    ]

    dist.all_gather(
        gathered_list,
        padded_tensor,
    )

    # Reconstruct on rank 0
    if rank == 0:

        result = []

        for i, size in enumerate(size_list):

            result.extend(
                gathered_list[i][: size.item()]
                .cpu()
                .tolist()
            )

        return result

    return []


def synchronize_between_processes() -> None:
    """
    Simple barrier:
    wait for all processes.
    """

    if is_initialized():
        dist.barrier()