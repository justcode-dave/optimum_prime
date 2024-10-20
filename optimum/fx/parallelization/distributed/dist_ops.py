"""
This module defines distributed operations tailored for parallelization in deep learning models.
It leverages PyTorch's `torch.distributed` API to implement communication patterns such as all-reduce, 
all-gather, and scatter, essential for synchronizing and splitting data across multiple devices. 
Furthermore, these operations are differentiable, meaning that they can integrate with the backpropagation 
process and propagate gradients during training.

Key Functionalities:
    - **all_reduce**: A function to sum up tensors from all devices and distribute the result back.
    - **all_gather**: A function to gather tensors from all devices into a single tensor.
    - **split**: A function to split a tensor across devices based on the dimension specified.
    - **scatter**: A function to distribute slices of the input tensor across devices.
    - **Differentiable Functions**:
        - `DifferentiableIdentity`: A differentiable identity operation for distributed use.
        - `DifferentiableAllReduceSum`: A differentiable version of all-reduce.
        - `DifferentiableAllGather`: A differentiable version of all-gather.
        - `DifferentiableScatter`: A differentiable version of scatter.
    
These operations support efficient data distribution and aggregation in distributed deep learning training, 
ensuring synchronization and correct gradient propagation across multiple devices.

Functions:
    - all_reduce: Performs a sum of the tensor values across the devices in the specified process group.
    - all_gather: Gathers tensors from all devices into a single concatenated tensor.
    - split: Splits the input tensor along the specified dimension across the devices.
    - scatter: Distributes slices of a tensor across devices, based on the dimension specified.

Differentiable Operations:
    - differentiable_all_reduce_sum: Differentiable all-reduce sum for gradient propagation.
    - differentiable_identity: Differentiable identity operation.
    - differentiable_all_gather: Differentiable all-gather operation for gathering data from all devices.
    - differentiable_scatter: Differentiable scatter operation to split data across devices.
"""

import torch
import torch.distributed as dist

from ..utils import ensure_divisibility


def all_reduce(group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, group=group)
    return tensor


def all_gather(group: dist.ProcessGroup, tensor: torch.Tensor, gather_dim: int = -1) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    gather_dim = (gather_dim + tensor.ndim) % tensor.ndim
    shape = [tensor.size(dim) * world_size if dim == gather_dim else tensor.size(dim) for dim in range(tensor.ndim)]
    if gather_dim != 0:
        shape[0], shape[gather_dim] = shape[gather_dim], shape[0]
    tensors = torch.empty(*shape, dtype=tensor.dtype, device=tensor.device)

    if gather_dim != 0:
        tensor = tensor.transpose(0, gather_dim)
    tensor = tensor.contiguous()

    dist.all_gather_into_tensor(tensors, tensor, group=group)
    if gather_dim != 0:
        tensors = tensors.transpose(0, gather_dim).contiguous()
    return tensors


def split(group: dist.ProcessGroup, tensor: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    rank = dist.get_rank(group)
    size = tensor.size()
    ensure_divisibility(size[split_dim], world_size)
    tensors = torch.split(tensor, size[split_dim] // world_size, dim=split_dim)
    tensor = tensors[rank].contiguous()

    return tensor


def scatter(
    group: dist.ProcessGroup, tensor: torch.Tensor, output_tensor: torch.Tensor, scatter_dim: int = 0
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        output_tensor.copy_(tensor)
        return tensor

    rank = dist.get_rank(group)
    if rank == 0:
        size = tensor.size()
        ensure_divisibility(size[scatter_dim], world_size)
        tensors = torch.split(tensor, size[scatter_dim] // world_size, dim=scatter_dim)
        scatter_list = [tensor.contiguous() for tensor in tensors]
        output_tensor.copy_(scatter_list[rank])
    else:
        scatter_list = None
    dist.scatter(tensor=output_tensor, scatter_list=scatter_list, src=0, group=group)
    return output_tensor


class DifferentiableIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group: dist.ProcessGroup):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return all_reduce(group=group, tensor=tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Any:
        return grad_output, None


class DifferentiableScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return split(group=group, tensor=tensor, split_dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableAllGather.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


class DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, dim: int = -1) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        return all_gather(group=group, tensor=tensor, gather_dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return DifferentiableScatter.apply(grad_output, group=ctx.group, dim=ctx.dim), None, None


def differentiable_all_reduce_sum(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_identity(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1) -> torch.Tensor:
    return DifferentiableAllGather.apply(tensor, group, dim)


def differentiable_scatter(tensor: torch.Tensor, group: dist.ProcessGroup, dim=-1) -> torch.Tensor:
    return DifferentiableScatter.apply(tensor, group, dim)
