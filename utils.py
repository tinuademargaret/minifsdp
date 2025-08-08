import torch
import torch.nn as nn
from typing import Union
import functools
import torch.distributed as dist

_FLAT_PARAM_PADDING_VALUE = 42


def get_orig_params(module):
    return list(module.parameters())


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # returns the number of elements that fits 16 bytes
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@functools.lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE
    )


def _is_truly_contiguous(x: torch.Tensor) -> bool:
    # Special case: Pytorch thinks that 1x1 channels_last convolution weights are
    # both contiguous and channels_last contiguous at the same time.
    # CuDNN does not agree though and refuses to select faster kernels.
    # It is the reason of having the extra check here.
    return x.stride(-1) == 1 and x.is_contiguous()


def _detach_if_needed(
    param_or_tensor: Union[nn.Parameter, torch.Tensor],
) -> torch.Tensor:
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )


def _convert_to_params(
    tensors: list[Union[torch.Tensor, nn.Parameter]],
) -> list[nn.Parameter]:
    return [t if isinstance(t, nn.Parameter) else nn.Parameter(t) for t in tensors]


def is_param_sync(model: torch.nn.Module, rank: int):
    """Check that all parameters are synced across ranks."""
    # Each rank computes a list of parameter sums
    param = next(model.parameters())
    param_sum = torch.tensor(param.data.float().sum().item(), device="cuda")

    # Gather all param_summaries from all ranks
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(param_sum) for _ in range(world_size)]
    dist.all_gather(gathered, param_sum)

    # On rank 0, compare all gathered summaries
    if rank == 0:
        for i in range(1, world_size):
            if not torch.allclose(gathered[0], gathered[i]):
                return False
    return True
