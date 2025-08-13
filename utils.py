import torch
import torch.nn as nn
from typing import Union
import functools
import torch.distributed as dist
from torch.utils._mode_utils import no_dispatch

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


def is_flattened(params, flat_param, is_padding_mask):
    for param, is_padding in zip(params, is_padding_mask):
        if is_padding:
            continue
        if param not in flat_param:
            return False
    return True


def _get_fsdp_handles(module):
    pass


def _get_fsdp_states(module):
    pass


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]],
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)


def _same_storage_size(a: torch.Tensor, b: int):
    return a.untyped_storage().size() // a.element_size() == b


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    if div_factor > 1:
        tensor.div_(div_factor)


def _no_dispatch_record_stream(tensor, stream):
    with no_dispatch():
        tensor.record_stream(stream)
