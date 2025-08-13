import torch
import torch.nn as nn
from typing import Union
import functools
import torch.distributed as dist
from torch.utils._mode_utils import no_dispatch
from enum import Enum, auto
from typing import Optional, Any, Union
from torch.distributed._composable.contract import _get_registry
from torch.distributed._composable_state import _get_module_state
from torch.distributed.fsdp._common_utils import _FSDPState

_FLAT_PARAM_PADDING_VALUE = 42
FSDP_FLATTENED = "_fsdp_flattened"


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


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


def _composable(module):
    "returns if a module is compatible with fsdp"
    # registry is a dict of module to distributed training strategy
    registry = _get_registry(module)

    if registry is None:
        return True
    # replicate is a distributed training strategy that replicates the model across all devices as in DDP
    return "replicate" not in registry


def _get_module_fsdp_state(module):
    # `_get_module_state` returns the distributed training state associated with a module from a global module state mapping
    state = _get_module_state(module)
    if state is None or not isinstance(state, _FSDPState):
        return None
    return state


def _get_fsdp_states_with_modules(module):
    # returns a tuple of list of fsdp states and list of corresponding modules in the heirachical order from the input module
    fsdp_states = []
    fsdp_modules = []

    visited_states = set()
    visited_modules = set()

    from collections import deque

    submodule_queue = deque([module])

    while submodule_queue:

        submodule = submodule_queue.pop()
        visited_modules.add(submodule)

        if not _composable(submodule):
            continue

        for child_module in reversed(list(submodule.children())):
            if child_module not in visited_modules:
                submodule_queue.appendleft(child_module)

        state = _get_module_fsdp_state(submodule)
        if state is not None and state not in visited_states:
            visited_states.add(state)
            fsdp_states.append(state)
            fsdp_modules.append(submodule)

    return fsdp_states, fsdp_modules


def _get_fsdp_states(module):
    fsdp_states, _ = _get_fsdp_states_with_modules(module)
    return fsdp_states


def _get_fsdp_handles(root_module):
    handles = [
        fsdp_state._handle
        for fsdp_state in _get_fsdp_states(root_module)
        if fsdp_state._handle is not None
    ]
    return handles


def _get_param_to_fqn(root_module):
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


def _set_fsdp_flattened(tensor):
    setattr(tensor, FSDP_FLATTENED, True)


def _is_fsdp_flattened(param):
    return getattr(param, FSDP_FLATTENED, False)


def _check_orig_params_flattened(module):
    for param_name, param in module.named_parameters():
        if not _is_fsdp_flattened(param):
            raise ValueError(f"Parameter {param_name} is not flattened")
