import torch
from enum import auto, Enum
from utils import _get_fsdp_handles, _get_fsdp_states, _to_kwargs


class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class ExecutionOrder:
    pass


def _init_streams(state):
    assert state._is_root
    assert state._device_handle.is_available()

    state._default_stream = state._device_handle.current_stream()
    state._unshard_stream = state._device_handle.Stream()
    state._post_backward_stream = state._device_handle.Stream()


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]


def _reset_flat_param_grad_info_if_needed(handles):
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
    if not isinstance(handles, list):
        handles = [handles]
    for handle in handles:
        if handle._use_orig_params:
            handle._reset_flat_param_grad_info_if_needed()


def _share_state_and_init_handle_attrs(root_state):
    handle = root_state._handle
    root_state.all_handles = root_state._exec_order_data.all_handles

    for handle in root_state.all_handles:
        flat_param = handle.flat_param

        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError(
                "FSDP optimizer in backward only supported with use_orig_params=True!"
            )
        handle._has_optim_in_backward = flat_param._params is not None and any(
            hasattr(param, "_in_backward_optimizers") for param in flat_param._params
        )

    for fsdp_state in root_state._all_fsdp_states:
        if fsdp_state is root_state:
            continue
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        handle = fsdp_state._handle


def _lazy_init(state, root_module):
    # already initialized
    if state._is_root is not None:
        return

    if not state._device_handle.is_available():
        raise RuntimeError("FSDP does not support CPU only execution")

    state._is_root = True

    if state.training_state != TrainingState.IDLE:
        msg = f"Error: expected to be in states {TrainingState.IDLE} but got {state.training_state}"
        if state.rank == 0:
            print(msg)
        raise ValueError(msg)

    state._all_fsdp_states = _get_fsdp_states(root_module)
    _init_streams(state)
    state._exec_order_data.init(state, root_module, state.process_group)

    # share state and init handle attr
    _share_state_and_init_handle_attrs(state)

    return state


def _root_pre_forward(state, module, args, kwargs):
    _lazy_init(state, module)
    if not state._is_root:
        return args, kwargs

    if state.forward_prefetch:
        handles = [
            fsdp_state._handle
            for fsdp_state in state._all_fsdp_states
            if fsdp_state._handle
        ]
        for handle in handles:
            handle._needs_pre_forward = True
            handle._prefetched = False

    _wait_for_computation_stream(
        state._device_handle.current_stream(), state._unshard_stream
    )

    _reset_flat_param_grad_info_if_needed(state.all_handles)

    args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, state.compute_device, False)
    args = args_tuple[0] if args_tuple else tuple()
    kwargs = kwargs_tuple[0] if kwargs_tuple else {}

    return args, kwargs
