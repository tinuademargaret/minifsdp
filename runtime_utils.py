import functools
import itertools
import warnings
from typing import Any, Optional, Union
import torch
from enum import auto, Enum
from utils import (
    _get_fsdp_states,
    _get_fsdp_handles,
    _get_param_to_fqn,
    _div_if_needed,
    _no_dispatch_record_stream,
    HandleTrainingState,
)
import torch.distributed as dist
from _flat_param import (
    HandleShardingStrategy,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
from torch.autograd.graph import register_multi_grad_hook
from torch.utils import _pytree as pytree
from torch.autograd import Variable
from torch.distributed.fsdp._common_utils import _get_param_to_fqns
from torch.distributed.utils import _to_kwargs, _apply_to_tensors

HOMOGENEOUS_ATTR_NAMES = (
    "_use_orig_params",
    "limit_all_gathers",
)


class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto()
    FORWARD_BACKWARD = auto()
    SUMMON_FULL_PARAMS = auto()


class _PrefetchMode(Enum):
    BACKWARD = auto()
    FORWARD = auto()


class BackwardPrefetch(Enum):
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()


class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""

    NONE = auto()
    WARNING = auto()
    WARNED = auto()


class _ExecOrderData:

    def __init__(self, backward_prefetch_limit, forward_prefetch_limit):
        self.handles_pre_forward_order = []
        self.handles_post_forward_order = []
        self._iter = 0
        self._backward_prefetch_limit = backward_prefetch_limit
        self._forward_prefetch_limit = forward_prefetch_limit

        self.process_group = None
        self.world_size = None
        self.all_handles = []
        self.param_to_fqn = {}
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE

    def init(self, root_module, process_group):
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()

        for handle in _get_fsdp_handles(root_module):
            index = len(self.all_handles)
            self.all_handles.append(handle)
            handle._handle_index = index
        # mapping from flat parameter to it's fully qualified name if use_orig_param is False or from the original parameter to it's fully qualified name if use_orig_param is True
        self.param_to_fqn = _get_param_to_fqns(root_module)

    @property
    def is_first_iter(self):
        return self._iter == 0

    def record_pre_forward(self, handle, is_training):
        if not handle:
            return

        self._check_order(handle, is_training)

        if not self.is_first_iter or handle._pre_forward_order_index is not None:
            return
        index = len(self.handles_pre_forward_order)
        handle._pre_forward_order_index = index
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle, is_training):
        if not is_training:
            return
        if self.is_first_iter:
            msg_prefix = "Forward order differs across ranks:"
            optional_local_indices = self._get_handle_indices(handle)
            device = handle.device  # guaranteed to be non-CPU
            num_valid_indices = sum(
                (index is not None) for index in optional_local_indices
            )
            tensor_kwargs = {
                "dtype": torch.int32,
                "device": device,
            }
            world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)
            local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)
            dist.all_gather_into_tensor(
                world_num_valid_indices,
                local_num_valid_indices,
                group=self.process_group,
            )

            world_num_valid_indices = world_num_valid_indices.cpu()

            assert self.world_size is not None

            for (r1, n1), (r2, n2) in itertools.combinations(
                (
                    (rank, world_num_valid_indices[rank])
                    for rank in range(self.world_size)
                ),
                2,
            ):
                if n1 != n2:
                    raise RuntimeError(
                        f"{msg_prefix} rank {r1} is all-gathering {n1} parameters "
                        f"while rank {r2} is all-gathering {n2} parameters"
                    )
            world_indices = torch.zeros(
                self.world_size * num_valid_indices, **tensor_kwargs
            )
            local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)
            dist.all_gather_into_tensor(
                world_indices, local_indices, group=self.process_group
            )

            world_indices = world_indices.cpu()

            for (r1, i1), (r2, i2) in itertools.combinations(
                (
                    (
                        rank,
                        world_indices[
                            rank * num_valid_indices : (rank + 1) * num_valid_indices
                        ],
                    )
                    for rank in range(self.world_size)
                ),
                2,
            ):
                if i1 != i2:
                    r1_param_names = self._get_names_from_handle_indices(i1)
                    r2_param_names = self._get_names_from_handle_indices(i2)
                    raise RuntimeError(
                        f"{msg_prefix} rank {r1} is all-gathering parameters "
                        f"for {r1_param_names} while rank {r2} is all-gathering "
                        f"parameters for {r2_param_names}"
                    )
        else:
            if self.warn_status == _ExecOrderWarnStatus.WARNED:
                return
            msg_prefix = None
            if self.current_order_index >= len(self.handles_pre_forward_order):
                msg_prefix = (
                    "Expected to not all-gather any more parameters in the "
                    "forward but trying to all-gather parameters for "
                )
            else:
                expected_handle = self.handles_pre_forward_order[
                    self.current_order_index
                ]
                if expected_handle != handle:
                    expected_param_names = self._get_names_from_handles(expected_handle)
                    msg_prefix = (
                        f"Expected to all-gather for {expected_param_names} "
                        "but trying to all-gather parameters for "
                    )
            if msg_prefix is not None:
                param_names = self._get_names_from_handles(handle)
                msg_suffix = (
                    f"{param_names}"
                    if param_names
                    else "a newly-added parameter since construction time"
                )
                warnings.warn(
                    "Forward order differs from that of the first iteration "
                    f"on rank {self.rank}. Collectives are unchecked and may "
                    f"give incorrect results or hang.\n{msg_prefix}{msg_suffix}"
                )
                self.warn_status = _ExecOrderWarnStatus.WARNING
            self.current_order_index += 1

    def _get_handle_indices(self, handle):
        indices: list[Optional[int]] = []
        if handle:
            indices.append(handle._handle_index)
        return tuple(indices)

    def _get_names_from_handles(self, handle):
        fqns: list[list[str]] = []
        if handle:
            flat_param = handle.flat_param
            if flat_param in self.param_to_fqn:
                fqns.append(self.param_to_fqn[flat_param])
        return fqns

    def get_handle_to_backward_prefetch(self, current_handle):
        """
        Returns a :class:`list` of the handles keys of the handles to backward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._post_forward_index
        if current_index is None:
            return None
        target_index = current_index - 1
        target_handle = None
        for _ in range(self._backward_prefetch_limit):
            if target_index < 0:
                break
            # does this not overide the handle from the previous iteration?
            target_handle = self.handles_post_forward_order[target_index]
            target_index -= 1
        return target_handle

    def get_handle_to_forward_prefetch(
        self,
        current_handle,
    ):
        """
        Returns a :class:`list` of the handles keys of the handles to forward
        prefetch given the current handles key. If there are no valid handles
        keys to prefetch, then this returns an empty :class:`list`.
        """
        current_index = current_handle._pre_forward_order_index
        if current_index is None:
            return None
        target_index = current_index + 1
        target_handle = None
        for _ in range(self._forward_prefetch_limit):
            if target_index >= len(self.handles_pre_forward_order):
                break
            target_handle = self.handles_pre_forward_order[target_index]
            target_index += 1
        return target_handle

    def record_post_forward(self, handle):
        """
        Records ``handles`` in the post-forward order, where ``handles`` should
        be a group of handles used in the same module's forward. If ``handles``
        is empty, then it is omitted.

        Unlike :meth:`record_pre_forward`, this records the order *every*
        iteration with the expectation that the recorded order is reset in
        :meth:`next_iter`.
        """
        if not handle:
            return
        # Only record the first usage of a handles key
        if handle._post_forward_index:
            self.handles_post_forward_order.append(handle)
            return
        index = len(self.handles_post_forward_order)
        handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)
    
    def next_iter(self):
        self._iter += 1
        self.handles_post_forward_order.clear()


def _init_streams(state):
    assert state._is_root
    assert state._device_handle.is_available()

    state._default_stream = state._device_handle.current_stream()
    state._unshard_stream = state._device_handle.Stream()
    state._post_backward_stream = state._device_handle.Stream()
    state._all_reduce_stream = state._default_stream


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
):
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    unshard_stream.wait_stream(computation_stream)


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
    if handle:
        handle.init_flat_param_attributes()
    attr_name_to_values: dict[str, set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()

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
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            assert hasattr(fsdp_state, attr_name), (
                "fsdp state mising attribute " + attr_name
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()

    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(
                f"Expected only one value for {attr_name} but got {len(attr_values)}"
            )


def _unshard(state, handle, unshard_stream):
    if not handle:
        return
    # synchronize with previous all gathers before starting new ones
    # the unshard stream is like a thread that performs the unsharding the eventqueue manages the amount of unsharding(i.e all-gathers) that can be done concurrently since they are memory intensive
    # if the event queue is full, we need to dequeue and wait for the event to complete before we can enqueue a new one
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            event.synchronize()

    with state._device_handle.stream(unshard_stream):
        handle.unshard()


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
    state._exec_order_data.init(root_module, state.process_group)

    # share state and init handle attr
    _share_state_and_init_handle_attrs(state)

    return state


def _get_handle_to_prefetch(state, current_handle):
    assert current_handle is not None, "Expected current handle but got None"
    valid_training_states = [
        HandleTrainingState.FORWARD,
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
    ]
    training_state = current_handle._training_state
    assert (
        training_state in valid_training_states
    ), f"Expected training state to be in {valid_training_states} but got {training_state}"

    eod = state._exec_order_data
    target_handle = None

    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    return target_handle


def _prefetch_handle(state, current_handle, prefetch_mode):
    if not current_handle:
        return

    handle = _get_handle_to_prefetch(state, current_handle)
    if not handle:
        return
    prev_training_state = handle._training_state
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        handle._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError(
            f"Expected prefetch mode to be _PrefetchMode.BACKWARD or _PrefetchMode.FORWARD but got {prefetch_mode} on rank {state.rank}"
        )

    _unshard(state, handle, state._unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True


def _should_free_in_backward(state, handle) -> bool:
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    # If not syncing gradients, then we do not free for strategies that do not
    # reshard after forward as a *heuristic* to tradeoff higher memory for
    # higher throughput.
    return (
        state._sync_gradients
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )


def _post_backward_use_sharded_grad_views(handle):
    if not handle._use_orig_params:
        return
    # Since the handle's `FlatParameter` completed its gradient computation, we
    # should reset the gradient noneness mask
    handle._reset_is_grad_none()
    # Delay using sharded gradient views until after the reduce-scatter instead
    # of immediately after resharding
    handle._use_sharded_grad_views()
    # this is where optimizer in bwd fusion happens
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            # Check for `None` gradient to filter parameters not in the rank
            if orig_param.grad is not None and hasattr(
                orig_param, "_in_backward_optimizers"
            ):
                for optim in orig_param._in_backward_optimizers:
                    optim.step()

                optim.zero_grad(set_to_none=True)
        handle._reset_flat_param_grad_info_if_needed()


def _accumulate_sharded_grad(state, handle, new_sharded_grad):      
    flat_param = handle.flat_param

    accumulate_grad = hasattr(flat_param, "_saved_grad_shard")

    if accumulate_grad:
        assert flat_param._saved_grad_shard.shape == new_sharded_grad.shape, f"Expected saved grad shard shape {flat_param._saved_grad_shard.shape} to match new sharded grad shape {new_sharded_grad.shape}"
        assert flat_param._saved_grad_shard.device == new_sharded_grad.device, f"Expected saved grad shard device {flat_param._saved_grad_shard.device} to match new sharded grad device {new_sharded_grad.device}"
        flat_param._saved_grad_shard += new_sharded_grad 
    else:
        flat_param._saved_grad_shard = new_sharded_grad

def _reduce_grad(state, handle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param

    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = state.process_group

        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=pg,
        )
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(
            state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
        )
    _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_backward_use_sharded_grad_views(handle)


def _get_reduce_scatter_tensors(
    state, unsharded_grad
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = (
        F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    new_sharded_grad = torch.empty_like(chunks[0])  # padded
    return padded_unsharded_grad, new_sharded_grad


def _catch_all_reshard(state):
    try:
        if state._handle:
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
            )
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        assert False, f"Got exception in the catch all reshard for {state}: {str(e)}"


def _finalize_params(state):
    handle = state._handle
    if not handle:
        return

    flat_param = handle.flat_param

    if hasattr(flat_param, "_post_backward_hook_state"):
        post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
        expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
        assert (
            post_backward_hook_state_len == expected_post_backward_hook_state_len
        ), f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}"
        flat_param._post_backward_hook_state[-1].remove()
        delattr(flat_param, "_post_backward_hook_state")

    if flat_param.requires_grad:
        if not state._sync_gradients:
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        assert hasattr(
            flat_param, "_post_backward_called"
        ), "Expects _post_backward_called to be set on the FlatParameter"
        flat_param._post_backward_called = False


def _register_post_backward_final_callback(state, module):
    assert (
        state._is_root
    ), "Only the root fsdp instance should register the post backward callback"
    if state._post_backward_callback_queued:
        return

    assert state.training_state == TrainingState.IDLE

    state._post_backward_callback_queued = True

    Variable._execution_engine.queue_callback(
        functools.partial(_post_backward_final_callback, state, module)
    )


def _reshard(state, handle, free_unsharded_flat_param):
    handle.reshard(free_unsharded_flat_param)
    if state.limit_all_gathers and free_unsharded_flat_param:
        free_event = state._device_handle.Event()
        free_event.record()
        state._free_event_queue.enqueue(free_event)
    handle._prefetched = False


def _post_backward_reshard(
    state,
    handle,
    *unused: Any,
) -> None:
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    _reshard(state, handle, free_unsharded_flat_param)
    # prefetch the next handle to reshard
    _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


@torch.no_grad()
def _post_backward_hook(
    state,
    handle,
    flat_param,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    flat_param = handle.flat_param
    flat_param._post_backward_called = True

    assert state.training_state == TrainingState.FORWARD_BACKWARD

    assert handle._training_state in (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
    ), f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}"

    handle._training_state = HandleTrainingState.BACKWARD_POST

    if flat_param.grad is None:
        return
    if flat_param.grad.requires_grad:
        raise RuntimeError("FSDP does not support gradients of gradients")

    _post_backward_reshard(state, handle)
    if not state._sync_gradients:
        if handle._use_orig_params:
            handle._use_unsharded_grad_views()
        return

    state._post_backward_stream.wait_stream(state._device_handle.current_stream())

    with state._device_handle.stream(state._post_backward_stream):
        autograd_computed_grad = flat_param.grad.data
        _reduce_grad(state, handle)

        # Since the unsharded gradient is produced in the computation
        # stream and consumed in the post-backward stream, inform the
        # caching allocator (before it goes out of scope)
        _no_dispatch_record_stream(autograd_computed_grad, state._post_backward_stream)


def _post_backward_reshard_only_hook(state, handle):
    state.training_state = TrainingState.FORWARD_BACKWARD
    handle._training_state = HandleTrainingState.BACKWARD_POST
    _post_backward_reshard(state, handle)


# why is this registered in post forward and then pre backward hook?
def _post_backward_final_callback(state, module):
    assert (
        state._is_root
    ), "the post backward callback should only be registered on the fsdp root instance"

    root_state = state

    if root_state._sync_gradients:
        current_stream = state._device_handle.current_stream()
        current_stream.wait_stream(root_state._post_backward_stream)
        if root_state._all_reduce_stream is not current_stream:
            current_stream.wait_stream(root_state._all_reduce_stream)

    root_state._exec_order_data.next_iter()

    for fsdp_state in state._all_fsdp_states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state.training_state = TrainingState.IDLE
        handle = fsdp_state._handle

        if handle:
            handle._ran_pre_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
    # Reset for cases like one forward and multiple backwards
    root_state._post_backward_callback_queued = False


def _pre_backward_hook(state, module, handle, grad):

    if (
        handle
        and hasattr(handle, "_ran_pre_backward_hook")
        and handle._ran_pre_backward_hook
    ):
        return grad

    if state._is_root and not state._post_backward_callback_queued:
        _register_post_backward_final_callback(state, module)
        _reset_flat_param_grad_info_if_needed(state.all_handles)
    elif handle:
        assert state.training_state == TrainingState.IDLE, "Expects to be in IDLE state"

    state.training_state = TrainingState.FORWARD_BACKWARD

    if not handle:
        return grad

    handle._training_state = HandleTrainingState.BACKWARD_PRE

    if handle._needs_pre_backward_unshard:

        if not handle._prefetched:
            _unshard(state, handle, state._unshard_stream)

        state._device_handle.current_stream().wait_stream(state._unshard_stream)

    handle._needs_pre_backward_unshard = False
    _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
    handle.prepare_gradient_for_backward()
    handle._ran_pre_backward_hook = True
    return grad


def _post_forward_reshard(state, handle):
    if not handle:
        return

    free_unsharded_flat_param = (
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )

    _reshard(state, handle, free_unsharded_flat_param)


def _register_pre_backward_hooks(state, module, outputs, handle):
    if not torch.is_grad_enabled():
        return outputs

    if state._is_root:
        state._post_backward_callback_queued = False

    if handle:
        handle._needs_pre_backward_unshard = False
        handle._ran_pre_backward_hook = False

    def _register_hook(t):
        if t.requires_grad:
            t.register_hook(
                torch.utils.hooks.unserializable_hook(
                    functools.partial(_pre_backward_hook, state, module, handle)
                )
            )
            if handle:
                handle._needs_pre_backward_unshard = True
        return t

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hook(state, handle):
    # don't  register if gradient is not being calculated
    if not torch.is_grad_enabled():
        return

    if not handle:
        return

    flat_param = handle.flat_param

    already_registered = hasattr(flat_param, "_post_backward_hook_state")
    if already_registered or not flat_param.requires_grad:
        return

    temp_flat_param = flat_param.expand_as(flat_param)
    assert (
        temp_flat_param.grad_fn is not None
    ), "grad_fn is needed to access AccumulateGrad"

    acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(
        functools.partial(_post_backward_hook, state, handle)
    )
    flat_param._post_backward_hook_state = (acc_grad, hook_handle)


def _register_post_backward_reshard_only_hook(state, handle, args, kwargs):
    if torch.is_grad_enabled():
        return

    if not handle:
        return
    flat_param = handle.flat_param

    inp_tensors = None

    already_registered = hasattr(flat_param, "_post_backward_hook_state")

    if already_registered or flat_param.requires_grad:
        return

    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [
            obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad
        ]

    assert inp_tensors is not None

    hook_handle = register_multi_grad_hook(
        inp_tensors, functools.partial(_post_backward_reshard_only_hook, state, handle)
    )

    flat_param._post_backward_hook_state = (hook_handle,)


def _root_pre_forward(state, module, args, kwargs):
    _lazy_init(state, module)
    assert state._is_root is not None, "expects a root to have been set"
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
    # at this point we are starting a new forward pass, so we need to for compute stream to finish especially for parameter updates to be completed
    _wait_for_computation_stream(
        state._device_handle.current_stream(), state._unshard_stream
    )

    _reset_flat_param_grad_info_if_needed(state.all_handles)

    args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, state.compute_device, False)
    args = args_tuple[0] if args_tuple else tuple()
    kwargs = kwargs_tuple[0] if kwargs_tuple else {}

    return args, kwargs


def _pre_forward_unshard(state, handle):
    if not handle:
        return

    if not handle._prefetched:
        _unshard(state, handle, state._unshard_stream)
    handle._needs_pre_forward = False
    # I'm not sure why we need to wait for the unshard event here, since each state has just one handle and we always wait for all backward computation to finish before starting a new forward pass, I dont't think there should be an overlap of the unsharding of on handle
    current_stream = state._device_handle.current_stream()
    if state._unshard_event is not None:
        current_stream.wait_event(state._unshard_event)
        state._unshard_event = None
    else:
        current_stream.wait_stream(state._unshard_stream)
    # forward prefetch where we try to unshard the next handle in the recorded forward execution order assuming the order of execution is static
    _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


def _pre_forward(state, handle, unshard_fn, module, args, kwargs):
    # if we've already prefetched the handle we don't need to do it again
    if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
        return args, kwargs

    state.training_state = TrainingState.FORWARD_BACKWARD
    state._exec_order_data.record_pre_forward(handle, module.training)
    if handle:
        handle._training_state = HandleTrainingState.FORWARD
    if unshard_fn is not None:
        unshard_fn(state, handle)

    _register_post_backward_hook(state, handle)

    _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
    return args, kwargs


def _post_forward(state, handle, reshard_fn, module, input, output):

    # we don't want to reshard param that has been all gathered during backward prefetching
    if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
        return output

    state._exec_order_data.record_post_forward(handle)

    if reshard_fn is not None:
        reshard_fn(state, handle)

    output = _register_pre_backward_hooks(state, module, output, handle)
    state.training_state = TrainingState.IDLE
    if handle:
        handle._training_state = HandleTrainingState.IDLE
    return output
