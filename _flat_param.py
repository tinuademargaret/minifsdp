import torch.nn as nn
import torch
from utils import (
    _get_aligned_numel,
    _construct_padding_tensor,
    _is_truly_contiguous,
    _detach_if_needed,
    _convert_to_params,
    _same_storage_size,
    _set_fsdp_flattened,
)
from itertools import accumulate, chain
from enum import auto, Enum
from typing import NamedTuple, Optional
from utils import HandleTrainingState
import torch.distributed as dist
from torch.distributed.utils import _free_storage, _alloc_storage
from torch.distributed.fsdp._common_utils import (
    _no_dispatch_record_stream,
    _FSDPDeviceHandle,
)


class HandleShardingStrategy(Enum):
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()
    _HYBRID_SHARD_ZERO2 = auto()


RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.HYBRID_SHARD,
)

NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.SHARD_GRAD_OP,
    HandleShardingStrategy._HYBRID_SHARD_ZERO2,
)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]
    numel_in_shard: Optional[int]
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g. either
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]` or
    # `param.as_strided((param.numel(),), (1,))[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]
    intra_param_end_idx: Optional[int]  # inclusive


class FlatParameter(nn.Parameter):
    """
    store initial shapes of each param in a unit, map?
    then convert to 1d tensor with actual storage
    Note: we'll also ignore initialization methods e.g xavier initialization for the params
    """

    _unpadded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _shard_param_infos: tuple[_ShardParamInfo, ...]
    _num_params: int
    _param_infos: tuple[ParamInfo, ...]
    _numels_with_padding: tuple[int, ...]
    _local_shard: torch.Tensor
    _full_param_padded: torch.Tensor
    _padded_unsharded_size: torch.Size

    # flat parameter is not a subclass of nn.Parameter, instead it creates a new nn.Parameter object with a new attribute _is_flat_param
    def __new__(cls, data=None, requires_grad=True):
        assert cls is FlatParameter, "subclasses FlatParameter not supported"
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)
        r._is_flat_param = True
        return r

    @classmethod
    def _init_metadata_(
        cls,
        self,
        param_infos,
        numels,
        shapes,
        strides,
        contiguities,
        fqns,
        params,
        is_padding_mask,
    ):

        # manually initialize attributes of FlatParameter
        assert len(param_infos) == len(shapes)
        assert len(param_infos) == len(strides)
        assert len(param_infos) == len(contiguities)
        assert len(param_infos) == len(fqns)
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._strides = strides
        self._contiguities = contiguities
        self._fqns = fqns
        self._is_padding_mask = is_padding_mask

        print(f"param_infos: {len(param_infos)}")
        print(f"params: {len(params)}")

        numels_without_padding: list[int] = []
        # Improvement: we can use an array to store numels and use boolean index with is_padding_mask to get the numels without padding
        for numel, is_padding in zip(numels, is_padding_mask):
            if not is_padding:
                numels_without_padding.append(numel)
        self._numels = tuple(numels_without_padding)
        self._numels_with_padding = tuple(numels)
        assert len(self._numels) == self._num_params

        self._modules = {pi.module for pi in self._param_infos}

        if params is not None:
            self._params = []
            for param, is_padding in zip(params, is_padding_mask):
                if not is_padding:
                    self._params.append(param)

            self._is_grad_none_mask = [False for _ in range(self._num_params)]
            self._tensors = [None for _ in range(self._num_params)]

            for param in self._params:
                _set_fsdp_flattened(param)
        else:
            self._params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
        _set_fsdp_flattened(self)
        self._post_backward_called = False


class FlatParameterHandle:
    """
    handles a viewing and sharding of a flat parameter
    """

    def __init__(
        self,
        params,
        fully_sharded_module,
        device,
        process_group,
        use_orig_params,
        sharding_strategy,
    ):
        params = list(params)
        self.device = device
        self._device_handle = _FSDPDeviceHandle.from_device(self.device)
        self.process_group = process_group
        self._use_orig_params = use_orig_params
        self._fully_sharded_module = fully_sharded_module
        self._handle_index = None
        self.param_dtype = params[0].dtype
        self.params = params
        self.rank = process_group.rank()
        self.world_size = process_group.size()

        self._training_state = HandleTrainingState.IDLE
        self._needs_pre_forward_unshard = False
        self._prefetched = False
        self._orig_param_dtype = params[0].dtype
        self._pre_forward_order_index = None
        self._sharding_strategy = sharding_strategy

        align_addresses = use_orig_params
        self._init_get_unflat_views_fn(align_addresses)
        # get aligned number
        # in pytorch they only use this when using orig params but alignment could still be beneficial for GPU computation efficiency even without original parameters
        self._aligned_numel = _get_aligned_numel(self.param_dtype)

        # Initialize flat parameter and it's metadata
        self._init_flat_param_and_metadata(
            params, fully_sharded_module, self._aligned_numel, use_orig_params
        )

    @property
    def uses_sharded_strategy(self):
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    def _setattr_param(self, module, param_name, param):
        if hasattr(module, param_name):
            delattr(module, param_name)
        setattr(module, param_name, param)

    def _check_sharded(self, tensor):
        msg_prefix = "Expects tensor to be sharded "
        assert tensor is not None, msg_prefix + "but got `None`"
        sharded_size = self.flat_param._sharded_size  # type: ignore[attr-defined]
        assert (
            tensor.size() == sharded_size,
            msg_prefix + f"with size {sharded_size} but got {tensor.size()}",
        )

    def _check_on_compute_device(self, tensor):
        assert (
            tensor.device == self.device,
            f"Expects tensor to be on the compute device {self.device}, was on {tensor.device}",
        )

    def _check_unsharded(self, tensor):
        msg_prefix = "Expects tensor to be unsharded "
        assert tensor is not None, msg_prefix + "but got `None`"
        unsharded_size = self.flat_param._unpadded_unsharded_size
        assert (
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()}",
        )

    def _init_flat_param_and_metadata(
        self, params, module, aligned_numel, use_orig_params
    ):

        if len(params) == 0:
            raise ValueError("No parameters provided")

        if aligned_numel < 0:
            raise ValueError(
                f"Invalid aligned number: {self._aligned_numel} for dtype: {self.param_dtype}"
            )

        param_dtype, param_requires_grad, param_device = self.validate_tensors()

        print(f"param_device: {param_device}")

        param_set = set(self.params)
        param_infos: list[ParamInfo] = []
        numels: list[int] = []
        shapes: list[torch.Size] = []
        strides: list[tuple[int, ...]] = []
        contiguities: list[bool] = []
        fqns: list[str] = []
        params_to_flatten: list[torch.Tensor] = []
        is_padding_mask: list[bool] = []
        total_numel = total_numel_without_padding = 0

        # since i'm not using nested modules, can I just get the module name and module object using module.named_modules()[0] or some other method instead of having this outer loop?
        # e.g nn.Linear(in_features, out_features, bias=True)
        # nn.Linear.weight and nn.Linear.bias are the parameters
        for module_name, module in module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if aligned_numel > 0:
                    # (total_numel % aligned_numel) how many elements are left after fitting all the elements in batches of aligned_numel
                    # numel_to_pad is then the number of elements needed to pad the remaining elements to be up to aligned_numel
                    # in the first iteration, total_numel is 0 so numel_to_pad is aligned_numel
                    numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                    if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                        padding_tensor = _construct_padding_tensor(
                            numel_to_pad, param_dtype, False, param_device
                        )
                        params_to_flatten.append(padding_tensor)
                        is_padding_mask.append(True)
                        numels.append(numel_to_pad)
                        total_numel += numel_to_pad
                params_to_flatten.append(param)
                is_padding_mask.append(False)
                param_infos.append(ParamInfo(param_name, module, module_name))
                numels.append(param.numel())
                shapes.append(param.shape)
                strides.append(param.stride())
                contiguities.append(_is_truly_contiguous(param))
                fqn = module_name + "." + param_name if module_name else param_name
                fqns.append(fqn)
                total_numel += param.numel()
                total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {self.params}\nmodule: {module}"
            )
        if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, param_dtype, False, param_device
                )
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        self.flat_param: FlatParameter = self.flatten_tensors(
            params_to_flatten, param_requires_grad
        )
        FlatParameter._init_metadata_(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            strides,
            contiguities,
            fqns,
            # FlatParameter should keep original parameter non-flat representation as well if use_orig_params is True
            _convert_to_params(params_to_flatten) if self._use_orig_params else None,
            is_padding_mask,
        )

    def _init_get_unflat_views_fn(self, align_addresses):
        self._get_unflat_views = (
            self._get_unflat_views_aligned
            if align_addresses
            else self._get_unflat_views_unaligned
        )

    # this basically means that when you are splitting a module into units one consideration you have to make is that parameters in a unit have the same dtype, device and requires_grad
    def validate_tensors(self):
        param_dtype = None
        param_requires_grad = None
        param_device = None

        for param in self.params:
            if isinstance(param, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if param_dtype is None and not param.is_floating_point():
                raise ValueError("Cannot flatten integer dtype tensors")
            if param_dtype is not None and param.dtype != param_dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {param_dtype} "
                    f"and {param.dtype}"
                )
            if (
                not self._use_orig_params
                and param_requires_grad is not None
                and param.requires_grad != param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if param_device is not None and param.device != param_device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{param_device} and {param.device}"
                )
            dtype = param.dtype
            param_requires_grad = param_requires_grad or param.requires_grad
            device = param.device
        assert param_requires_grad is not None, "Requires non-empty `tensors` list"
        return dtype, param_requires_grad, device

    def flatten_tensors(self, params_to_flatten, requires_grad):
        flat_tensors = torch.cat(
            [
                (
                    torch.flatten(_detach_if_needed(tensor))
                    if _is_truly_contiguous(tensor)
                    else _detach_if_needed(tensor).as_strided((tensor.numel(),), (1,))
                )
                for tensor in params_to_flatten
            ],
            dim=0,
        )

        return FlatParameter(flat_tensors, requires_grad=requires_grad)

    @torch.no_grad()
    def shard(self):
        flat_param = self.flat_param
        assert (
            flat_param.storage_offset() == 0
        ), f"FlatParameter is not the sole owner of the storage: {flat_param.storage_offset()}"
        rank = self.rank
        world_size = self.world_size

        chunks = (
            torch.flatten(flat_param).chunk(world_size)
            if _is_truly_contiguous(flat_param)
            else flat_param.as_strided((flat_param.numel(),), (1,)).chunk(world_size)
        )
        if len(chunks) < (rank + 1):
            # This rank gets an empty chunk fully padded with zeros since there
            # are not enough chunks across ranks
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        # I expect chunk to be perfectly split across ranks
        shard = chunk.clone()
        # free the memory occupied by the unsharded flat param
        allocated = flat_param._typed_storage()._size() > 0
        if allocated:
            flat_param._typed_storage()._resize_(0)
        flat_param.set_(shard)
        start_idx = shard.numel() * self.rank
        end_idx = shard.numel() * (self.rank + 1) - 1
        self._init_shard_metadata(start_idx, end_idx)
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(self, start_idx, end_idx):
        # self.flat_param should now contain the shard for this rank
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()

        assert (
            start_idx >= 0 and start_idx <= end_idx
        ), f"start_idx: {start_idx} end_idx: {end_idx}"

        shard_param_infos = self._get_shard_metadata(start_idx, end_idx)
        assert (
            len(shard_param_infos) == flat_param._num_params
        ), f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]

    def _get_unflat_views_aligned(
        self,
        tensor=None,
    ):
        """
        Return unflattened ``Tensor`` views into ``tensor`` with handling for padding.

        This method has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        splits = torch.split(tensor, flat_param._numels_with_padding, dim=0)
        idx = 0
        views = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views.append(
                (
                    split.view(flat_param._shapes[idx])
                    if flat_param._contiguities[idx]
                    else split.as_strided(
                        flat_param._shapes[idx], flat_param._strides[idx]
                    )
                )
            )

            idx += 1
        return views

    def _get_unflat_views_unaligned(
        self,
        tensor=None,
    ):
        """
        Return unflattened ``Tensor`` views into ``tensor``.

        If `tensor`` is ``None``,  ``flat_param`` is used. The unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        views = (
            (
                subtensor.view(shape)
                if contiguous
                else subtensor.as_strided(shape, stride)
            )
            for (subtensor, shape, stride, contiguous) in zip(
                torch.split(tensor, flat_param._numels, dim=0),
                flat_param._shapes,
                flat_param._strides,
                flat_param._contiguities,
                flat_param._param_extensions,
            )
        )
        return views

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ):
        """
        Compute the shard metadata based on ``unsharded_start_idx`` and ``unsharded_end_idx`` (inclusive).

        ``unsharded_start_idx`` and ``unsharded_end_idx`` give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        assert len(flat_param_offsets) == len(
            self.flat_param._numels_with_padding
        ), f"Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}"
        shard_param_infos = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1

        for (
            (unsharded_param_start_idx, unsharded_param_end_idx),
            is_padding,
        ) in zip(flat_param_offsets, self.flat_param._is_padding_mask):
            if is_padding:
                continue
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                assert (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ), (
                    f"Invalid `offset_in_shard` of {offset_in_shard} for "
                    f"sharded flat parameter with {sharded_flat_param_numel} numel"
                )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    def _get_flat_param_offsets(self) -> list[tuple[int, int]]:
        # accumulate returns an iterator that yields running sums of the input sequence
        # e.g. [1,2,3,4] -> [1, 1+2, 1+2+3, 1+2+3+4] -> [1,3,6,10]
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    def _get_padded_unsharded_flat_param(self):
        flat_param = self.flat_param
        return flat_param._full_param_padded

    def _use_sharded_views(self) -> None:
        """
        Set the original parameter variables' data to be flattened views into the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
        self._unsharded_flat_param_for_skipped_views = None
        flat_param = self.flat_param
        # Construct once and reuse for all parameters not in the local shard
        size_0_empty_tensor = torch.empty(
            0,
            dtype=self.flat_param.dtype,  # in case `flat_param` changed dtype
            device=self.flat_param.device,
            requires_grad=False,
        )
        for param, shard_param_info, (param_name, module, _) in zip(
            flat_param._params, flat_param._shard_param_infos, flat_param._param_infos
        ):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                # Allow the original data to be freed via garbage collection
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset : offset + numel_in_shard]

        # we've not implemented saving tensors yet
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            # Clear the saved `Tensor`s since they are unneeded now
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None

    def _use_unsharded_flat_param(
        self,
        padded_unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switch to use the *unpadded* unsharded flat parameter.

        This is a view into the *padded* unsharded flat parameter.
        """
        # first of all this attribute _unpadded_unsharded_size doesn't give the unpadded unsharded size of the flat param
        # also for the slicing of the flat_param_part, the padding is distributed across each actual param in flat param so i'm not sure the slicing gives the unpadded flat_param in the way that we intended.
        # one way to confirm this is to check that the size of padded_unsharded_flat_param is the same as the size of the flat_param_part

        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param_part = padded_unsharded_flat_param[: unsharded_size.numel()]
        # slicing [:] is not visible to autograd because of .data
        self.flat_param.data = flat_param_part
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if self._use_orig_params:
            if self._skipped_use_sharded_views and in_pre_backward:
                # This call corresponds to the complementary pre-backward
                # `_use_unsharded_views()` to the skipped pre-forward
                # `_use_sharded_views()`, so we should skip this one too.
                return
            # We use `Tensor` views in the forward so that they are tracked by
            # autograd. We use them in the pre-backward as well to support
            # reentrant activation checkpointing, which needs the views to be
            # tracked by autograd in the backward pass's recomputed forward.
            self._use_unsharded_views(
                as_params=(not in_forward and not in_pre_backward)
            )
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def _use_unsharded_views(self, as_params):
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()

        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, flat_param._param_infos)
        ):
            if self._use_orig_params and as_params:
                param = self.flat_param._params[i]
                self._setattr_param(module, param_name, param)
                param.data = view
            elif as_params:
                self._setattr_param(
                    module,
                    param_name,
                    nn.Parameter(view, requires_grad=flat_param.requires_grad),
                )
            else:  # `as_params=False`
                param_var = view
                if self._use_orig_params:
                    if self._training_state == HandleTrainingState.FORWARD:
                        # Save the `Tensor` for the pre-backward
                        self.flat_param._tensors[i] = view  # save for pre-backward
                    elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                        # Use the saved `Tensor` variable from the forward to
                        # preserve the autograd graph so that the post-backward
                        # hook fires (e.g. for reentrant AC)
                        tensor = self.flat_param._tensors[i]
                        tensor.data = view
                        param_var = tensor
                self._setattr_param(module, param_name, param_var)
                if (
                    self._use_orig_params
                    and self._training_state == HandleTrainingState.FORWARD
                ):
                    module._parameters[param_name] = param_var

    def _reset_is_grad_none(self):
        if not self._use_orig_params:
            return
        assert (
            self._training_state == HandleTrainingState.BACKWARD_POST
        ), "Expects to only be called in the post backward state after gradient computation"
        flat_param = self.flat_param
        assert flat_param._params is not None
        for i, param in enumerate(flat_param._params):
            if param.requires_grad:
                assert flat_param._is_grad_none_mask is not None
                flat_param._is_grad_none_mask[i] = False

    def needs_unshard(self) -> bool:
        """Return if the handle's flat parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        already_unsharded = _same_storage_size(
            unsharded_flat_param, unsharded_flat_param.numel()
        )
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocate the *padded* unsharded flat parameter.

        The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        assert self.uses_sharded_strategy, "Expected sharding strategy but got NO_SHARD"
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_freed(unsharded_flat_param)
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)
        return unsharded_flat_param

    def _check_storage_freed(self, tensor: torch.Tensor) -> None:
        """
        Check if the storage of the unsharded flat parameter is freed.
        """
        assert (
            _same_storage_size(tensor, 0),
            "Expected storage to be freed but got non-zero size",
        )

    def _all_gather_flat_param(
        self, padded_unsharded_flat_param: torch.Tensor
    ) -> torch.Tensor:

        sharded_flat_param = self.flat_param.data
        expected_numel = sharded_flat_param.numel() * self.world_size
        assert (
            padded_unsharded_flat_param.numel() == expected_numel
        ), f"Expected {expected_numel} but got {padded_unsharded_flat_param.numel()}"

        pg = self.process_group

        dist.all_gather_into_tensor(padded_unsharded_flat_param, sharded_flat_param, pg)

        return padded_unsharded_flat_param

    def _free_unsharded_flat_param(self):

        if self._uses_sharded_strategy:
            unsharded_flat_param = self.flat_param._full_param_padded
            self._check_on_compute_device(unsharded_flat_param)
            _no_dispatch_record_stream(
                unsharded_flat_param, self._device_handle.current_stream()
            )
            _free_storage(unsharded_flat_param)

    def _use_sharded_flat_param(self):
        flat_param = self.flat_param
        # note that we are never going to skip views because we are only using the fully sharded flat param
        flat_param.data = flat_param._local_shard
        if self._use_orig_params:
            self._use_sharded_views()

            if self._training_state == HandleTrainingState.FORWARD:

                accumulated_grad_in_no_sync = (
                    flat_param.grad is not None
                    and self.uses_sharded_strategy
                    and flat_param.grad.shape == flat_param._unpadded_unsharded_size
                )
                if accumulated_grad_in_no_sync:
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()

    def reshard(self, free_unsharded_flat_param):
        self._use_sharded_flat_param()
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()

    def unshard(self):

        if not self.needs_unshard():

            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)
        assert self.flat_param.device == self.device, "Expected device to be the same"

    def _prepare_gradient_for_backward(self):
        assert self._training_state in (
            HandleTrainingState.BACKWARD_PRE,
            HandleTrainingState.IDLE,
        ), "Expects to be in Backward pre or idle state"
        flat_param = self.flat_param
        if (
            flat_param.grad is not None
            and flat_param.grad.size() != flat_param._unpadded_unsharded_size
            or flat_param.grad.device != flat_param.device
        ):
            self._check_on_compute_device(self.flat_param)
            prev_iter_synced_gradients = (
                flat_param.grad.size() == flat_param._local_shard.size()
            )

            if prev_iter_synced_gradients:
                flat_param._saved_grad_shard = flat_param.grad.data
                sharded_grad = flat_param._saved_grad_shard

                local_shard_dtype = flat_param._local_shard.dtype

            else:
                padded_unsharded_size = flat_param._padded_unsharded_size
                assert (
                    flat_param.grad.size() == padded_unsharded_size
                ), "Expects .grad to be the unsharded gradient in no_sync with size {padded_unsharded_size} but got size {flat_param.grad.size()}"

            flat_param.grad = None

    def prepare_gradient_for_optim(self):

        def cast_grad_to_param_dtype(flat_param):
            if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                if self._use_orig_params:
                    self._use_sharded_grad_views()

        flat_param = self.flat_param

        if hasattr(flat_param, "_saved_grad_shard"):
            self._check_sharded(flat_param.grad)
            self._check_on_compute_device(flat_param)
            if flat_param._saved_grad_shard is not None:
                self._check_on_compute_device(flat_param._saved_grad_shard)
            if flat_param._post_backward_called:
                flat_param.grad = flat_param._saved_grad_shard
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype(flat_param)
        else:
            assert (
                not self.uses_sharded_strategy or not flat_param._post_backward_called
            ), "All sharded parameters that received a gradient in the post-backward should use `_saved_grad_shard`"

        # I suspect this is part of the gradient accumulation issue
        if hasattr(flat_param, "_saved_grad_shard"):
            delattr(flat_param, "_saved_grad_shard")

    @torch.no_grad()
    def _use_sharded_grad_views(self):
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        grad = self.sharded_grad
        if grad is None:
            for param in chain(flat_param._params, flat_param._sharded_params):
                param.grad = None
            return
        self._check_sharded(grad)
        for param, shard_param_info, is_grad_none in zip(
            flat_param._params,
            flat_param._shard_param_infos,
            flat_param._is_grad_none_mask,
        ):
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                if param.requires_grad and not is_grad_none:
                    offset = shard_param_info.offset_in_shard
                    param.grad = grad[offset : offset + numel_in_shard].reshape(
                        param.shape
                    )
                else:
                    param.grad = None

    def init_flat_param_attributes(self):
        flat_param = self.flat_param
        self._check_on_compute_device(self.flat_param)
        flat_param._local_shard = flat_param.data

        unsharded_param_dtype = flat_param.dtype
        padded_unsharded_numel = flat_param.numel() * self.world_size
        # empty does not mean free, it just means no meaningful values yet
        flat_param._full_param_padded = torch.empty(
            padded_unsharded_numel, device=self.device, dtype=unsharded_param_dtype
        )
        flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
        # free storage but keep tensor object
        _free_storage(flat_param._full_param_padded)

    def _reset_flat_param_grad_info_if_needed(self):
        if not self._use_orig_params:
            return

        flat_param = self.flat_param

        assert flat_param._params is not None

        all_grad_none = True
        requires_grad = False
        for param in flat_param._params:
            all_grad_none &= param.grad is None
            requires_grad |= param.requires_grad
        if all_grad_none:
            flat_param.grad = None

        flat_param.requires_grad = requires_grad
