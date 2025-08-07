import os
import itertools
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.distributed_c10d import _get_default_group

from utils import (
    get_orig_params,
    _get_aligned_numel,
    _construct_padding_tensor,
    _is_truly_contiguous,
    _detach_if_needed,
    _convert_to_params,
)
from model import MLP, DataloaderLite, generate_dataset

PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class FlatParameter(nn.Parameter):
    """
    store initial shapes of each param in a unit, map?
    then convert to 1d tensor with actual storage
    Note: we'll also ignore initialization methods e.g xavier initialization for the params
    """

    # flat parameter is not a subclass of nn.Parameter, instead it creates a new nn.Parameter object with a new attribute _is_flat_param
    def __new__(cls, data=None, requires_grad=True):
        assert cls is FlatParameter, "subclasses FlatParameter not supported"
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)
        r._is_flat_param = True
        return r

    @classmethod
    def __init_metadata(
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
        else:
            self._params = None
            self._is_grad_none_mask = None
            self._tensors = None
        self._unpadded_unsharded_size = self.size()
        # Tracks whether the `FlatParameter`'s post-backward hook has been
        # called to modify the behavior of the post-backward callback
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
    ):
        self.device = device
        self.process_group = process_group
        self.use_orig_params = use_orig_params
        self._fully_sharded_module = fully_sharded_module
        self.param_dtype = params[0].dtype
        self.params = params

        align_addresses = use_orig_params
        # get aligned number
        self._aligned_numel = (
            _get_aligned_numel(self.param_dtype) if align_addresses else 0
        )

        # Initialize flat parameter and metadata
        self._init_flat_param_metadata()

    def _init_flat_param_metadata(self, module, aligned_numel):

        if self._aligned_numel < 0:
            raise ValueError(
                f"Invalid aligned number: {self._aligned_numel} for dtype: {self.param_dtype}"
            )

        param_dtype, param_requires_grad, param_device = self.validate_tensors()

        param_set = set(self.params)
        param_infos = []
        numels = []
        shapes = []
        strides = []
        contiguities = []
        fqns = []
        params_to_flatten = []
        is_padding_mask = []
        total_numel = total_numel_without_padding = 0

        for module_name, module in module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param not in param_set:
                    continue
                if aligned_numel > 0:
                    # (total_numel % aligned_numel) how many elements are left after fitting all the elements in batches of aligned_numel
                    # numel_to_pad is then the number of elements needed to pad the remaining elements to be up to aligned_numel
                    numel_to_pad = aligned_numel - (total_numel % aligned_numel)
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
        self.flat_param = self.flatten_tensors(params_to_flatten, param_requires_grad)
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            strides,
            contiguities,
            fqns,
            _convert_to_params(params_to_flatten) if self.use_orig_params else None,
            is_padding_mask,
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

    def shard(self):
        pass


class FSDP(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        use_orig_params=False,
        device_id=None,
        sync_module_states=False,
    ):
        """
        split module into units, 1 layer -> 1 unit, map?
        convert each unit into a flat parameter
        shard flat parameter across devices
        Note: let's limit ourselves to non nested modules
        Note: full sharding only
        Note:  the splitting into FSDP “units” that the paper talks about is exactly what auto_wrap_policy (and related wrapping logic) controls in the PyTorch codebase,
        when auto_wrap_policy is not given, FSDP often just wraps one module at a time in a layer-by-layer fashion
        Note: no support for traceable_wrapper_subclass tensors e.g QuantizedTensor
        Note: no mixed precision support
        NOte: no support for shared params
        """
        super().__init__()
        self.module = module
        self.device_id = device_id

        # get default group created by init_process_group
        self.process_group = _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.size()
        self.use_orig_params = use_orig_params
        self.training_state = "Idle"
        self.is_root = None
        self._exec_order_data = None
        self._fully_sharded_module_to_handle = {}
        self._handle = None
        self.params = []
        self.sync_module_states = sync_module_states

        # split module into units

        self._init_param_from_module()

    # this function would handle the materialization and sharding of each unit
    def _init_param_from_module(self):
        # check that all modules are initialized on the same device
        # check where module is initialized, meta, cpu, or gpu
        # Actually what I want to do here is to materialize a layer then shard it immediately. normally this is achieved via the auto_wrap_policy in pytorch, when there is no auto_wrap_policy the entire module is treated as a unit.
        params = get_orig_params(self.module)
        if len(params) == 0:
            raise ValueError("Module has no parameters")

        is_meta = any(param.device.type == "meta" for param in params)

        if is_meta:
            # materialize module if needed
            self.materialize_meta_module()

        device = {param.device.type for param in params}
        assert len(device) == 1

        # sync module states, so that all processes have the same module states, has some communication overhead, perform ablations on this, overhead vs accuracy
        if self.sync_module_states:
            module_states = []
            for param in params:
                # create a view of the param
                detached_param = param.detach()
                module_states.append(detached_param)
            # broadcast params to all processes in buckets of size PARAM_BROADCAST_BUCKET_SIZE, modifies tensors in place
            dist._broadcast_coalesced(
                self.process_group,
                module_states,
                src=0,
                bucket_size_bytes=PARAM_BROADCAST_BUCKET_SIZE,
            )

        # using flatparamhandle
        #   - flatten params
        #   - handle.shard
        print(f"Hello in FSDP from rank {self.rank} of {self.world_size}")

    def materialize_meta_module(self):
        #   - get device to materialize on
        materialization_device = self.device_id
        #   - get modules to materialize; we do not focus on using nested submodules in our model for demonstration
        modules_to_materialize = list(self.module.modules())
        try:
            #   - in no_grad context, for each module, move module to device and reset_parameters
            with torch.no_grad():
                for module in modules_to_materialize:
                    module_state_iter = itertools.chain(
                        module.parameters(recurse=False), module.buffers(recurse=False)
                    )
                    has_module_states = len(list(module_state_iter)) > 0
                    # skips modules with no activation function
                    if has_module_states:
                        module.to_empty(device=materialization_device, recurse=False)
                        module.reset_parameters()
        except Exception as e:
            raise e

    def forward(self):
        # root_pre_forward
        #   - set handles forward prefetch
        #   - set unshard stream for all gather and pre unshard stream for prefetch should wait for computation stream for optimization step
        #   - reset grad for flat params. if we use_orig_params, then .zero_grad clears orig_params grads which is just a view of unsharded flat param grad, and after resharding we'll dangling references into the unsharded flat params
        # pre_forward
        #   - If we are backward prefetching return; why?
        #   - record preforward in execution order object
        #   - set training state to forward
        #   - call unshard fn, ideally this should be the allgather operation
        #   - register post backward hook for reduce scatter and resharding
        #   - reallocate the _cpu_grad if optimizer overlap set the grad to None in the backward pass. why?
        #   - register_post_backward_reshard_only_hook, what's the difference with register post backward hook
        # call modules forward computation
        # post_forward
        #   - If we are in the baclward prefetching return
        #   - record post forward in execution order object
        #   - call reshard fn
        #   - register pre backward hook, why here and not in pre_forward?
        #   - set training state to idle, why?

        pass

    def print_module(self):
        for idx, m in enumerate(self.module.named_modules()):
            print(idx, "->", m)


def main(device_id):
    module = MLP().to(device="meta")
    fsdp_model = FSDP(module, device_id=device_id)
    fsdp_model.print_module()


def run(rank, world_size, device=None):
    print(f"Hello from rank {rank} of {world_size} of {device}")
    assert rank == dist.get_rank()
    assert world_size == dist.get_world_size()

    with torch.device("meta"):
        model = MLP()
    fsdp_model = FSDP(model)
    optimizer = torch.optim.SGD(model.parameters())
    mse_loss = nn.MSELoss()

    bsz = 50
    epochs = 100

    if rank == 0:
        x, y = generate_dataset()
    else:
        x = torch.empty((5000, 512))
        y = torch.empty((5000, 10))
    dist.broadcast(x, src=0)
    dist.broadcast(y, src=0)

    dataloader = DataloaderLite(bsz, x, y, rank, world_size)

    for i in range(epochs):
        for j in range(len(x) // (world_size * bsz)):
            optimizer.zero_grad()
            input, target = dataloader.next_batch()
            # input, target = input.to(device), target.to(device)
            output = fsdp_model(input)
            loss = mse_loss(output, target)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"epoch: {i}, loss: {loss.item()}")


def init_process(rank=None, world_size=None, fn=None, backend="gloo", cuda=False):
    if cuda:
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
        # fn(rank, world_size, device)
        main(device)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        try:
            fn(rank, world_size)
        except Exception as e:
            print(f"Error: {e}")
            dist.destroy_process_group()


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        init_process(fn=run, backend="nccl", cuda=True)
    else:
        world_size = 2
        processes = []

        mp.set_start_method("spawn")

        for rank in range(world_size):
            process = mp.Process(target=init_process, args=(rank, world_size, run))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
