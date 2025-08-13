import os
import itertools
from itertools import accumulate
from typing import NamedTuple, Optional
from enum import auto, Enum

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
    is_param_sync,
    is_flattened,
    _same_storage_size,
)
from runtime_utils import HandleTrainingState
from model import MLP, DataloaderLite, generate_dataset
from _flat_param import FlatParameterHandle, HandleShardingStrategy

PARAM_BROADCAST_BUCKET_SIZE = int(250 * 1024 * 1024)


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
        Note: no support for shared params
        Note: no cpu offloading
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

        self._fsdp_wrapped_module = module

    # this function would handle the materialization and sharding of each unit
    def _init_param_from_module(self):
        # check that all modules are initialized on the same device
        # check where module is initialized, meta, cpu, or gpu
        # Actually what I want to do here is to materialize a layer then shard it immediately. normally this is achieved via the auto_wrap_policy in pytorch, when there is no auto_wrap_policy the entire module is treated as a unit.

        # I will change this to use the module.named_parameters() if it turns out there are no additional fn in get_orig_params
        params = get_orig_params(self.module)
        if len(params) == 0:
            raise ValueError("Module has no parameters")

        is_meta = any(param.device.type == "meta" for param in params)

        if is_meta:
            # materialize module if needed
            self.materialize_meta_module()

        # we need to update params to the initialized
        params = get_orig_params(self.module)

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
            assert is_param_sync(self.module, self.rank), "Parameters are not synced"

        handle = FlatParameterHandle(
            params,
            self.module,
            self.device_id,
            self.process_group,
            self.use_orig_params,
        )
        handle.shard()
        assert not self._handle, "FSDP already initialized"
        self.params.append(handle.flat_param)
        self._handle = handle
        self._fully_sharded_module_to_handle[handle._fully_sharded_module] = handle
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

    def forward(self, *args, **kwargs):
        handle = self._handle
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
    # deffered initialization of the model
    module = MLP().to(device="meta")
    fsdp_model = FSDP(module, device_id=device_id, use_orig_params=True)
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
        # try:
        fn(device)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     dist.destroy_process_group()
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
        init_process(fn=main, backend="nccl", cuda=True)
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
