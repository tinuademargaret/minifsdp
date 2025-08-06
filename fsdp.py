import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.distributed_c10d import _get_default_group


from model import MLP, DataloaderLite, generate_dataset


class FlatParameter(nn.Parameter):

    def __init__(self, unit):
        """
        store initial shapes of each param in a unit, map?
        then convert to 1d tensor with actual storage
        Note: we'll also ignore initialization methods e.g xavier initialization for the params
        """

        pass


class FlatParameterHandle:
    """
    handles a viewing and sharding of a flat parameter
    """

    def __init__(self):
        # get aligned number
        # Initialize flat parameter and metadata
        pass

    def get_aligned_number(self):
        pass

    def validate_tensors(self):
        pass

    def flatten_tensors(self):
        pass

    def shard(self):
        pass


class FSDP(nn.Module):

    def __init__(self, module: nn.Module):
        """
        split module into units, 1 layer -> 1 unit, map?
        convert each unit into a flat parameter
        shard flat parameter across devices
        Note: let's limit ourselves to non nested modules
        Note: full sharding only
        Note:  the splitting into FSDP “units” that the paper talks about is exactly what auto_wrap_policy (and related wrapping logic) controls in the PyTorch codebase,
        when auto_wrap_policy is not given, FSDP often just wraps one module at a time in a layer-by-layer fashion
        """
        super().__init__()
        self.module = module

        # get default group created by init_process_group
        self.process_group = _get_default_group()
        self.rank = self.process_group.rank()
        self.world_size = self.process_group.world_size()

        self.__init_param_from_module()

    def _init_param_from_module(self):

        # check where module is initialized
        # materialize module if needed
        # get modules to materialize;
        # get device to materialize on
        # in no_grad context, for each module, move module to device and reset_parameters
        # sync module states
        # using flatparamhandle to flatten params
        # handle.shard
        print(f"Hello in FSDP from rank {self.rank} of {self.world_size}")

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


def main():
    module = MLP().to(device="meta")
    fsdp_model = FSDP(module)
    fsdp_model.print_module()


def run(rank, world_size, device=None):
    print(f"Hello from rank {rank} of {world_size} of {device}")
    assert rank == dist.get_rank()
    assert world_size == dist.get_world_size()

    with torch.devic("meta"):
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
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{rank}"
        fn(rank, world_size, device)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12340"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # try:
        # fn(rank, world_size)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     dist.destroy_process_group()


if __name__ == "__main__":
    # cuda = torch.cuda.is_available()
    # if cuda:
    #     init_process(fn=run, backend="nccl", cuda=True)
    # else:
    world_size = 2
    processes = []

    mp.set_start_method("spawn")

    for rank in range(world_size):
        process = mp.Process(target=init_process, args=(rank, world_size, run))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

    main()
