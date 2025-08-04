import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size):
    print(f"Hello from rank {rank} of {world_size}")
    assert rank == dist.get_rank()
    assert world_size == dist.get_world_size()


def init_process(rank=None, world_size=None, fn=None, backend="gloo", cuda=False):
    if cuda:
        dist.init_process_group(backend=backend)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        fn(rank, world_size)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        fn(rank, world_size)


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
