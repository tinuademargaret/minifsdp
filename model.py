import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size):
    print(f"I am rank {rank} of {world_size}")


def init_process(rank, world_size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = 2
    processes = []

    mp.set_start_method("spawn")

    for rank in range(world_size):
        process = mp.Process(target=init_process, args=(rank, world_size, run))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()
