import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_dataset():
    # Parameters for synthetic dataset
    N = 5000  # number of samples (smaller for memory)
    D_in = 512  # input feature dimension
    D_out = 10  # output dimension
    noise_std = 0.01  # small Gaussian noise

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    # Generate random but fixed "true" weights and bias
    W_true = torch.randn(D_in, D_out)
    b_true = torch.randn(D_out)

    # Generate input data
    X = torch.randn(N, D_in)

    # Generate targets from the true mapping + noise
    y = X @ W_true + b_true + noise_std * torch.randn(N, D_out)

    return X, y


class DataloaderLite:

    def __init__(self, bsz, x, y, rank, world_size):
        self.bsz = bsz
        self.x = x
        self.y = y
        self.world_size = world_size
        self.rank = rank

        assert (
            len(self.x) % (self.world_size * self.bsz) == 0
        ), "data must be divisible by total batch size"

        self.curr_idx = rank * bsz

    def next_batch(self):
        x = self.x[self.curr_idx : self.curr_idx + self.bsz]
        y = self.y[self.curr_idx : self.curr_idx + self.bsz]

        self.curr_idx += self.world_size * self.bsz

        if self.curr_idx + (self.world_size * self.bsz):
            self.curr_idx = self.rank * self.bsz

        return x, y


def run(rank, world_size, device=None):
    print(f"Hello from rank {rank} of {world_size} of {device}")
    assert rank == dist.get_rank()
    assert world_size == dist.get_world_size()

    model = MLP()
    ddp_model = DDP(model)
    optimizer = optim.SGD(ddp_model.parameters())
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
            output = ddp_model(input)
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
        fn(rank, world_size)
        # except Exception as e:
        #     print(f"Error: {e}")
        #     dist.destroy_process_group()


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
