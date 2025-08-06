import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1024, 512, 2)

    def forward(self, x):
        return self.conv(x)


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
