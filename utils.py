import torch


def get_orig_params(module):
    return list(module.parameters())
