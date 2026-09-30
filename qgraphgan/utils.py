"""Logging, seeding and plotting helpers."""

import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def log(*args):
    print(*args)
    sys.stdout.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
