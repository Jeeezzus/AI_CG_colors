from __future__ import annotations

from torch.utils.data import Dataset
from pathlib import Path
import gzip
import torch
import numpy as np
from tqdm import tqdm
from typing import NamedTuple

class Triplet(NamedTuple):
    illustration: torch.Tensor
    lineart: torch.Tensor
    palette: torch.Tensor

@torch.jit.script
def post_lineart(x: torch.Tensor) -> torch.Tensor: return (x * 255).clip(0, 255).permute(1, 2, 0)
@torch.jit.script
def post_illustration(x: torch.Tensor) -> torch.Tensor: return (127.5 + 127.5 * x).clip(0, 255).permute(1, 2, 0)
@torch.jit.script
def pre_lineart(x: torch.Tensor) -> torch.Tensor: return 1.0 - x / 255
@torch.jit.script
def pre_illustration(x: torch.Tensor) -> torch.Tensor: return 2.0 * x / 255 - 1.0

class PaintsTorchDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.paths = list(path.glob("*.npy.gz"))
        
        def load(path: Path) -> np.ndarray:
            with gzip.open(path, "rb") as f:
                return np.load(f)
        self.buffers = list(map(load, tqdm(self.paths, desc="Loading")))

    def __len__(self) -> int: return len(self.buffers)
    def __getitem__(self, idx: int) -> Triplet:
        buffer = self.buffers[idx]
        H, W, C = buffer.shape
        y = np.random.randint(0, H - 512 + 1)
        x = np.random.randint(0, W - 512 + 1)
        tensor = torch.from_numpy(buffer).permute(2, 0, 1).float()
        tensor = tensor[:, y:y + 512, x:x + 512]
        return Triplet(
            pre_illustration(tensor[:3].float()),
            pre_lineart(tensor[3 + np.random.randint(0, 3), None].float()),
            pre_illustration(tensor[6:].float()),
        )