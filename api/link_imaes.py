from typing import List
import torch
from .model import VAE


def link_images(image1, image2, n_frames: int, config: dict) -> List[str]:
    model_path = config["model_path"]
    device = config["device"]
    model = VAE.load(model_path, device)
    x = get_input(image1, image2)
    z1, z2 = model.encode(x).chunk(2, dim=1)
    y = get_frames(model, z1, z2, n_frames)
    return y


def get_input(image1, image2) -> torch.Tensor:
    return

def get_frames(
    model: VAE,
    z1: torch.Tensor,
    z2: torch.Tensor,
    n_frames: int
) -> torch.Tensor:
    z = linear_complement(z1, z2, n_frames)
    y = model.decode(z)
    return y


def linear_complement(x1, x2, n):
    xs = [torch.linspace(s, e, n) for s, e in zip(x1, x2)]
    x = torch.stack(xs, dim=1).T
    return x
    