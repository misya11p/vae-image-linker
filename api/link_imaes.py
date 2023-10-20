from typing import List
import base64
import io
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import VAE


def link_images(
    image1: str,
    image2: str,
    config: dict
) -> List[str]:
    """

    Args:
        image1 (str): base64 encoded image
        image2 (str): base64 encoded image
        n_frames (int): number of frames to generate
        config (dict): configuration dictionary

    Returns:
        List[str]: list of base64 encoded images
    """
    model_path = config["model_path"]
    n_frames = config["n_frames"]
    device = config["device"]
    image_size = config["image_size"]
    model = VAE(3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    x = get_input(image1, image2, image_size)
    z, _, _ = model.encoder(x)
    z1, z2 = z.chunk(2, dim=0)
    z = linear_complement(z1, z2, n_frames)
    y = model.decoder(z)
    save_images(y)
    images = decode_images(y)
    return images

def linear_complement(x1, x2, n):
    x1 = x1.detach().numpy().squeeze()
    x2 = x2.detach().numpy().squeeze()
    x = torch.tensor(np.linspace(x1, x2, n))
    return x


def get_input(image1: str, image2: str, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    img1 = b64_to_image(image1)
    img2 = b64_to_image(image2)
    img1 = transform(img1)
    img2 = transform(img2)
    x = torch.stack([img1, img2])
    return x

def b64_to_image(image: str) -> Image:
    image = image.split(',')[1] if 'base64,' in image else image
    image = base64.b64decode(image)
    image = np.frombuffer(image, dtype=np.uint8)
    image = Image.open(io.BytesIO(image))
    return image


def decode_images(images: torch.Tensor) -> List[str]:
    images = images.detach().cpu()
    images = [tensor_to_b64(image) for image in images]
    return images

to_pil = transforms.ToPILImage()
def tensor_to_b64(image: torch.Tensor) -> str:
    buffer = io.BytesIO()
    image = to_pil(image)
    image.save(buffer, format="PNG")
    image_binary = buffer.getvalue()
    image_b64 = base64.b64encode(image_binary)
    image_b64 = image_b64.decode("utf-8")
    return image_b64


def save_images(images: torch.Tensor) -> None:
    images = images.detach().cpu()
    images = [to_pil(image) for image in images]
    for i, image in enumerate(images):
        image.save(f"images/{i}.png")
