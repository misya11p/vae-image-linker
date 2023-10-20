from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision import transforms


def get_dataloader(
    data_root: str = "./data",
    image_size: int = 96,
    batch_size: int = 64,
) -> DataLoader:
    """
    Get dataloader for training.

    Args:
        data_root (str): Root directory of the dataset. Defaults to "./data".
        image_size (int): Size of the image. Defaults to 128.
        batch_size (int): Batch size. Defaults to 64.

    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomResizedCrop(size=image_size, scale=(0.3, 1.0)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    image_dataset = STL10(root=data_root, split="train", transform=transform)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
