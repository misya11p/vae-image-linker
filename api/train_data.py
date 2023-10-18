import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10, ImageFolder
from torchvision import transforms


# class ImageDataset(Dataset):
#     def __init__(
#         self,
#         image_dataset: Dataset,
#         image_size: int,
#         mag: int
#     ):
#         transform = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.Lambda(lambda x: x.convert("RGB")),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomResizedCrop(size=image_size, scale=(0.3, 1.0)),
#             transforms.RandomRotation(90),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ToTensor()
#         ])
#         image_data = torch.stack(
#             [transform(x) for x, _ in image_dataset]
#         )
#         # line_image_data = torch.stack(
#         #     [transform(x) for x, _ in line_image_dataset] * mag
#         # )
#         # self.data = torch.cat((image_data, line_image_data))
#         self.data = image_data
#         self.n_data = len(self.data)

#     def __len__(self):
#         return self.n_data

#     def __getitem__(self, idx):
#         return self.data[idx]


def get_dataloader(
    data_root: str = "./data",
    image_size: int = 128,
    batch_size: int = 64,
    mag: int = 1
) -> DataLoader:
    """
    Get dataloader for training.

    Args:
        data_root (str): Root directory of the dataset. Defaults to "./data".
        image_size (int): Size of the image. Defaults to 128.
        batch_size (int): Batch size. Defaults to 64.
        mag (int): Magnification of the line image. Defaults to 1.

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
    # line_image_path = os.path.join(data_root, "line-images-folder")
    image_dataset = STL10(root=data_root, split="train", transform=transform)
    # line_image_dataset = ImageFolder(root=line_image_path)
    # dataset = ImageDataset(
    #     image_dataset,
    #     # line_image_dataset,
    #     image_size,
    #     mag
    # )
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return dataloader
