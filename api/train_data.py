import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10, ImageFolder
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dataset, line_image_dataset, image_size=96):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        image_data = torch.stack(
            [transform(x) for x, _ in image_dataset]
        )
        line_image_data = torch.stack(
            [transform(x) for x, _ in line_image_dataset]
        )
        self.data = torch.cat((image_data, line_image_data))
        self.n_data = len(self.data)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(
    image_path: str = "./data",
    line_image_path: str = "./data/line-images-folder",
    image_size: int = 96,
    batch_size: int = 64
) -> DataLoader:
    image_dataset = STL10(root=image_path, split="train")
    line_image_dataset = ImageFolder(root=line_image_path)
    dataset = ImageDataset(image_dataset, line_image_dataset, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
