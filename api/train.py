from argparse import ArgumentParser
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dlprog import train_progress
from model import VAE
from train_data import get_dataloader

prog = train_progress()


def main(
    image_size: int = 128,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    mag: int = 10,
    data_root: str = "data",
    save_path: str = "models/model.pth",
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataloader = get_dataloader(data_root, image_size, batch_size, mag)
    model = VAE(image_size, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("start training")
    train(model, optimizer, dataloader, device, n_epochs)
    model.save(save_path)
    print("finish training")


def loss_fn(x, y, mean, log_var):
    loss_recons = F.binary_cross_entropy(y, x, reduction='mean')
    loss_reg = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
    return loss_recons + loss_reg

def train(
    model: VAE,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    n_epochs: int = 30,
) -> None:
    model.train()
    prog.start(n_epochs=n_epochs, n_iter=len(dataloader))
    for _ in range(n_epochs):
        for x in dataloader:
            optimizer.zero_grad()
            x = x.to(device).squeeze()
            y, mean, log_var, _ = model(x)
            loss = loss_fn(x, y, mean, log_var)
            loss.backward()
            optimizer.step()
            prog.update(loss.item())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mag", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_path", type=str, default="models/model.pth")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(**vars(args))
