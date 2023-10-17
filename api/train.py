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
    device: str = "auto",
    data_root: str = "./data",
    image_size: int = 128,
    save_path: str = "model.pth",
    n_epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 64,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataloader = get_dataloader(data_root, image_size, batch_size)
    model = VAE(image_size, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("start training")
    train(model, optimizer, dataloader, device, n_epochs)
    model.save(save_path)
    print("finish training")


def loss_fn(x, y, mean, log_var):
    loss_recons = F.binary_cross_entropy(y, x, reduction='sum')
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
            x = x.to(device)
            y, mean, log_var, _ = model(x)
            loss = loss_fn(x, y, mean, log_var)
            loss.backward()
            optimizer.step()
            prog.update(loss.item())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="model.pth")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(**vars(args))
