import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def generate() -> None:
    """
    Generate and display real and fake images using a GAN.

    Args:
      input (list): List of input data.

    Returns:
      None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # load data
    batch_size: int = 64
    train_set = datasets.MNIST(
        "./datasets", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # G = generator().to(device)
    # G.load_state_dict(torch.load("./models/GAN/Generator_epoch_100.pth"))
    G = torch.load("./models/GAN/Generator_epoch_100.pth")

    if not os.path.exists("./results/GAN"):
        os.makedirs("./results/GAN")

    idx = 0

    for i, _ in train_loader:
        idx += 1
        plt.imsave(
            f"results/GAN/real_{idx}.png",
            i[0][0].cpu().detach().numpy().reshape(28, 28),
        )
        real_inputs = i[0][0]
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        plt.imsave(
            f"results/GAN/fake_{idx}.png",
            fake_inputs[0][0].cpu().detach().numpy().reshape(28, 28),
        )
