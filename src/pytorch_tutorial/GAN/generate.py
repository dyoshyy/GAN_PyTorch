import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from pytorch_tutorial.GAN.model import generator


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
    G = generator().to(device)
    G.load_state_dict(torch.load("models/GAN/Generator_epoch_100.pth"))

    for i, _ in train_loader:
        print("real")
        plt.imshow(i[0][0].reshape(28, 28))
        plt.show()
        real_inputs = i[0][0]
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        print("fake")
        plt.imshow(fake_inputs[0][0].cpu().detach().numpy().reshape(28, 28))
        plt.show()
        break
