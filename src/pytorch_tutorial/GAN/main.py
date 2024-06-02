import os

from pytorch_tutorial.GAN.generate import generate
from pytorch_tutorial.GAN.train import train


def GenerativeAdversarialNetworks() -> None:
    model_path = "/models/GAN/Generator_epoch_100.pth"

    if os.path.exists(model_path):
        # Pretrained model exists
        generate()
    else:
        # Pretrained model does not exist
        train()
        generate()
