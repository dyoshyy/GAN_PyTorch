import argparse

from pytorch_tutorial.DCGAN import DeepConvolutionalGenerativeAdversarialNetworks
from pytorch_tutorial.GAN import GenerativeAdversarialNetworks
from pytorch_tutorial.VAE import VariationalAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a specific model.")
    parser.add_argument(
        "model",
        type=str,
        choices=["GAN", "VAE", "DCGAN"],
        help="The model to run: 'GAN', 'VAE', 'DCGAN'",
    )
    args = parser.parse_args()

    model = args.model

    match model:
        case "GAN":
            GenerativeAdversarialNetworks()
        case "VAE":
            VariationalAutoencoder()
        case "DCGAN":
            DeepConvolutionalGenerativeAdversarialNetworks()
        case _:
            raise ValueError(f"Unknown model: {model}")
