import argparse

from pytorch_tutorial.GAN import GenerativeAdversarialNetworks
from pytorch_tutorial.VAE import VariationalAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a specific model.")
    parser.add_argument(
        "model",
        type=str,
        choices=["GAN", "VAE"],
        help="The model to run: 'GAN' or 'VAE'",
    )
    args = parser.parse_args()

    model = args.model

    if model == "GAN":
        GenerativeAdversarialNetworks()
    elif model == "VAE":
        VariationalAutoencoder()
    else:
        raise ValueError(f"Unknown model: {model}")
