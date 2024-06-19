import numpy as np
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from pytorch_tutorial.DCGAN.constants import dataset_name, results_root


def plot_real_and_fake_images(real_images, fake_images, device) -> None:
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_images.to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                fake_images.to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(f"{results_root}/{dataset_name}_real_vs_fake.png")
