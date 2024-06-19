import os
import random

import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from torch import nn, optim

from pytorch_tutorial.DCGAN.constants import (
    beta1,
    data_root,
    dataset_name,
    lr,
    manualSeed,
    model_root,
    ngpu,
    num_epochs,
    results_root,
)
from pytorch_tutorial.DCGAN.dataset import load_data
from pytorch_tutorial.DCGAN.networks import Discriminator, Generator, weights_init
from pytorch_tutorial.DCGAN.train import train


def DeepConvolutionalGenerativeAdversarialNetworks() -> None:
    # make paths
    model_path = "./models/DCGAN/Generator_epoch_10.pth"
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    else:
        print("Dataset already exists.")
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
    )

    # check if model exists
    if not os.path.exists(model_path):
        print("Pretrained model does not exist.")
        # train
        # Create the dataloader
        dataloader = load_data()

        # Creating model and Handling multi-GPU
        if (device.type == "cuda") and (ngpu > 1):
            netG = nn.DataParallel(Generator(ngpu=ngpu).to(device), list(range(ngpu)))
            netD = nn.DataParallel(
                Discriminator(ngpu=ngpu).to(device), list(range(ngpu))
            )
        else:
            netG = nn.DataParallel(Generator(ngpu=ngpu).to(device))
            netD = nn.DataParallel(Discriminator(ngpu=ngpu).to(device))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netG.apply(weights_init)
        netD.apply(weights_init)

        # print the model
        print(netG)
        print(netD)

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Training the model
        G_loss, D_loss, img_list = train(
            num_epochs=num_epochs,
            dataloader=dataloader,
            netG=netG,
            netD=netD,
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            criterion=nn.BCELoss(),
            device=device,
        )

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [
            [plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list
        ]
        ani = animation.ArtistAnimation(
            fig, ims, interval=1000, repeat_delay=1000, blit=True
        )

        ani.save(
            f"{results_root}/{dataset_name}_progress.gif",
            writer=animation.PillowWriter(fps=1),
        )

    # generate fake images
    G = Generator(ngpu=ngpu).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()
