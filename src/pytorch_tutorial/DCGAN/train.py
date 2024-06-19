import torch
import torchvision.utils as vutils

from pytorch_tutorial.DCGAN.constants import fake_label, fixed_noise, nz, real_label


def train(  # noqa: PLR0913
    num_epochs: int,
    dataloader: torch.utils.data.DataLoader,
    netG: torch.nn.Module,
    netD: torch.nn.Module,
    optimizerG: torch.optim.Optimizer,
    optimizerD: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[list[float], list[float], list[torch.Tensor]]:
    """
    Trains the DCGAN model.

    Args:
      num_epochs (int): The number of training epochs.
      dataloader (torch.utils.data.DataLoader): The data loader for loading the
      training data.
      netG (torch.nn.Module): The generator network.
      netD (torch.nn.Module): The discriminator network.
      optimizerG (torch.optim.Optimizer): The optimizer for the generator network.
      optimizerD (torch.optim.Optimizer): The optimizer for the discriminator network.
      criterion (torch.nn.Module): The loss function.
      device (torch.device): The device to run the training on.

    Returns:
      tuple[list[float], list[float], list[torch.Tensor]]:
      A tuple containing the generator losses,discriminator losses,
      and a list of generated images.
    """
    G_losses = []
    D_losses = []
    iters = 0
    img_list = []

    # Change the model to training mode
    netG.train()
    netD.train()

    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            # classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D,
            # perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t"
                    f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t"
                    f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if (epoch) % 5 == 0:
            torch.save(netG.state_dict(), f"./models/DCGAN/Generator_epoch_{epoch}.pth")
            print("Model saved.")

    return G_losses, D_losses, img_list
