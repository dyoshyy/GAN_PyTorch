import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from pytorch_tutorial.VAE.network import VariationalAutoEncoder


def train(dataloader_train, dataloader_valid) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VariationalAutoEncoder(2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1000

    loss_valid = 10**7
    loss_valid_min = 10**7

    num_no_improved = 0

    num_batch_train = 0
    num_batch_valid = 0

    writer = SummaryWriter(log_dir="./logs")

    for num_iter in range(num_epochs):
        # Train
        model.train()
        for x, t in dataloader_train:
            lower_bound, _, _ = model(x, device)
            loss = -sum(lower_bound)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss_train/KL",
                -lower_bound[0].cpu().detach().numpy(),
                num_iter + num_batch_train,
            )
            writer.add_scalar(
                "Loss_train/Reconst",
                -lower_bound[1].cpu().detach().numpy(),
                num_iter + num_batch_train,
            )
            num_batch_train += 1

        num_batch_train -= 1  # 次回のエポックで辻褄が合うように調整

        # Validation
        model.eval()
        loss = []
        for x, t in dataloader_valid:
            lower_bound, _, _ = model(x, device)
            loss.append(-sum(lower_bound).cpu().detach().numpy())
            writer.add_scalar(
                "Loss_valid/KL",
                -lower_bound[0].cpu().detach().numpy(),
                num_iter + num_batch_valid,
            )
            writer.add_scalar(
                "Loss_valid/Reconst",
                -lower_bound[1].cpu().detach().numpy(),
                num_iter + num_batch_valid,
            )
            num_batch_valid += 1

        num_batch_valid -= 1  # 次回のエポックで辻褄が合うように調整
        loss_valid = np.mean(loss)
        loss_valid_min = np.minimum(loss_valid, loss_valid_min)

        print(f"Epoch: {num_iter+1}, Loss: {loss_valid}, Min Loss: {loss_valid_min}")

        # 今までのlossの最小値よりも大きければカウントを増やす
        if loss_valid_min < loss_valid:
            num_no_improved += 1
            print(f"Num no improved: {num_no_improved}")
        else:
            num_no_improved = 0
            torch.save(model.state_dict(), f"./z_{model.z_dim}.pth")

        # 10回以上改善が見られなければ終了
        if num_no_improved > 10:
            break

    writer.close()
