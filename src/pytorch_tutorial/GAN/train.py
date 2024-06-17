import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from pytorch_tutorial.GAN.model import discriminator, generator


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting hyper parameters
    epochs: int = 100
    lr: float = 2e-4
    batch_size: int = 64
    loss = nn.BCELoss()

    # calling the model
    G = generator().to(device)
    D = discriminator().to(device)

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # transform data to image
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # load data
    train_set = datasets.MNIST(
        "datasets/", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # training

    for epoch in range(1, epochs + 1):
        for idx, (imgs, _) in enumerate(train_loader):
            idx += 1

            # training discriminator

            # リアル画像をDに入力
            real_inputs = imgs.to(device)
            real_outputs = D(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            # ノイズからフェイクを生成
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            # リアル画像とフェイク画像のデータとラベルを結合
            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)

            # ロスの計算と最適化
            D_loss = loss(outputs, targets)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # training generator

            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)

            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = loss(fake_outputs, fake_targets)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            if idx % 100 == 0 or idx == len(train_loader):
                print(
                    f"Epoch {epoch} Iteration {idx}: discriminator_loss {D_loss.item()} generator_loss {G_loss.item()}"  # noqa: E501
                )
        if (epoch) % 10 == 0:
            model_dir = "models/GAN"
            # ディレクトリが存在しない場合は作成
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(G, f"{model_dir}/Generator_epoch_{epoch}.pth")
            print("Model saved.")
