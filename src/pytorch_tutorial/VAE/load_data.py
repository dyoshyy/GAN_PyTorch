from typing import Tuple

import torch  # 機械学習フレームワークとしてpytorchを使用
import torch.utils.data
from torchvision import datasets, transforms


def load_data() -> (
    Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]
):
    """
    Returns:
        Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader
        ]: train, valid, testのデータローダー
    """
    # MNISTのデータをとってくるときに一次元化する前処理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )

    # trainデータとtestデータに分けてデータセットを取得
    dataset_train_valid = datasets.MNIST(
        "./datasets", train=True, download=True, transform=transform
    )
    dataset_test = datasets.MNIST(
        "./datasets", train=False, download=True, transform=transform
    )

    # trainデータの20%はvalidationデータとして利用
    size_train_valid = len(dataset_train_valid)  # 60000
    size_train = int(size_train_valid * 0.8)  # 48000
    size_valid = size_train_valid - size_train  # 12000
    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset_train_valid, [size_train, size_valid]
    )
    # 取得したデータセットをDataLoader化する
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1000, shuffle=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1000, shuffle=False
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1000, shuffle=False
    )

    return dataloader_train, dataloader_valid, dataloader_test
