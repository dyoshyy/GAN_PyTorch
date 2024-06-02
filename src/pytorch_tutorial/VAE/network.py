from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class VariationalAutoEncoder(nn.Module):  # type: ignore
    def __init__(self, z_dim) -> None:
        """
        Args:
            z_dim (int): 潜在変数の次元

        Returns:
            None

        Note:
            eps (float): オーバーフローとアンダーフローを防ぐ微小量
        """
        super(VariationalAutoEncoder, self).__init__()

        # Constants
        self.eps = np.spacing(1)
        self.x_dim = 28 * 28
        self.z_dim = z_dim

        # Encoder
        self.enc_fc1 = nn.Linear(self.x_dim, 400)
        self.enc_fc2 = nn.Linear(400, 200)
        self.enc_fc3_mean = nn.Linear(200, self.z_dim)  # 近似事後分布の平均
        self.enc_fc3_logvar = nn.Linear(200, self.z_dim)  # 近似事後分布の分散の対数

        # Decoder
        self.dec_fc1 = nn.Linear(self.z_dim, 200)
        self.dec_fc2 = nn.Linear(200, 400)
        self.dec_drop = nn.Dropout(p=0.2)  # 過学習を防ぐためにドロップアウト
        self.dec_fc3 = nn.Linear(400, self.x_dim)

    def encoder(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 入力データ
        """
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(
        self, mean: torch.Tensor, log_var: torch.Tensor, device: str
    ) -> torch.Tensor:
        """
        Args:
            mean (torch.Tensor): 近似事後分布の平均
            log_var (torch.Tensor): 近似事後分布の分散の対数
            device (str): デバイス

        Returns:
            torch.Tensor: 潜在変数
        """
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): 潜在変数

        Returns:
            torch.Tensor: 出力データ
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(
        self, x: torch.Tensor, device: str
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 入力データ
            device (str): デバイス

        Returns:
            KL (torch.float): KLダイバージェンス
            reconstruction (torch.float): 再構成誤差
            z (torch.tensor): (バッチサイズ, z_dim)サイズの潜在変数
            y (torch.tensor): (バッチサイズ, 入力次元数)サイズの再構成データ
        """
        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(
            1 + log_var - mean**2 - torch.exp(log_var)
        )  # KLダイバージェンス
        reconstruction = torch.sum(
            x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps)
        )  # 再構成誤差

        return [KL, reconstruction], z, y
