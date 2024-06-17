import os

from pytorch_tutorial.GAN.generate import generate
from pytorch_tutorial.GAN.train import train


def GenerativeAdversarialNetworks() -> None:
    """
    モデルが存在する場合は画像を生成、存在しない場合はモデルを学習して画像を生成する
    """

    model_path = "./models/GAN/Generator_epoch_100.pth"
    print("Checking path:", model_path)
    print("Absolute path:", os.path.abspath(model_path))
    print("Exists:", os.path.exists(model_path))
    print("Current directory:", os.getcwd())

    if os.path.exists(model_path):
        # Pretrained model exists
        generate()
    else:
        # Pretrained model does not exist
        print("Pretrained model does not exist.")
        train()
        generate()


if __name__ == "__main__":
    GenerativeAdversarialNetworks()
