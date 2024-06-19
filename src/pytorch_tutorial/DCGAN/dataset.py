import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms

from pytorch_tutorial.DCGAN.constants import batch_size, data_root, image_size, workers


def load_data() -> DataLoader:
    # Create the dataset
    dataset = dset.ImageFolder(
        root=data_root,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    return dataloader
