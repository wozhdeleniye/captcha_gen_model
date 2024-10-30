import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from data_preparation import make_dataset
import const


class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_dataset():
    image_paths, labels = make_dataset()
    print(image_paths.shape, labels.shape)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CaptchaDataset(image_paths=image_paths, labels=labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=const.batch_size, shuffle=True)

    return dataloader
