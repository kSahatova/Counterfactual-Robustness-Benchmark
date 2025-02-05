import medmnist
from torch.utils.data import Dataset


def load_medmnist_data(
    data_flag: str = "pneumoniamnist",
    data_dir: str = "",
    download: bool = True,
    size: int = 224,
):
    """
    Loads train, val, and test datasets from MedMNIST
    """
    info = medmnist.INFO[data_flag]

    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(
        split="train",
        download=download,
        size=size,
        root=data_dir,
    )

    val_dataset = DataClass(
        split="val",
        download=download,
        size=size,
        root=data_dir,
    )

    test_dataset = DataClass(
        split="test",
        download=download,
        size=size,
        root=data_dir,
    )

    return train_dataset, val_dataset, test_dataset


class PneumoniaMNISTDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_flag = "pneumoniamnist"
        self.data = medmnist.PneumoniaMNIST(
            split=split, size=224, root=data_dir, mmap_mode="r"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.labels[idx]
        image = self.data.imgs[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label
