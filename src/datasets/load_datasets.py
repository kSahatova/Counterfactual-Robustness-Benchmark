import medmnist
import numpy as np
from typing import Any

from torch.utils.data import Dataset

from imblearn.under_sampling import RandomUnderSampler
from src.utils.logger import setup_logger


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


class MedMNISTDataset(Dataset):
    """
    A customization of the MedMNIST dataset that allows integration of advanced transformations
    """

    def __init__(
        self,
        data_flag: str = "pneumoniamnist",
        size: int = 224,
        data_dir: str = "",
        split: str = "train",
        transform: Any = None,
        download: bool = True,
        increase_channels: bool = False,
        undersample: bool = True,
    ):
        self.logger = setup_logger(__name__)
        info = medmnist.INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])

        self.img_size = size
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.increase_channels = increase_channels
        self.undersample = undersample

        self.data = DataClass(
            split=split, size=size, root=data_dir, download=download, mmap_mode="r"
        )

        if self.undersample:
            self._undersample()

    def _add_channels_dimension(self, image):
        """Adds extra dimension for channels since MedMNIST provides 3 dimensional array for images"""
        if (len(image.shape) < 4) and isinstance(image, np.ndarray):
            return np.expand_dims(image, axis=0)
        else:
            self.logger.error(
                "Provided type of image is not np.ndarray. The corresponding implementation for the input data type is required."
            )

    def _increase_channels(self, image):
        """Increase the number of channels from 1 to 3"""
        if self.increase_channels:
            return np.tile(image, [3, 1, 1])
        else:
            return image
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns an image and label for the given index"""
        label = self.data.labels[idx]

        metadata = {
            "bbox": [0.15, 0.3, 0.9, 0.4],
            "text": "R",
        }

        image = self.data.imgs[idx]
        augmented = self.transform(image=image, textimage_metadata=metadata)["image"]
        image = self._add_channels_dimension(augmented)
        image = self._increase_channels(image)

        return image, label

    def _undersample(self):
        """Undersample the dataset to balance the classes."""

        rus = RandomUnderSampler(sampling_strategy=0.7, random_state=0)

        reshaped_images = np.reshape(self.data.imgs, (self.data.imgs.shape[0], -1))
        images, labels = rus.fit_resample(reshaped_images, self.data.labels)
        images = images.reshape(images.shape[0], self.img_size, self.img_size)

        if len(labels.shape) < 2:
            labels = np.expand_dims(labels, axis=-1)

        self.data.imgs = images
        self.data.labels = labels

        self.data.info["n_samples"][self.split] = images.shape[0]
