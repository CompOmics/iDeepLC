import platform
from typing import Tuple, Optional, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader


# Making the pytorch dataset
class MyDataset(Dataset):
    def __init__(self, sequences: np.ndarray, retention: np.ndarray) -> None:
        self.sequences = sequences
        self.retention = retention

    def __len__(self) -> int:
        return len(self.retention)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx], self.retention[idx]


def data_initialize(
        dataset_name: Optional[str] = None,
        batch_size: int = 96,
        **kwargs
) -> Union[Tuple[DataLoader, DataLoader, DataLoader, np.ndarray],
Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]]:
    """
    Initialize data loaders for training, and validation.

    :param eval_type: The type of evaluation ('20datasets', 'ptm', 'aa_glycine').
    :param dataset_name: The specific dataset name to use, only required for aa_glycine.
    :param test_aa: The test amino acid, only required for aa_glycine.
    :param batch_size: The batch size for the DataLoader.
    :param kwargs: Additional arguments that might be needed for specific datasets.
    :return: The DataLoader objects for training, and validation, and the shape of the training data.
    """

    dataset_path = f"../data/matrices/{dataset_name}/"
    train_x = np.load(dataset_path + 'train_x.npy')
    train_y = np.load(dataset_path + 'train_y.npy')
    val_x = np.load(dataset_path + 'val_x.npy')
    val_y = np.load(dataset_path + 'val_y.npy')
    test_x = np.load(dataset_path + 'test_x.npy')
    test_y = np.load(dataset_path + 'test_y.npy')

    # Create the PyTorch Dataset objects
    train_dataset = MyDataset(train_x, train_y)
    val_dataset = MyDataset(val_x, val_y)
    test_dataset = MyDataset(test_x, test_y)

    # Set the number of workers based on the platform
    workers = 0 if platform.system() == 'Windows' else 4

    # TODO: check if batch_size works for val and test
    # Create DataLoader objects
    dataloader_train = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                  num_workers=workers)
    dataloader_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=workers)
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
                                 num_workers=workers)

    # passing the training X shape
    for batch in dataloader_train:
        x_shape = batch[0].shape
        break

    return dataloader_train, dataloader_val, dataloader_test, x_shape
