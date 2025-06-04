import platform
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import utilities


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
        eval_type: str,
        dataset_name: Optional[str] = None,
        test_aa: Optional[str] = None,
        batch_size: int = 96,
        **kwargs
) -> Union[Tuple[DataLoader, DataLoader, DataLoader, np.ndarray],
Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]]:
    """
    Initialize data loaders for training, validation, and testing based on the evaluation type.

    :param eval_type: The type of evaluation ('20datasets', 'ptm', 'aa_glycine').
    :param dataset_name: The specific dataset name to use, only required for aa_glycine.
    :param test_aa: The test amino acid, only required for aa_glycine.
    :param batch_size: The batch size for the DataLoader.
    :param kwargs: Additional arguments that might be needed for specific datasets.
    :return: The DataLoader objects for training, validation, and testing, and the shape of the training data.
    """


    if eval_type == "20datasets":
        dataset_path = f"../data/20_datasets_evaluation/{dataset_name}/"
        train_x = np.load(dataset_path + 'train_x.npy')
        train_y = np.load(dataset_path + 'train_y.npy')
        val_x = np.load(dataset_path + 'val_x.npy')
        val_y = np.load(dataset_path + 'val_y.npy')
        test_x = np.load(dataset_path + 'test_x.npy')
        test_y = np.load(dataset_path + 'test_y.npy')

    elif eval_type == "ptm":
        dataset_path = f"../data/PTM_evaluation/{dataset_name}/"
        train_x = np.load(dataset_path + 'train_x.npy')
        train_y = np.load(dataset_path + 'train_y.npy')
        val_x = np.load(dataset_path + 'val_x.npy')
        val_y = np.load(dataset_path + 'val_y.npy')
        test_x = np.load(dataset_path + 'test_x.npy')
        test_y = np.load(dataset_path + 'test_y.npy')
        test_no_mod_x = np.load(dataset_path + 'test_no_mod_x.npy')
        test_no_mod_y = np.load(dataset_path + 'test_no_mod_y.npy')

    elif eval_type == "aa_glycine":
        dataset_path = f"../data/modified_glycine_evaluation/{dataset_name}_{test_aa}_"
        df_train = pd.read_csv(f'{dataset_path}train.csv', keep_default_na=False)
        df_val = pd.read_csv(f'{dataset_path}valid.csv', keep_default_na=False)
        df_test = pd.read_csv(f'{dataset_path}test.csv', keep_default_na=False)
        df_test_g = df_test.copy()
        # Replace the amino acid with glycine
        df_test_g['seq'] = df_test_g['seq'].str.replace(test_aa, 'G')

        # Reform the sequences and modifications into the standard format
        reformed_train = [utilities.reform_seq(df_train['seq'][i], m) for i, m in enumerate(df_train['modifications'])]
        reformed_val = [utilities.reform_seq(df_val['seq'][i], m) for i, m in enumerate(df_val['modifications'])]
        reformed_test = [utilities.reform_seq(df_test['seq'][i], m) for i, m in enumerate(df_test['modifications'])]
        reformed_test_g = [utilities.reform_seq_ignore_mod(df_test_g['seq'][i], m, test_aa) for i, m in
                           enumerate(df_test_g['modifications'])]

        # Convert the sequences into matrices
        train_x, train_y, train_prediction, train_error = utilities.df_to_matrix(reformed_train, df_train)
        val_x, val_y, val_prediction, val_error = utilities.df_to_matrix(reformed_val, df_val)
        test_x, test_y, test_prediction, test_error = utilities.df_to_matrix(reformed_test, df_test)
        test_g_x, test_g_y, test_g_prediction, test_g_error = utilities.df_to_matrix(reformed_test_g, df_test_g)
    else:
        raise ValueError(f"Unsupported evaluation type: {eval_type}")

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

    if eval_type == 'ptm':
        # For 'ptm', also create the test_no_mod DataLoader
        test_no_mod_dataset = MyDataset(test_no_mod_x, test_no_mod_y)
        dataloader_test_no_mod = DataLoader(test_no_mod_dataset, shuffle=False, batch_size=batch_size,
                                            pin_memory=True, num_workers=workers)
        return dataloader_train, dataloader_val, dataloader_test, dataloader_test_no_mod, x_shape
    elif eval_type == "aa_glycine":
        # For 'aa_glycine', also create the test_g DataLoader
        test_g_dataset = MyDataset(test_g_x, test_g_y)
        dataloader_test_g = DataLoader(test_g_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
                                       num_workers=workers)
        return dataloader_train, dataloader_val, dataloader_test, dataloader_test_g, x_shape

    return dataloader_train, dataloader_val, dataloader_test, x_shape
