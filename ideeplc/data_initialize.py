import platform
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from iDeepLC.ideeplc.utilities import df_to_matrix


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
        csv_path: str,
        batch_size: int = 96,
        **kwargs
) -> Union[Tuple[DataLoader, DataLoader, DataLoader, np.ndarray],
Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]]:
    """
    Initialize data loaders for prediction based on a CSV file containing raw peptide sequences.

    :param csv_path: Path to the CSV file containing raw peptide sequences.
    :param batch_size: Batch size for DataLoader.
    :return: DataLoader for prediction.
    """
    # Load peptides from CSV file
    df = pd.read_csv(csv_path)

    raw_peptides = df['peptide'].tolist()

    sequences, tr, predictions, errors = df_to_matrix(raw_peptides)

    prediction_dataset = MyDataset(sequences, tr)

    # Set the number of workers based on the platform
    workers = 0 if platform.system() == 'Windows' else 4

    # Create DataLoader objects
    dataloader_pred = DataLoader(prediction_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
                                 num_workers=workers)

    # passing the training X shape
    for batch in dataloader_pred:
        x_shape = batch[0].shape
        break

    return dataloader_pred, x_shape
