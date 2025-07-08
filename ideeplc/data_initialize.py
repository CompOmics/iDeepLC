import platform
import logging
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ideeplc.utilities import df_to_matrix

LOGGER = logging.getLogger(__name__)
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
        **kwargs
) -> Union[Tuple[MyDataset, np.ndarray], Tuple[MyDataset, np.ndarray]]:
    """
    Initialize peptides matrices based on a CSV file containing raw peptide sequences.

    :param csv_path: Path to the CSV file containing raw peptide sequences.
    :return: DataLoader for prediction.
    """

    LOGGER.info(f"Loading peptides from {csv_path}")
    try:
        # Load peptides from CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        LOGGER.error(f"File {csv_path} not found.")
        raise
    except pd.errors.EmptyDataError:
        LOGGER.error(f"File {csv_path} is empty.")
        raise
    except Exception as e:
        LOGGER.error(f"Error reading {csv_path}: {e}")
        raise

    if 'seq' not in df.columns:
        LOGGER.error(f"CSV file must contain a 'seq' column with peptide sequences.")
        raise ValueError("Missing 'seq' column in the CSV file.")

    raw_peptides = df['seq'].tolist()
    LOGGER.info(f"Loaded {len(raw_peptides)} peptides sequences from the file.")
    try:
        # Convert sequences to matrix format
        sequences, tr, errors = df_to_matrix(raw_peptides, df)
    except Exception as e:
        LOGGER.error(f"Error converting sequences to matrix format: {e}")
        raise
    if errors:
        LOGGER.warning(f"Errors encountered during conversion: {errors}")

    prediction_dataset = MyDataset(sequences, tr)

    # Create DataLoader objects
    dataloader_pred = DataLoader(prediction_dataset)
    # passing the training X shape
    for batch in dataloader_pred:
        x_shape = batch[0].shape
        break
    LOGGER.info(f"Dataset initialized with data shape {x_shape}.")
    return prediction_dataset, x_shape
