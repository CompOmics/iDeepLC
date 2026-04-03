import logging
from typing import Tuple, Union, Iterator
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from ideeplc.utilities import df_to_matrix, reform_seq

LOGGER = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, sequences: np.ndarray, retention: np.ndarray) -> None:
        self.sequences = sequences
        self.retention = retention

    def __len__(self) -> int:
        return len(self.retention)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sequences[idx], self.retention[idx]


def data_initialize(
    csv_path: str, **kwargs
) -> Union[Tuple[MyDataset, np.ndarray], Tuple[MyDataset, np.ndarray]]:
    """
    Initialize peptide matrices based on a CSV file containing raw peptide sequences.

    :param csv_path: Path to the CSV file containing raw peptide sequences.
    :return: Dataset for prediction or fine-tuning and x_shape.
    """
    LOGGER.info(f"Loading peptides from {csv_path}")

    try:
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

    if "seq" not in df.columns:
        LOGGER.error("CSV file must contain a 'seq' column with peptide sequences.")
        raise ValueError("Missing 'seq' column in the CSV file.")
    if "modifications" not in df.columns:
        LOGGER.error(
            "CSV file must contain a 'modifications' column with peptide modifications."
        )
        raise ValueError("Missing 'modifications' column in the CSV file.")
    if "tr" not in df.columns:
        LOGGER.error("CSV file must contain a 'tr' column with retention times.")
        raise ValueError("Missing 'tr' column in the CSV file.")

    reformed_peptides = [
        reform_seq(seq, mod) for seq, mod in zip(df["seq"], df["modifications"])
    ]
    LOGGER.info(
        f"Loaded and reformed {len(reformed_peptides)} peptides sequences from the file."
    )

    try:
        sequences, tr, errors = df_to_matrix(
            reformed_peptides,
            df,
            mod_features_csv=kwargs.get("mod_features_csv"),
        )

    except Exception as e:
        LOGGER.error(f"Error converting sequences to matrix format: {e}")
        raise

    if errors:
        LOGGER.warning(f"Errors encountered during conversion: {errors}")

    prediction_dataset = MyDataset(sequences, tr)

    if len(prediction_dataset) == 0:
        LOGGER.error("No valid peptide entries were found in the input file.")
        raise ValueError("No valid peptide entries were found in the input file.")

    # Keep historical x_shape contract expected by model/tests: (batch, channels, length)
    x_shape = (1,) + prediction_dataset[0][0].shape
    LOGGER.info(f"Dataset initialized with data shape {x_shape}.")
    return prediction_dataset, x_shape


def data_initialize_chunked(
    csv_path: str, chunk_size: int = 10000, **kwargs
) -> Iterator[Tuple[pd.DataFrame, MyDataset, np.ndarray]]:
    """
    Initialize peptide matrices from a CSV file in chunks.

    :param csv_path: Path to the CSV file containing raw peptide sequences.
    :param chunk_size: Number of rows to load per chunk.
    :return: Iterator yielding dataframe chunk, dataset chunk, and x_shape.
    """
    LOGGER.info(f"Loading peptides from {csv_path} in chunks of {chunk_size}")

    try:
        chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    except FileNotFoundError:
        LOGGER.error(f"File {csv_path} not found.")
        raise
    except pd.errors.EmptyDataError:
        LOGGER.error(f"File {csv_path} is empty.")
        raise
    except Exception as e:
        LOGGER.error(f"Error reading {csv_path}: {e}")
        raise

    for chunk_idx, df in enumerate(chunk_iter, start=1):
        if "seq" not in df.columns:
            LOGGER.error("CSV file must contain a 'seq' column with peptide sequences.")
            raise ValueError("Missing 'seq' column in the CSV file.")
        if "modifications" not in df.columns:
            LOGGER.error(
                "CSV file must contain a 'modifications' column with peptide modifications."
            )
            raise ValueError("Missing 'modifications' column in the CSV file.")
        if "tr" not in df.columns:
            LOGGER.error("CSV file must contain a 'tr' column with retention times.")
            raise ValueError("Missing 'tr' column in the CSV file.")

        reformed_peptides = [
            reform_seq(seq, mod) for seq, mod in zip(df["seq"], df["modifications"])
        ]
        LOGGER.info(
            f"Chunk {chunk_idx}: loaded and reformed {len(reformed_peptides)} peptides sequences."
        )

        try:
            sequences, tr, errors = df_to_matrix(
                reformed_peptides,
                df,
                mod_features_csv=kwargs.get("mod_features_csv"),
            )

        except Exception as e:
            LOGGER.error(
                f"Error converting sequences to matrix format in chunk {chunk_idx}: {e}"
            )
            raise

        if errors:
            LOGGER.warning(f"Errors encountered during conversion in chunk {chunk_idx}: {errors}")

        prediction_dataset = MyDataset(sequences, tr)

        if len(prediction_dataset) == 0:
            LOGGER.warning(f"Chunk {chunk_idx} contains no valid peptide entries.")
            continue

        # Keep historical x_shape contract expected by model/tests: (batch, channels, length)
        x_shape = (1,) + prediction_dataset[0][0].shape
        LOGGER.info(f"Chunk {chunk_idx} initialized with data shape {x_shape}.")
        yield df, prediction_dataset, x_shape



def get_input_shape_from_first_chunk(csv_path: str, chunk_size: int = 10000, **kwargs):

    """
    Get the input shape from the first valid chunk of a CSV file.

    :param csv_path: Path to the CSV file containing raw peptide sequences.
    :param chunk_size: Number of rows to load per chunk.
    :return: x_shape for model initialization.
    """
    for _, dataset_chunk, x_shape in data_initialize_chunked(

        csv_path=csv_path, chunk_size=chunk_size, **kwargs

    ):
        LOGGER.info(f"Detected input shape from first valid chunk: {x_shape}")
        return x_shape

    LOGGER.error("No valid chunks found in the input file.")
    raise ValueError("No valid chunks found in the input file.")