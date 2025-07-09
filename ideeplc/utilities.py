from typing import List, Tuple, Dict, Union, Optional, Any

import numpy as np
import pandas as pd
import tqdm
from numpy import ndarray
from pyteomics import proforma, mass


class Config:
    """
    Configuration class for the encoding of peptides.
    """
    max_length: int = 60
    num_features: int = 1
    atoms: List[str] = ["C", "H", "O", "N", "P", "S"]
    amino_acids_order: Dict[str, int] = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    sequence_metadata: List[str] = ["length"] + atoms


config = Config()

feature_names = [
    "chemical_features",
    "diamino_chemical_features",
    "diamino_atoms",
    "atoms",
    "sequence_metadata",
    "one_hot",
]

feature_lengths = [
    config.num_features,
    config.num_features,
    len(config.atoms),
    len(config.atoms),
    len(config.sequence_metadata),
    len(config.amino_acids_order),
]

# Feature indices calculation
feature_indices = {
    name: (sum(feature_lengths[:i]), sum(feature_lengths[:i + 1]))
    for i, name in enumerate(feature_names)
}

num_channels = sum(feature_lengths)

def aa_atomic_composition_array() -> Dict[str, np.ndarray]:
    """Create an array of atomic compositions for amino acids."""
    return {aa: np.array(
        [mass.std_aa_comp[aa].get(atom, 0) for atom in config.atoms],
        dtype=np.float32)
        for aa in config.amino_acids_order}


def aa_chemical_feature() -> Dict[str, np.ndarray]:
    """Get chemical features for amino acids."""
    df_aminoacids = pd.read_csv('ideeplc/structure_feature/aa_stan.csv')
    # Convert the dataframe to a dictionary
    amino_acids = df_aminoacids.set_index('AA').T.to_dict('list')
    # Convert the dictionary to a dictionary of numpy arrays for each AA
    features_arrays = {aa: np.array(features, dtype=np.float32) for aa, features in amino_acids.items()}
    return features_arrays


def mod_chemical_features() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Get modification features."""
    df = pd.read_csv('ideeplc/structure_feature/ptm_stan.csv')
    # Convert the dataframe to a dictionary and transpose it
    df = df.set_index('name').T
    # Convert the DataFrame to a dictionary of modifications with their chemical features
    modified = df.to_dict('list')
    dic = {}
    for key, values in modified.items():
        main_key, sub_key = key.split('#')
        # Create a nested dictionary with the modification name and the amino acid
        dic.setdefault(main_key, {})[sub_key] = dict(zip(df.index, values))
    return dic


def peptide_parser(peptide: str) -> Tuple:
    """Parse the peptide sequence and modifications."""
    modifications = []
    parsed_sequence, modifiers = proforma.parse(peptide)
    sequence = "".join([aa for aa, _ in parsed_sequence])

    for loc, (_, mods) in enumerate(parsed_sequence):
        if mods:
            modifications.append(f"{loc + 1}:{mods[0].name}")
    modifications = "|".join(modifications)

    return parsed_sequence, modifiers, sequence, modifications


def empty_array() -> np.ndarray:
    """Create an empty array for encoding sequences and modifications."""
    return np.zeros(shape=(num_channels, config.max_length + 2), dtype=np.float32)


# Encode sequence and modifications
def encode_sequence_and_modification(sequence: str, parsed_sequence: List, modifications_dict: Dict,
                                     aa_to_feature: Dict[str, np.ndarray],
                                     n_term: List = None, c_term: List = None) -> np.ndarray:
    """Encode the sequence and modifications into a matrix."""
    encoded = empty_array()
    start_index = feature_indices["chemical_features"][0]

    for j, aa in enumerate(sequence):
        encoded[start_index: start_index + config.num_features, j + 1] = aa_to_feature[aa]

    for loc, (aa, mods) in enumerate(parsed_sequence):
        if mods:
            for mod in mods:
                name = mod.name
                encoded[start_index: start_index + config.num_features, loc + 1] = list(
                    modifications_dict[name][aa].values())
        if n_term:
            for mod in n_term:
                name = f"{mod.name}(N-T)"
                encoded[start_index: start_index + config.num_features, 1] = list(
                    modifications_dict[name][sequence[0]].values())
        if c_term:
            loc = len(parsed_sequence) + 1
            for mod in c_term:
                name = mod.name
                encoded[start_index: start_index + config.num_features, loc] = list(
                    modifications_dict[name][sequence[-1]].values())

    return encoded


def encode_diamino_sequence_and_modification(encode_seq_mod: np.ndarray) -> np.ndarray:
    """Encode diamino sequence and modifications."""
    encoded = empty_array()
    start_index = feature_indices["chemical_features"][0]
    start_index_diamino = feature_indices["diamino_chemical_features"][0]

    for loc in range(1, len(encode_seq_mod.T) - 1, 2):
        encoded[start_index_diamino:start_index_diamino + config.num_features, int(loc / 2) + 1] = encode_seq_mod[
                                                                                                   start_index:start_index + config.num_features,
                                                                                                   loc + 1] + encode_seq_mod[
                                                                                                              start_index:start_index + config.num_features,
                                                                                                              loc]

    return encoded


def encode_sequence_and_modification_atomic(sequence: str, parsed_sequence: List,
                                            amino_acids_atoms: Dict[str, np.ndarray],
                                            n_term: List = None, c_term: List = None) -> np.ndarray:
    """Encode atomic composition for sequence and modifications."""
    encoded = empty_array()
    start_index = feature_indices["atoms"][0]

    for loc, (aa, mods) in enumerate(parsed_sequence):
        if mods:
            for mod in mods:
                mod_comp = mod.composition
                encoded[start_index: start_index + len(config.atoms), loc + 1] = [mod_comp.get(a, 0) for a in
                                                                                  config.atoms]
        if n_term:
            for mod in n_term:
                mod_comp = mod.composition
                encoded[start_index:start_index + len(config.atoms), 1] = [mod_comp.get(a, 0) for a in config.atoms]
        if c_term:
            loc = len(parsed_sequence) + 1
            for mod in c_term:
                mod_comp = mod.composition
                encoded[start_index: start_index + len(config.atoms), loc] += [mod_comp.get(a, 0) for a in config.atoms]

    for j, aa in enumerate(sequence):
        encoded[start_index: start_index + len(config.atoms), j + 1] += amino_acids_atoms[aa]

    return encoded


def encode_diamino_sequence_and_modification_atomic(encode_seq_mod_atomic: np.ndarray) -> np.ndarray:
    """Encode diamino atomic sequence and modifications."""
    encoded = empty_array()
    start_index = feature_indices["diamino_atoms"][0]
    summed = encode_seq_mod_atomic[feature_indices["atoms"][0]:feature_indices["atoms"][1], 1:-1].reshape(
        len(config.atoms), -1, 2).sum(axis=2)
    encoded[start_index:start_index + len(config.atoms), 1:int(config.max_length / 2) + 1] = summed

    return encoded


def encode_sequence_metadata(sequence: str, encode_seq_mod_atomic: np.ndarray) -> np.ndarray:
    """Encode metadata about the sequence."""
    encoded = empty_array()
    start_index = feature_indices["sequence_metadata"][0]
    seq_len = len(sequence)

    encoded[start_index, 1:-1] = seq_len / config.max_length
    start_index += 1

    atom_counts = encode_seq_mod_atomic[feature_indices["atoms"][0]:feature_indices["atoms"][1], :]
    first_4_aa = atom_counts[:, 1:5]
    last_4_aa = atom_counts[:, -4:]
    total_aa = atom_counts.sum(axis=1, keepdims=True)

    combined = np.concatenate((first_4_aa, last_4_aa, total_aa), axis=1)
    encoded[start_index:start_index + len(config.atoms), 1:1 + combined.shape[1]] = combined

    return encoded


def encode_sequence_one_hot(sequence: str) -> np.ndarray:
    """One-hot encode the sequence."""
    encoded = empty_array()
    start_index = feature_indices["one_hot"][0]

    for j, aa in enumerate(sequence):
        encoded[start_index + config.amino_acids_order[aa], j + 1] = 1

    return encoded


def df_to_matrix(seqs: Union[str, List[str]],
                 df: Optional[pd.DataFrame] = None
                 ) -> tuple[ndarray, list[Any], list[list[str | list[str] | int | Exception]]] | ndarray | Any:
    """
    Convert a peptide or a list of peptides to their matrix representation.

    If a DataFrame with 'tr' and 'predictions' columns is also provided,
    the function returns a tuple: (matrix, tr_values, prediction_values, errors).
    Otherwise, it returns only the encoded matrix representation.

    Parameters
    ----------
    seqs : str or List[str]
        A single peptide sequence or a list of peptide sequences.
    df : pd.DataFrame, optional
        DataFrame with 'tr' and 'predictions' columns. Must align with the list of sequences.

    Returns
    -------
    If `seqs` is a string:
        np.ndarray : matrix representation of the peptide.
    If `seqs` is a list and `df` is None:
        np.ndarray : stacked matrix representation of all peptides.
    If `seqs` is a list and `df` is given:
        Tuple[np.ndarray, List[float], List] :
            - stacked matrix
            - list of tr values
            - list of errors (with peptide, index, and exception)
    """

    # Normalize to list of sequences
    single_input = isinstance(seqs, str)
    if single_input:
        seqs = [seqs]

    seqs_encoded = []
    tr = []
    errors = []
    modifications_dict = mod_chemical_features()
    aa_to_feature = aa_chemical_feature()
    amino_acids_atoms = aa_atomic_composition_array()

    for idx, peptide in tqdm.tqdm(enumerate(seqs), total=len(seqs)):
        try:
            parsed_sequence, modifiers, sequence, modifications = peptide_parser(peptide)

            encode_seq_mod = encode_sequence_and_modification(
                sequence, parsed_sequence, modifications_dict, aa_to_feature,
                modifiers["n_term"], modifiers["c_term"]
            )
            encode_di_seq_mod = encode_diamino_sequence_and_modification(encode_seq_mod=encode_seq_mod)

            encode_seq_mod_atomic = encode_sequence_and_modification_atomic(
                sequence, parsed_sequence, amino_acids_atoms,
                modifiers["n_term"], modifiers["c_term"]
            )
            encode_di_seq_mod_atomic = encode_diamino_sequence_and_modification_atomic(encode_seq_mod_atomic)

            encode_seq_meta = encode_sequence_metadata(sequence, encode_seq_mod_atomic)
            seq_hot = encode_sequence_one_hot(sequence)

            peptide_encoded = (
                encode_seq_mod
                + encode_di_seq_mod
                + encode_di_seq_mod_atomic
                + encode_seq_mod_atomic
                + encode_seq_meta
                + seq_hot
            )

            seqs_encoded.append(peptide_encoded)

            if df is not None:
                tr.append(df['tr'].iloc[idx])

        except Exception as e:
            errors.append([peptide, idx, e])
            continue

    seqs_stack = np.stack(seqs_encoded)
    if single_input:
        return seqs_stack[0]
    elif df is not None:
        return seqs_stack, tr, errors
    else:
        return seqs_stack


def reform_seq(seq: str, mod: str) -> str:
    """Reform a sequence by adding modifications in the correct positions."""
    mod_list = [m for m in mod.split('|')]
    mod_list_tuple = []

    if not mod:
        return seq

    while mod_list:
        mod_list_tuple.append((int(mod_list.pop(0)), mod_list.pop(0)))
    mod_list_tuple.sort()

    while mod_list_tuple:
        index, modification = mod_list_tuple.pop()
        seq = seq[:index] + f'[{modification}]' + seq[index:]
        if seq.startswith('['):
            seq = seq.replace(']', ']-', 1)

    return seq


def reform_seq_ignore_mod(seq: str, mod: str, aa: str) -> str:
    """Reform a sequence by ignoring modifications at specific amino acid positions."""
    mod_list = [m for m in mod.split('|')]
    mod_list_tuple = []

    if not mod:
        return seq

    while mod_list:
        mod_list_tuple.append((int(mod_list.pop(0)), mod_list.pop(0)))
    mod_list_tuple.sort()

    while mod_list_tuple:
        index, modification = mod_list_tuple.pop()

        if index == 0 and modification.startswith('Acetyl'):
            seq = seq[:index] + f'[{modification}]' + seq[index:]
        elif seq[index - 1] == aa or seq[index - 1] == "G":
            continue
        else:
            seq = seq[:index] + f'[{modification}]' + seq[index:]

    if seq.startswith('['):
        seq = seq.replace(']', ']-', 1)

    return seq
