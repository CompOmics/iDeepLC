![ideeplc2](https://github.com/user-attachments/assets/86e9b793-39be-4f62-8119-5c6a333af487)

# iDeepLC: A deep Learning-based retention time predictor for unseen modified peptides with a novel encoding system

## Overview

iDeepLC is a deep learning-based tool for retention time prediction in proteomics.

## Features

- **Retention Time Prediction**: Predict retention times for peptides, including modified ones.
- **Fine-Tuning**: Fine-tune the pre-trained model for specific datasets.
- **Visualization**: Generate scatter plots and other figures for analysis.

## installation

Intall the package using pip:

```sh
pip install iDeepLC
```

## Usage

The iDeepLC package provides a CLI for easy usage. Below are some examples:
#### Prediction
```sh
ideeplc --input <path/to/peptide_file.csv> --save
```
#### Fine-tuning
```sh
ideeplc --input <path/to/peptide_file.csv> --save --finetune
```
#### Calibration
```sh
ideeplc --input <path/to/peptide_file.csv> --save --calibrate
```
#### Example
```sh
ideeplc --input ./data/example_input/Hela_deeprt --save --finetune --calibrate
```

For more detailed CLI usage, you can run:
```sh
ideeplc --help
```

## Input file format

The input file should be a CSV file with the following columns:
- `seq`: The amino acid sequence of the peptide. (e.g., `ACDEFGHIKLMNPQRSTVWY`)
- `modifications`: A string representing modifications in the sequence. (e.g., `11|Oxidation|16|Phospho`)
- `tr`: The retention time of the peptide in seconds. (e.g., `1285.63`)

For example:
```csv
NQDLISENK,,2705.724
LGSPPPHK,3|Phospho,2029.974
RMQSLQLDCVAVPSSR,2|Oxidation|4|Phospho,4499.832
```

## Citation

If you use **iDeepLC** in your research, please cite our paper:

📄 **iDeepLC: A deep Learning-based retention time predictor for unseen modified peptides with a novel encoding system**  
🖊 **Alireza Nameni, Arthur Declercq, Ralf Gabriels, Robbe Devreese, Lennart Martens, Sven Degroeve , and Robbin Bouwmeester**  
📅 **2025**  
🔗 **DOI**