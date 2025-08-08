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

### Graphical user interface (GUI)

If you prefer not to install Python or any dependencies, you can use the **standalone iDeepLC GUI** for Windows.  
This is a single `.exe` file that runs without any installation.

### How it works
- When you run the `.exe`, a **terminal window** will first appear.  
  This terminal acts as the **logger** for the GUI, showing progress and messages as the program runs.
- Any **results** and **generated figures** will be saved **in the same folder** where the `.exe` file is located.

### Running the executable
1. Download the `.exe` file from the [latest release](https://github.com/CompOmics/iDeepLC/releases).
2. Double-click the file to run it.
3. If Windows shows a security message:  
   - **"Windows protected your PC"** — this is a standard warning for applications not signed with a commercial certificate.  
   - Click **More info** and then **Run anyway** to start iDeepLC.  
     > *This warning appears because the executable is built by the developers without a paid code-signing certificate.  
     > The file is safe if downloaded from the official GitHub release page.*


<img width="700" height="630" alt="image" src="https://github.com/user-attachments/assets/1a31ea64-b377-4b86-945b-848fc7c9f123" />



### CLI

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
