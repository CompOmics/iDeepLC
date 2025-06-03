![ideeplc2](https://github.com/user-attachments/assets/86e9b793-39be-4f62-8119-5c6a333af487)

# iDeepLC: A deep Learning-based retention time predictor for unseen modified peptides with a novel encoding system

## Overview

iDeepLC is a deep learning-based tool for retention time prediction in proteomics. It supports multiple evaluation types, including **20 datasets evaluation**, **modified glycine evaluation**, and **PTM evaluation**.

The repository provides tools for **training models**, **evaluating retention time predictions**, and **generating figures** for analysis.

## Repository Structure

```
Zenodo/
│── data/                            # Input datasets for evaluation
│   ├── 20_datasets_evaluation/       # Data for 20 dataset evaluation
│   ├── modified_glycine_evaluation/  # Data for modified glycine evaluation
│   ├── PTM_evaluation/               # Data for PTM evaluation
│   ├── structure_feature/            # Amino acids and PTMs structure information
│
│── iDeepLC_Zenodo/                    # Main implementation
│   ├── wandb/                        # Weights & Biases logs (if used)
│   ├── config.py                     # Configuration settings for deep learning model such as epoch, batch size and etc
│   ├── data_initialize.py             # Data loading and preprocessing
│   ├── evaluate.py                    # Model evaluation functions
│   ├── figure.py                      # Functions to generate evaluation figures
│   ├── LICENSE                        # License file
│   ├── main.py                        # Main script to train/evaluate models
│   ├── model.py                       # Model architecture
│   ├── plots_generator.ipynb          # Jupyter Notebook for generating manuscript figures
│   ├── README.md                      # GitHub Documentation
│   ├── requirements.txt               # Required dependencies
│   ├── train.py                        # Training functions
│   ├── utilities.py                    # Input data generator
│
│── saved_model/                        # Pre-trained/newly-trained models
│   ├── 20_datasets_evaluation/         # Pretrained models for 20 datasets
│   ├── modified_glycine_evaluation/    # Pretrained models for glycine evaluation
│   ├── modified_glycine_evaluation_DeepLC/  # DeepLC glycine output
│   ├── PTM_evaluation/                 # Pretrained PTM evaluation models
│   ├── PTM_evaluation_DeepLC/          # DeepLC PTM output
│
│── README.md                            # Project documentation
```

---

## How to Use

### 1️⃣ **Generating Figures (Manuscript Plots)**  
Use the provided Jupyter Notebook:

```sh
cd iDeepLC_Zenodo
jupyter notebook plots_generator.ipynb
```

This will generate **all the figures presented in the manuscript**.

---

### 2️⃣ **Training & Evaluation**
The `main.py` script allows users to **train models**, **evaluate them**, and **generate figures**.

#### **Run Training + Evaluation**
To train a new model and evaluate it:
You need to choose one of the three evaluation types(20datasets, ptm, aa_glycine) and one dataset based on the evaluation type. If you select aa_glycine evaluation type, you need to also choose an amino acid.
```sh
python main.py --eval_type 20datasets --dataset_name arabidopsis --train
python main.py --eval_type aa_glycine --dataset_name arabidopsis  --test_aa A  --train
python main.py --eval_type ptm --dataset_name Acetyl  --train
python main.py --eval_type ptm --dataset_name Acetyl  --train --save_results
```
- For `aa_glycine`, figures will only be generated after **all amino acids** are processed.

#### **Run Only Evaluation + Figure Generation**
If you want to **use pretrained models** for evaluation and figure generation:

```sh
python main.py --eval_type 20datasets --dataset_name arabidopsis
```

This loads the pre-trained model such as `saved_model/20_datasets_evaluation/arabidopsis/best.pth`, evaluates it, and **generates figures**.

---

## Dependencies

To install dependencies:

```sh
pip install -r requirements.txt
```
## Data download 

Due to size restriction, the data is not included in this repository. The data can be downloaded from Zenodo at the following link: [Zenodo Data](https://zenodo.org/records/15011301).

---

## Citation

If you use **iDeepLC** in your research, please cite our paper:

📄 **iDeepLC: A deep Learning-based retention time predictor for unseen modified peptides with a novel encoding system**  
🖊 **Alireza Nameni, Arthur Declercq, Ralf Gabriels, Robbe Devreese, Lennart Martens, Sven Degroeve , and Robbin Bouwmeester**  
📅 **2025**  
🔗 **DOI**
