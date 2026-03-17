
<h1 align="center">
AI for Antibiotics (AI4AB)

</h1>

## Deep learning recognises antibiotic mode of action from brightfield images
<p align="center">
    <a href="https://www.biorxiv.org/content/10.1101/2025.03.30.645928v3"><img alt="Paper" src="https://img.shields.io/badge/paper-bioRxiv-%23b62b39"></a>
    <a href="https://github.com/krentzd/ai4ab"><img alt="github" src="https://img.shields.io/github/stars/krentzd/ai4ab?style=social"></a>
    <a href="https://github.com/krentzd/ai4ab"><img alt="github" src="https://img.shields.io/github/forks/krentzd/ai4ab?style=social"></a>
</p>
</p>

## Overview 
This repository contains the source code to reproduce the analysis from "Deep learning recognises antibiotic mode of action from brightfield images".
![width=10](docs%2Fimages%2FOverview_figure.png)
## Installation

### Install dependencies within conda environment

1) Clone the repository with `git clone https://github.com/krentzd/ai4ab.git`
2) Create a conda environment with `python=3.9`
3) Navigate to the direcotry containing the cloned repository and install the necessary packages in your conda environment with `pip install -r requirements.txt`

#### Singularity image 

   
## Usage 

### Datset preparation

Your dataset must obey the following folder structure: 

```
├── DATA_DIR
    ├── Plate_1
        ├── Compound_1_Concentration_1
            ├── img_1.tiff
            ├── img_2.tiff
            ├── ...
        ├──...
        ├── Compound_N_Concentration_M
    ├── ...
    ├── Plate_K
```

Use the following script to preprocess a TIFF dataset acquired on a Revvity Opera Phenix high-content screening system according to the above-described folder structure: 

```cli
cd preprocessing
python dataset_preprocessing.py \
    --im_dir IMAGE_DIR \             # IMAGE_DIR must containt TIFF files and an Index.xml file
    --target_dir DATA_DIR \ 
    --plate_map_path PLATE_MAP.csv   # PLATE_MAP.csv must contain 'cond' and 'Destination well' columns
```

### Model training

To train a model from scratch, run the following command in your terminal: 
```cli
cd model
python run_training.py \
    --data_dir DATA_DIR \
    --save_dir SAVE_DIR \ 
    --train_dir Plate_1 Plate_2 \
    --test_dir Plate_N \
```
### Model testing
To test the model on the Plate defined in `test_dir`, run the following command: 

```cli
cd model
python run_testing.py \
    --save_dir SAVE_DIR \
    --ckpt -1 \                   # -1 selects the checkpoint with the lowest validation loss
```

## Reproduce figures from manuscript

1) Download embedding data [here](https://drive.proton.me/urls/3MRM7J3MW4#dZKoPQBYuxpw)
2) Unzip file and move embedding data to directory `DATA` in `ai4ab`
3) Run analysis notebooks

## How to cite
```bibtex
@article{krentzel2025deep,
  title={Deep learning recognises antibiotic modes of action from brightfield images},
  author={Krentzel, Daniel and Kho, Kelvin and Petit, Julienne and Mahtal, Nassim and Chiaravalli, Jeanne and Shorte, Spencer L and Wehenkel, Anne Marie and Boneca, Ivo G and Zimmer, Christophe},
  journal={bioRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
