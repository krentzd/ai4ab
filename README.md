
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
![width=6](docs%2Fimages%2FOverview_figure.png)
## Installation

### Install dependencies within conda environment

1) Clone the repository with `git clone https://github.com/krentzd/ai4ab.git`
2) Create a conda environment with `python=3.9`
3) Navigate to the direcotry containing the cloned repository and install the necessary packages in your conda environment with `pip install -r requirements.txt`

### Singularity image 
Alternatively, you can build a [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) image using the provided recipe as follows:

```bash
git clone https://github.com/krentzd/ai4ab.git
cd ai4ab
singularity build ai4ab.sif ai4ab.def
```

After building the singularity image, you can directly run the training and testing scripts from the terminal:

```bash
singularity exec --bind PATH_TO_AI4AB:PATH_TO_AI4AB --nv ai4ab.sif python model/run_training.py --data_dir DATA_DIR --save_dir SAVE_DIR --train_dir TRAIN_DIR --test_dir TEST_DIR

# or

singularity exec --bind PATH_TO_AI4AB:PATH_TO_AI4AB --nv ai4ab.sif python model/run_training.py --save_dir SAVE_DIR
```

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

```bash
python preprocessing/dataset_preprocessing.py \
    --im_dir IMAGE_DIR \             # IMAGE_DIR must containt TIFF files and an Index.xml file
    --target_dir DATA_DIR \ 
    --plate_map_path PLATE_MAP.csv   # PLATE_MAP.csv must contain 'cond' and 'Destination well' columns
```

### Model training

To train a model from scratch, run the following command within your conda environment: 

```bash
python model/run_training.py \
    --data_dir DATA_DIR \
    --save_dir SAVE_DIR \ 
    --train_dir Plate_1 Plate_2 \
    --test_dir Plate_N \
```
### Model testing
To test the model on the Plate defined in `test_dir`, run the following command within your conda environment: 

```bash
python model/run_testing.py \
    --save_dir SAVE_DIR \
    --ckpt -1 \                       # -1 selects the checkpoint with the lowest validation loss
```
### Inference on a different dataset
To obtain embeddings and predictions on a different dataset, run the following command within your conda environment:

```bash
python model/run_testing.py \
    --save_dir SAVE_DIR \
    --data_dir DATA_DIR \             # This is the path to the data directory
    --test_dir Plate_1 Plate_2 \      # This specifies what plate(s) should be tested 
    --ckpt -1 \                       # -1 selects the checkpoint with the lowest validation loss
```

## Reproduce figures from manuscript

1) Download embedding data [here](https://drive.proton.me/urls/3MRM7J3MW4#dZKoPQBYuxpw)
2) Unzip file and move embedding data to directory `DATA` in `ai4ab`
3) Run analysis notebooks in the `analysis` folder

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
