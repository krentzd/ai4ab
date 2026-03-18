# ai4ab — AI for Antibiotics

Deep learning pipeline for classifying antibiotic mechanisms of action (MoA) from brightfield microscopy images of bacteria. Built on an EfficientNet-B0 backbone with average-pooling tile aggregation.

## Citation

If you use this code, please cite the associated preprint:

> Krentzel D, Kho K, Petit J, Mahtal N, Shorte SL, Wehenkel AM, Boneca IG, Zimmer C.  
> **Deep learning recognises antibiotic modes of action from brightfield images.**  
> *bioRxiv* 2025.03.30.645928  
> https://doi.org/10.1101/2025.03.30.645928

## Requirements

Build the Singularity container from `ai4ab.def`:

```bash
singularity build ai4ab.sif ai4ab.def
```

## Project structure

```
ai4ab/
├── config.py               # TrainingConfig dataclass and CLI parsers
├── run_training.py         # Training entry point
├── run_testing.py          # Testing entry point
├── data/
│   ├── dataset.py          # TiffDataset
│   ├── transforms.py       # Train/test augmentation pipelines
│   ├── class_params.py     # MoA dictionaries and class weights
│   └── loader.py           # load_data
├── models/
│   └── cnn.py              # AvgPoolCNN
├── training/
│   ├── trainer.py          # Training loop with TensorBoard logging
│   └── evaluator.py        # Inference and result saving
├── utils/
│   ├── transforms.py       # OverlappingCropMultiChannel
│   ├── filesystem.py       # Directory helpers
│   ├── checkpoint.py       # Checkpoint loading
│   ├── viz.py              # Plotting utilities
│   ├── utils.py            # Shared utilities
│   └── io.py               # Tensor utilities
└── tests/
    └── test_pipeline.py    # Behaviour parity tests
```

## Training

```bash
python run_training.py \
    --data_dir /data/ecoli \
    --save_dir /runs/exp01 \
    --train_dir rep1 rep2 \
    --test_dir rep3 \
    --channels 0 1 2 \
    --epochs 100 \
    --use_e_coli_moa
```

TensorBoard logs are written to `<save_dir>/tensorboard/`. Launch with:

```bash
tensorboard --logdir /runs/exp01/tensorboard
```

## Testing

```bash
python run_testing.py \
    --save_dir /runs/exp01 \
    --test_dir rep4 \
    --ckpt 50
```

Overridable fields: `--data_dir`, `--test_dir`, `--dropped_classes`.

## Tests

```bash
cd ai4ab
python -m pytest tests/test_pipeline.py -v
```
