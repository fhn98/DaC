# Decompose-and-Compose: A Compositional Approach to Mitigating Spurious Correlation

This repository is the official implementation of the paper [Decompose-and-Compose: A Compositional Approach to Mitigating Spurious Correlation](https://openaccess.thecvf.com/content/CVPR2024/html/Noohdani_Decompose-and-Compose_A_Compositional_Approach_to_Mitigating_Spurious_Correlation_CVPR_2024_paper.html), accepted at CVPR 2024 main conference.

## Requirements

Our code requires Python 3.9 or higher to run successfully.
Please use either `requirements.txt` with `pip` to install dependencies.

## Datasets

The following datasets are supported: Waterbirds, CelebA, Dominoes, and MetaShift.

### Waterbirds and CelebA

Follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets.

### CelebA

Our code expects the following files/folders in the `[root_dir]/celebA` directory:

- `data/celeba_metadata.csv`
- `data/img_align_celeba/`

You can download these dataset files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset).

### Waterbirds

Our code expects the following files/folders in the `[root_dir]/` directory:

- `data/waterbird_complete95_forest2water2/`

You can download a tarball of this dataset [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz).

### Dominoes

The code for preparing the Dominoes datasets can is provided in `./notebooks/Dominoes.ipynb`. The code is mainly adapted from [this repo](https://github.com/mpagli/Agree-to-Disagree/tree/main).
You can download a presaved instance of the dataset from [this link](https://drive.google.com/drive/folders/1iXOFqxA6IAWTS_MD9xy3SD7FMTGChC2t?usp=sharing).

### MetaShift

We use the implementation provided by [DISC repo](https://github.com/Wuyxin/DISC) for this dataset. You can download the dataset from [here](https://drive.usercontent.google.com/download?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT&authuser=0).


## Training

1. **ERM Training**
 Waterbirds Example
```bash
python main.py --experiment ERM --dataset WaterBirds --data_path /path/to/waterbird_complete95_forest2water2 --optimizer SGD --lr 1e-3 --weight_decay 1e-3 --num_epochs 100 --batch_size 128
```

2. **Adaptive Masking**
Before running the main experiment, you first need to run `adaptive_mask.py` to extract and save the masks for images by adaptive masking. Here is an Example for the Waterbirds dataset.
```bash
python adaptive_mask.py --dataset WaterBirds --data_path /path/to/waterbird_complete95_forest2water2 --model_path /path/to/ERM model --batch_size 128
```

3. **DaC Main Experiment**
Example for the Waterbirds dataset
```bash
python main.py --experiment DaC --dataset WaterBirds --data_path /path/to/waterbird_complete95_forest2water2 --mask_path /path/to/saved masks --save_path /path/to/saved/checkpoints --optimizer Adam --scheduler StepLr --step_size 5 --gamma 0.5 --lr 5e-3 --weight_decay 0 --num_epochs 20 --alpha 10 --quantile 0.8 --batch_size 64
```


## Usage
To run an experiment, use the `main.py` script with the appropriate arguments:

```bash
python main.py [--lr LEARNING_RATE] [--optimizer {Adam,SGD}] [--scheduler {none, StepLr}]
               [--experiment {ERM,DaC}] [--dataset {WaterBirds, CelebA, Domino, MetaShift}]
               [--data_path DATASET_PATH] [--mask_path MASK_PATH]  [--save_path SAVE_PATH]
               [--num_envs NUM_ENVS] [--num_test_envs NUM_TEST_ENVS] [--num_classes NUM_CLASSES]
               [--weight_decay WEIGHT_DECAY] [--step_size STEP_SIZE] [--gamma GAMMA]
               [--num_epochs NUM_EPOCHS] [--model_path MODEL_PATH] [--batch_size BATCH_SIZE]
               [--invert_mask INVERT_MASK] [--quantile QUANTILE] [--alpha ALPHA] [--seed SEED]
```

### Arguments

- `--lr`: learning rate (default: `0.005`).
- `--optimizer`: Type of optimizer (choices: `Adam`, `SGD`; default: `Adam`).
- `--scheduler`: Type of scheduler (choices: `none`, `StepLr`; default: `none`).
- `--experiment`: Type of experiment (choices: `ERM`, `DaC`; default `DaC`).
- `--dataset`: Name of the dataset (choices: `WaterBirds`, `CelebA`, `Domino`, `MetaShift`; required).
- `--data_path`: Path to the data (required).
- `--mask_path`: Path to the masks (default: `./data/masks/`).
- `--save_path`: Path to save checkpoints (default: `./`).
- `--num_envs`: Number of training environments (default: `4`).
- `--num_test_envs`: Number of validation and test environments (default: `4`).
- `--num_classes`: Number of Classes (default: `2`).
- `--weight_decay`: Weight decay (default: `0`).
- `--step_size`: Step Size for StepLr scheduler (default: `5`).
- `--gamma`: Gamma parameter in StepLr scheduler (default: `0.5`)
- `--num_epochs`: Number of epochs (default: `20`).
- `--model_path`: Path to the ERM model (default: `./ckpts/`).
- `--batch_size`: Batch size (default: `32`).
- `--invert_mask`: Flag for inverting the masks (default: `False`).
- `--quantile`: Quantile of low-loss selected samples (default: `0.8`).
- `--alpha`: Coefficient of combined images loss (default: `6`).
- `--seed`: Seed (default: `0`).




