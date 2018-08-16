# finetune-in-PyTorch

Finetune networks in pytorch

## Requirements

1. imgaug
2. tqdm

## Usage:

1. Use `gen_dataset.py` to generate datasets in csv files from data folders.

   The data were located in seperate folders according to classes.

   The generated datasets are `train.csv` and `valid.csv`.

2. Use `finetune.sh` to train networks according to generated datasets.

   ```
   ./finetune.sh 1
   ```

   to train networks using GPU 1.

3. Use `inference.py` to test some data with the trained model.

4. Use `check_train.py` to check the trainning dataset and fetch out the unconsistent data.
