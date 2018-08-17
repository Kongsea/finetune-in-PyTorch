# finetune-in-PyTorch

Finetune networks in pytorch

## Requirements

1. imgaug
2. tqdm

## Usage

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

## Data and dataset

1. Use `gen_dataset.py` to generate datasets in csv files from data folders.

2. Use `gen_and_save_images.py` to test the dataset and dataloader from csv files.

3. Use `gen_new_data.py` to generate a csv from a folder to test the files.

4. Use `fetch_data_accordingto_error.py` to fetch error data from the trainning or validation data.

## Models

1. ResNet

   ResNet 50, 101 and 152 from official repositories.

2. Se_ResNeXt

   `senet.py` is the model definition, `train_se_resnext.py` and `train_se_resnext.sh` are used to train models.

3. PNasNet

   `pnasnet.py` is the model definition, `train_pnasnet.py` and `train_pnasnet.sh` are used to train models.

## Learning rate decay methods

1. Decay according to steps

   `finetune.py` and `finetune.sh`.

2. Reduce on Plateau

   `finetune_rop.py` and `finetune_rop.sh`.

## References

Some codes were borrowed and modified from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
