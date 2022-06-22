# A bunch of experiments on integrating `fastai` with `timm`

The main idea of this repo is finding the best way to integrate `timm` backbones into fastai so they fine tune good on downstream tasks.

## Implementing a splitter for `timm` models

Timm models are different than torchvision ones, and need a proper splitter of parameters so they can be properly finetuned.
- [03_timm_group_split.ipynb](03_timm_group_split.ipynb) provides the main chunks of code to do this, a [PR](https://github.com/fastai/fastai/pull/3636) is on course.

## Experiments

We use W&B for experiment tracking trough the `WandbCallback` in fastai. The task we are trying to optimize is fine tunning a pretrained backbone on a new dataset. 

The fine tunning is performed with `fine_tune` for a fixed number of epochs.

- The [fine_tune.py](fine_tune.py) script enables fine tunning a model:
```bash
$ python fine_tune.py --help                                                                      
usage: fine_tune.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--num_experiments NUM_EXPERIMENTS] [--learning_rate LEARNING_RATE]
                    [--img_size IMG_SIZE] [--resize_method RESIZE_METHOD] [--model_name MODEL_NAME] [--split_func SPLIT_FUNC] [--pool POOL] [--seed SEED]
                    [--wandb_project WANDB_PROJECT] [--wandb_entity WANDB_ENTITY] [--dataset DATASET]

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --num_experiments NUM_EXPERIMENTS
  --learning_rate LEARNING_RATE
  --img_size IMG_SIZE
  --resize_method RESIZE_METHOD
  --model_name MODEL_NAME
  --split_func SPLIT_FUNC
  --pool POOL
  --seed SEED
  --wandb_project WANDB_PROJECT
  --wandb_entity WANDB_ENTITY
  --dataset DATASET
```
## Running a Sweep with W&B
- Create a W&B account on https://wandb.ai/signup

- Setup your machine for fastai, you will need to install latest fastai and wandb. We can provide credtis on jarvislabs.ai if needed.

- Clone this repo

Time to run the experiment, We can perform hyperparameter sweep using W&B. 

- Setting your sweep YAML file: To run a sweep you will need first to configure what parameters you want to explore, this is done on the `sweep.yaml` file. In this file, you will specify the parameters the sweep will explore. For instance, setting:
```
learning_rate:
    values: [0.002, 0.008]
```
tells the sweep to choose the `learning_rate` param from the 2 values: `0.002` and `0.008`. What happens if we want a distribution?
```
learning_rate:
    distribution: uniform
    min: 0.0001
    max: 0.01
```
You can check all the possibilities for defining parameters [here](https://docs.wandb.ai/guides/sweeps/configuration).

- Creating the sweep: In a terminal window, run:

```bash
wandb sweep sweep.yaml
```

- Running the actual training of the sweep (you can paste the output of the previous command here)

```bash
wandb agent <SWEEP_ID>
```


References:
- If you want to know more about refactoring your code for sweeps, take a look at [How to Perform Massive Hyperparameter Experiments with W&B](https://wandb.ai/fastai/fine_tune_timm/reports/How-to-Perform-Massive-Hyperparameter-Experiments-with-W-B---VmlldzoyMDAyNDk2)