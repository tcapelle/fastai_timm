# A bunch of experiments on integrating `fastai` with `timm`

The main idea of this repo is finding the best way to integrate `timm` backbones into fastai so they fine tune good on downstream tasks.

## Implementing a splitter for `timm` models

Timm models are different than torchvision ones, and need a proper splitter of parameters so they can be properly finetuned.
- [03_timm_group_split.ipynb](03_timm_group_split.ipynb) provides the main chunks of code to do this, a [PR](https://github.com/fastai/fastai/pull/3636) is on course.

## Experiments

We use W&B for experiment tracking trough the `WandbCallback` in fastai. The task we are trying to optimize is fine tunning a pretrained backbone on the Oxford pets dataset. This is very straight forward:
```python
def get_dls(batch_size, img_size, seed):
    dataset_path = untar_data(URLs.PETS)
    files = get_image_files(dataset_path/"images")
    pat = re.compile('(^[a-zA-Z]+_*[a-zA-Z]+)')
    labels = [pat.match(f.name)[0] for f in files]
    dls = ImageDataLoaders.from_name_re(dataset_path, files, 
                                        r'(^[a-zA-Z]+_*[a-zA-Z]+)', 
                                        valid_pct=0.2, 
                                        seed=seed, 
                                        bs=batch_size,
                                        item_tfms=Resize(img_size)) 
    return dls
```

The fine tunning is performed with `fine_tune` for a fixed number of epochs. We are also tyring a strategy of doing a pre `fit`:
```python
learn.fit(args.fit_epochs, args.fit_learning_rate)
learn.fine_tune(args.ft_epochs, args.ft_learning_rate)
```

- The [fine_tune.py](fine_tune.py) script enables fine tunning a model:
```bash
> python fine_tune.py --help                                                                                                                            paperspace at psyer6c5z (-)(main)
usage: fine_tune.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--num_experiments NUM_EXPERIMENTS] [--learning_rate LEARNING_RATE] [--img_size IMG_SIZE] [--model_name MODEL_NAME] [--seed SEED] [--mixup]
                    [--force_torchvision] [--wandb_project WANDB_PROJECT]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --epochs EPOCHS
  --num_experiments NUM_EXPERIMENTS
  --learning_rate LEARNING_RATE
  --img_size IMG_SIZE
  --model_name MODEL_NAME
  --seed SEED
  --mixup
  --force_torchvision
  --wandb_project WANDB_PROJECT
```

- You can aslo perform an hyperparameter sweep using the [sweep.py](sweep.py) file, that insed calls the [jeremy_ft.py](jeremy_ft.py) using the 1-fit 3-fine_tune strategy. You can modify the sweep hyperparams inside [sweep.py](sweep.py).

```bash
> python sweep.py --help                                                                                                                              paperspace at psyer6c5z (-)(main)
usage: sweep.py [-h] [--sweep_id SWEEP_ID] [--sweep_count SWEEP_COUNT] [--sweep_method SWEEP_METHOD] [--sweep_goal SWEEP_GOAL] [--sweep_metric_name SWEEP_METRIC_NAME]
                [--early_terminate_min_iter EARLY_TERMINATE_MIN_ITER] [--early_terminate_type EARLY_TERMINATE_TYPE] [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --sweep_id SWEEP_ID
  --sweep_count SWEEP_COUNT
  --sweep_method SWEEP_METHOD
  --sweep_goal SWEEP_GOAL
  --sweep_metric_name SWEEP_METRIC_NAME
  --early_terminate_min_iter EARLY_TERMINATE_MIN_ITER
  --early_terminate_type EARLY_TERMINATE_TYPE
  --model_name MODEL_NAME

```

A bunch of sweeps have been performed, and are available [here](https://wandb.ai/capecape/fine_tune_timm/sweeps)

You can also follow the project workspace [here](https://wandb.ai/capecape/fine_tune_timm)