import wandb
import argparse
import torchvision as tv
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
import open_clip

WANDB_PROJECT = 'ft_pets_planet'
WANDB_ENTITY = 'fastai'

config_defaults = SimpleNamespace(
    batch_size=32,
    epochs=5,
    num_experiments=1,
    learning_rate=2e-3,
    img_size=224,
    resize_method="crop",
    model_name="resnet34",
    clip_checkpoint=None,
    use_torchvision=False,
    pool="concat",
    seed=42,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    split_func="default",
    dataset="PETS",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--img_size', type=int, default=config_defaults.img_size)
    parser.add_argument('--resize_method', type=str, default=config_defaults.resize_method)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--clip_checkpoint', type=str, default=config_defaults.clip_checkpoint)
    parser.add_argument('--use_torchvision', type=bool, default=config_defaults.use_torchvision)
    parser.add_argument('--split_func', type=str, default=config_defaults.split_func)
    parser.add_argument('--pool', type=str, default=config_defaults.pool)
    parser.add_argument('--seed', type=int, default=config_defaults.seed)
    parser.add_argument('--wandb_project', type=str, default=WANDB_PROJECT)
    parser.add_argument('--wandb_entity', type=str, default=WANDB_ENTITY)
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    return parser.parse_args()

def get_clip_model(model_name, pretrained, n_out):
    """ 
    Returns a CLIP model with the given name and pretrained state
    """
    backbone = open_clip.create_model(model_name, pretrained, jit=False).visual
    backbone.proj = None
    if 'ViT' in model_name:
        n_hidden = backbone.ln_post.weight.shape[0]
    else:
        n_hidden = backbone.attnpool.c_proj.weight.shape[0]
    linear = nn.Linear(n_hidden, n_out)
    model = nn.Sequential(backbone, linear)
    return model

def get_gpu_mem(device=0):
    "Memory usage in GB"
    gpu_mem = torch.cuda.memory_stats_as_nested_dict(device=device)
    return (gpu_mem["reserved_bytes"]["small_pool"]["peak"] + gpu_mem["reserved_bytes"]["large_pool"]["peak"])*1024**-3

def get_pets(batch_size, img_size, seed, method="crop"):
    dataset_path = untar_data(URLs.PETS)
    files = get_image_files(dataset_path/"images")
    dls = ImageDataLoaders.from_name_re(
            dataset_path, files, r'(^[a-zA-Z]+_*[a-zA-Z]+)', valid_pct=0.2,
            seed=seed, bs=batch_size, item_tfms=Resize(img_size, method=method))
    return dls, [error_rate, accuracy]

def get_planets(batch_size=64, img_size=224, seed=42, method="crop"):
    dataset_path=untar_data(URLs.PLANET_SAMPLE)
    dls = ImageDataLoaders.from_csv(
            dataset_path, folder="train", csv_fname="labels.csv", label_delim=" ",
            suff=".jpg", bs=batch_size, seed=seed, item_tfms=Resize(img_size, method=method))
    metrics = [accuracy_multi, FBetaMulti(beta=2)]
    return dls, metrics

def get_dataset(dataset_name, *args, **kwargs):
    if dataset_name   == "PETS":    return get_pets(*args, **kwargs)
    elif dataset_name == "PLANETS": return get_planets(*args, **kwargs)
    else: raise Exception("Dataset not found, supports: PETS or PLANETS")

def train(config=config_defaults):
    with wandb.init(project=config.wandb_project, group="timm", entity=config.wandb_entity, config=config):
        config = wandb.config
        dls, metrics = get_dataset(config.dataset, config.batch_size, config.img_size, config.seed, config.resize_method)
        if config.clip_checkpoint is not None:
            model = get_clip_model(config.model_name, config.clip_checkpoint, dls.c)
            mean, std =  (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
            dls.add_tfms([Normalize.from_stats(mean, std)], 'after_batch')
            learn = Learner(
                    dls, model, metrics=metrics,
                    splitter=default_split if config.split_func=="default" else None,
                    cbs=WandbCallback(log=None, log_preds=False)).to_fp16()
        else:
            if config.use_torchvision: 
                config.model_name = getattr(tv.models, config.model_name)
        learn = vision_learner(
                dls, config.model_name, metrics=metrics, concat_pool=(config.pool=="concat"),
                splitter=default_split if config.split_func=="default" else None,
                cbs=WandbCallback(log=None, log_preds=False)).to_fp16()
        ti = time.perf_counter()
        learn.fine_tune(config.epochs, config.learning_rate)
        wandb.summary["GPU_mem"] = get_gpu_mem(learn.dls.device)
        wandb.summary["model_family"] = config.model_name.split('_')[0]
        wandb.summary["fit_time"] = time.perf_counter() - ti

if __name__ == "__main__":
    args = parse_args()
    train(config=args)
