import argparse
from fastai.vision.all import *

config_defaults = SimpleNamespace(
    batch_size=32,
    epochs=5,
    num_experiments=1,
    learning_rate=8e-3,
    img_size=224,
    resize_method="crop",
    model_name="resnet34",
    pool="concat",
    seed=42,
    split_func="default",
    dataset="PLANETS",
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
    parser.add_argument('--split_func', type=str, default=config_defaults.split_func)
    parser.add_argument('--pool', type=str, default=config_defaults.pool)
    parser.add_argument('--seed', type=int, default=config_defaults.seed)
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    return parser.parse_args()

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
    else: raise Exception(f"Dataset {dataset_name} not found, supports: PETS or PLANETS")

def train(config=config_defaults):
    dls, metrics = get_dataset(config.dataset, config.batch_size, config.img_size, config.seed, config.resize_method)
    learn = vision_learner(
            dls, config.model_name, metrics=metrics, concat_pool=(config.pool=="concat"),
            splitter=default_split if config.split_func=="default" else None).to_fp16()
    ti = time.perf_counter()
    learn.fine_tune(config.epochs, config.learning_rate)

if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments): train(config=args)

