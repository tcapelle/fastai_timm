import wandb, argparse
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

GROUP_NAME = "fit-streategies"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fit_epochs', type=int, default=0)
    parser.add_argument('--ft_epochs', type=int, default=10)
    parser.add_argument('--num_experiments', type=int, default=1)
    parser.add_argument('--fit_learning_rate', type=float, default=0.002)
    parser.add_argument('--ft_learning_rate', type=float, default=0.002)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--split_func', type=str, default="default")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_torchvision', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='fine_tune_timm')
    return parser.parse_args()


def get_pets(batch_size, img_size, seed):
    "The dog breeds pets datasets"
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

def get_learner(args, **kwargs):
    "A simple vision learner form a timm backbone"
    dls = get_pets(args.batch_size, args.img_size, args.seed)
    if args.force_torchvision: args.model_name = getattr(models, args.model_name)
    learn = vision_learner(dls, 
                           args.model_name, 
                           metrics=[accuracy, error_rate], 
                           splitter=default_split if args.split_func=="default" else None,
                           pretrained=True, **kwargs).to_fp16()
    return learn

def train():
    args = parse_args()
    with wandb.init(project=args.wandb_project, group=GROUP_NAME, config=args):
        args = wandb.config  # enables override by wandb.sweeps
        learn = get_learner(args, cbs=[WandbCallback(log_preds=False)])
        if args.fit_epochs>0: learn.fit(args.fit_epochs, args.fit_learning_rate)
        learn.fine_tune(args.ft_epochs, args.ft_learning_rate)
        
if __name__ == "__main__":
    train()