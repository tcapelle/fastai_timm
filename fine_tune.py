import wandb
import timm
import argparse
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from torchvision import models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_experiments', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--force_torchvision', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='fine_tune_timm')
    return parser.parse_args()

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
    

    
def train():
    args = parse_args()
    group_name = "torchvision" if args.force_torchvision else "timm"
    for _ in range(args.num_experiments):
        for model_name in timm.list_models(args.model_name):
            print(f"Training {model_name}")
            with wandb.init(project=args.wandb_project, group=group_name, config=args):
                dls = get_dls(args.batch_size, args.img_size, args.seed)
                cbs = [MixedPrecision(), WandbCallback(log_preds=False)]
                if args.mixup: cbs.append(MixUp())
                if args.force_torchvision: 
                    model_name = getattr(models, model_name)
                learn = vision_learner(dls, 
                                       model_name, 
                                       metrics=[accuracy, error_rate], 
                                       cbs=cbs)
                learn.fine_tune(args.epochs, args.learning_rate)

if __name__ == "__main__":
    train()
    



