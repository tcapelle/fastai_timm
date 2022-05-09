import wandb, argparse
from jeremy_ft import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--sweep_count', type=int, default=30)
    parser.add_argument('--sweep_method', type=str, default="bayes")
    parser.add_argument('--sweep_goal', type=str, default="minimize")
    parser.add_argument('--sweep_metric_name', type=str, default="error_rate")
    parser.add_argument('--early_terminate_min_iter', type=int, default=5)
    parser.add_argument('--early_terminate_type', type=str, default="hyperband")
    parser.add_argument('--model_name', type=str, default="resnet34")
    return parser.parse_args()

def do_sweep():
    args = parse_args()
    sweep_configs = {
        "method": args.sweep_method,
        "metric": {"name": args.sweep_metric_name, 
                   "goal": args.sweep_goal},
        "early_terminate": {
            "type": args.early_terminate_type,
            "min_iter": args.early_terminate_min_iter,
        },
        "parameters": {
            "model_name": {"value": args.model_name},
            "fit_learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "ft_learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "fit_epochs": {"values": [0,1]},
            "split_func": {"values": ["default", "timm"]},
        },
    }
    if args.sweep_id is not None:
        sweep_id = wandb.sweep(
            sweep_configs,
            project="fine_tune_timm",
        )
    wandb.agent(sweep_id, function=train, count=args.sweep_count)


if __name__ == "__main__":
    do_sweep()

