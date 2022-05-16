import wandb, argparse, pandas
from types import SimpleNamespace
from fine_tune import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--sweep_count', type=int, default=None)
    parser.add_argument('--sweep_method', type=str, default="grid")
    # parser.add_argument('--sweep_goal', type=str, default="minimize")
    # parser.add_argument('--sweep_metric_name', type=str, default="error_rate")
    # parser.add_argument('--early_terminate_min_iter', type=int, default=5)
    # parser.add_argument('--early_terminate_type', type=str, default="hyperband")
    return parser.parse_args()

models = pandas.read_csv("timm_models.csv").model.to_list()

def do_sweep():
    args = parse_args()
    sweep_configs = {
        "method": args.sweep_method,
        # "metric": {"name": args.sweep_metric, 
        #            "goal": args.sweep_goal},
        # "early_terminate": {
        #     "type": args.early_terminate_type,
        #     "min_iter": args.early_terminate_min_iter,
        # },
        "parameters": {
            "model_name": {"values": models},
            # "split_func": {"values": ["default", "timm"]},
            "concat_pool": {"values": [True, False]},
            "resize_method":{"values":["crop", "squish"]},
            "num_experiments": {"values": [1,2,3]},  #just to log the number of the exp
        },
    }
    if args.sweep_id is None:
        sweep_id = wandb.sweep(
            sweep_configs,
            project="fine_tune_timm",
        )
    else:
        print(f"Attaching runs to sweep: {args.sweep_id}")
        sweep_id = "fine_tune_timm"+"/"+args.sweep_id
    wandb.agent(sweep_id, function=train, count=args.sweep_count)


if __name__ == "__main__":
    do_sweep()

