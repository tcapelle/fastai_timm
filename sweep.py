import wandb
from functools import partial
from types import SimpleNamespace
from jeremy_ft import train

config = SimpleNamespace(sweep_method="bayes",
                         sweep_goal="minimize",
                         sweep_metric_name="error_rate",
                         early_terminate_type = "hyperband",
                         early_terminate_min_iter = 5, 
                         sweep_count = 30
                         )


def do_sweep():
    sweep_configs = {
        "method": config.sweep_method,
        "metric": {"name": config.sweep_metric_name, "goal": config.sweep_goal},
        "early_terminate": {
            "type": config.early_terminate_type,
            "min_iter": config.early_terminate_min_iter,
        },
        "parameters": {
            "model_name": {
                "values": [
                    "haloregnetz_b",
                ]
            },
            "fit_learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "ft_learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
            "fit_epochs": {"values": [0,1]},
            "split_func": {"values": ["default", "timm"]},
        },
    }
    sweep_id = wandb.sweep(
        sweep_configs,
        project="fine_tune_timm",
    )
    wandb.agent(sweep_id, function=train, count=config.sweep_count)


if __name__ == "__main__":
    do_sweep()

