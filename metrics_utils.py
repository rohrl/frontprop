import wandb


def init_wandb(w_boost: float, t_decay: float,
               neurons: int, iterations: int,
               project: str, dataset: str,
               architecture="FC"):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # track hyperparameters and run metadata
        config={
            "weight_boost": w_boost,
            "threshold_decay": t_decay,
            "architecture": architecture,
            "dataset": dataset,
            "iterations": iterations,
            "neurons": neurons
        }
    )


def log_metric(metrics_dict):
    wandb.log(metrics_dict)


def close_wandb():
    wandb.finish()
