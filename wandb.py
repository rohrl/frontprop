import wandb


def init_wandb(w_boost: float, t_decay: float,
               project="frontprop", dataset="MNIST",
               epochs=1, architecture="FC"):
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
            "epochs": epochs,
        }
    )


def log_metric(metrics_dict):
    wandb.log(metrics_dict)


def close_wandb():
    wandb.finish()
