import wandb
from collections import deque
import torch
import matplotlib.pyplot as plt

import fp_modules as fp
from utils.fp_utils import shanon_entropy_binned as entropy


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


class TrailingStats:
    """
    Records stats and plots them
    """

    def __init__(self, maxsize=None):
        self.q = deque(maxlen=maxsize)
        self.maxsize = maxsize
        self.var_names = set()

    """
    data - dictionary of string -> number
    """

    def put(self, data):
        self.q.append(data)
        self.var_names.update(data.keys())

    def __len__(self):
        return len(self.q)

    def report(self):
        for n in self.var_names:
            values = torch.empty(len(self))
            for i, d in enumerate(self.q):
                values[i] = d.get(n, torch.tensor(float('nan')))
            plt.xlim([0, len(self)])
            minv, avgv, maxv = (round(torch.min(values).item(), 2),
                                round(torch.mean(values).item(), 2),
                                round(torch.max(values).item(), 2))
            plt.title(f"{n}: min={minv}, avg={avgv}, max={maxv}")
            plt.plot(values.numpy(), label=n)
            plt.show()

    def add_from_layer(self, layer: fp.FpLinear):
        # FIXME: this accesses internal fields in the layer - refactor
        weights_entropy = torch.tensor([entropy(layer.weight.data[i]) for i in range(layer.out_features)])

        layer_stats = {
            'excited_count': torch.count_nonzero(layer.excitations).item(),
            # FIXME
            # 'output_std': torch.std(layer.output).item(),
            # 'output_min': layer.output.min().item(),
            # 'output_max': layer.output.max().item(),
            # 'output_avg': layer.output.mean().item(),
            # 'output_entropy': entropy(layer.output).item(),
            'avg_weights_entropy': torch.mean(weights_entropy).item(),
            'std_weights_entropy': torch.std(weights_entropy).item()
        }

        self.put(layer_stats)

        return layer_stats
