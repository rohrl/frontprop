import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import fp_modules as fp


def plot_matrix(dims, *datas):
    """
    Plot the 2D matrix (takes multiple).
    They can be 2D, or flattened (then will use `dims` to set width/height).
    :param dims:
    :param datas:
    :return:
    """
    PLOTS_PER_ROW = 20
    cols = min(len(datas), PLOTS_PER_ROW)
    rows = math.ceil(len(datas) / PLOTS_PER_ROW)
    fig = plt.figure(figsize=(cols, rows))
    for i, data in enumerate(datas):
        data = data.cpu().numpy()
        if data.shape != dims:
            data = np.reshape(data, dims)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(data, cmap='gray')
        plt.axis('off')
        # plt.title(f"#{i}")
    plt.show()


def shanon_entropy_binned(numbers, range=(0, 1), bins=100):
    """
    Calculate discrete entropy for continuous values, by binning them.
    Note this is Shannon's measure of information entropy.
    Not cross-entropy loss, used commonly in training.

    TODO: check if this is correct
    """
    # h, bin_edges = np.histogram(numbers, bins=bins, density=False, range=range)
    # h = h / h.max() # normalise
    # entropy = scipy.stats.entropy(h)

    # Calculate histogram
    hist = torch.histc(numbers, bins=bins, min=range[0], max=range[1])
    # Normalize histogram
    hist /= hist.max()
    # Calculate entropy
    hist += torch.finfo(torch.float32).eps
    entropy = -(hist * torch.log2(hist)).sum()

    return entropy


def sphere_rnd_gen(n, d):
    """
    Generate n random points on a d-dimensional sphere.
    See https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere

    This is essential if we're normalising to unit length (ie projecting onto a hypersphere),
    because a uniform distribution from a unit cube won't be uniform on the sphere.
    """
    x = torch.randn(n, d)
    x /= torch.norm(x, dim=1, keepdim=True)
    return x


def draw_layer(layer: fp.FpLinear, dims):
    """Draw neuron's weights."""
    print("Neurons' weights:")
    plot_matrix(dims, *([layer.weight.data[i] for i in range(layer.out_features)]))
