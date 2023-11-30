import matplotlib.pyplot as plt
import numpy as np
import torch


def show_center_recep_field(img, out):
    """Calculates the gradients of the input with respect to the output center pixel, and visualizes the overall
    receptive field.

    Args:
        img: Input image for which we want to calculate the receptive field on.
        out: Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients
    if img.shape[2] == 1:
        loss = out[..., img.shape[3] -1].sum()  
    else:
        loss = out[..., img.shape[2] - 1, img.shape[3] // 2].sum() 
    # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    loss.backward(retain_graph=True)
    img_grads = img.grad.abs()
    img.grad.fill_(0)  # Reset grads

    # Plot receptive field
    img = img_grads.squeeze().cpu().numpy()
    _, ax = plt.subplots()
    if img.ndim == 1:
        ax.plot(img > 0, "o")
        ax.set_xlabel("Time")
        ax.set_ylabel("Binary receptive field")
    else:
        ax.imshow(img > 0)
        ax.set_title("Binary receptive field")
    plt.show()
    plt.close()


def view_receptive_field(noise_model, img_shape):
    inp_img = torch.zeros(1, 1, *img_shape).requires_grad_()
    out = noise_model(inp_img)
    show_center_recep_field(inp_img, out[:, [0]])


def autocorrelation(a, max_lag=100, title=None):
    """Calculates the autocorrelation of a 1D or 2D array.

    Args:
        a: Input array.
        max_lag: Maximum lag to calculate the autocorrelation for.
    """

    a = a-a.mean()
    if a.shape[-2] == 1:
        results = np.zeros((1, max_lag))
        max_v_lag = 1
    else:
        results = np.zeros((max_lag, max_lag))
        max_v_lag = max_lag
    for i in range(0, max_v_lag):
        for j in range(0, max_lag):
            if i == 0 and j == 0:
                covar = np.mean(a**2)
            if i == 0 and j != 0:
                covar = np.mean(a[...,j:]*a[...,:-j])
            if j == 0 and i != 0:
                covar = np.mean(a[...,i:,:]*a[...,:-i,:])
            if i != 0 and j != 0:
                covar = np.mean(a[...,i:,j:]*a[...,:-i,:-j])
            results[i, j] = covar

    ac = results/(a**2).mean()
    if a.shape[-2] == 1:
        plt.plot(ac[0])
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
    else:
        plt.imshow(ac, vmin=-1, vmax=1, cmap="seismic")
        plt.colorbar()
        plt.xlabel("Vertical lag")
        plt.ylabel("Horizontal lag")
    if title is not None:
        plt.title(title)
    plt.show()


def plot(arr, titles=None, vmin=None, vmax=None, cmap="inferno", colorbar=False, figsize=(10, 5)):
    """Plots a list of 1D or 2D arrays.

    If each element of arr is 1D (shape [N, C, 1, W] or [N, 1, W]), 
    they will be plotted as line plots. 
    If each element of arr is 2D, (shape [N, C, H, W] or [N, H, W]),
    they will be plotted as images.

    Args:
        arr: List of arrays to plot.
        title: List of titles for the plot.
        vmin: Minimum value for the colorbar.
        vmax: Maximum value for the colorbar.
    """
    arr = [a.squeeze() for a in arr]
    if arr[0].ndim == 1:
        plt.figure(figsize=figsize)
        for i, a in enumerate(arr):
            plt.plot(a, label=titles[i])
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
    else:
        _, ax = plt.subplots(1, len(arr), figsize=figsize)
        if len(arr) == 1:
            ax = [ax]
        for i, a in enumerate(arr):
            ax[i].imshow(a, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[i].set_title(titles[i])
            if colorbar:
                ax[i].colorbar()
    plt.show()