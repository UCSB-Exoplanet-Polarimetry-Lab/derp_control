import numpy as np
import matplotlib.pyplot as plt

def plot_4x4_grid(array, title=None, vmin=None, vmax=None, cmap='viridis'):
    """
    Plots an N x N x 4 x 4 array into a 4 x 4 grid of subplots with a shared colorbar.

    Parameters:
        array: numpy.ndarray
            The data array of shape (N, N, 4, 4).
        title: str, optional
            Title for the figure.
        vmin, vmax: float, optional
            Color limits for the plots.
        cmap: str, optional
            Colormap to use.
    """

    if array.shape[2:] != (4, 4):
        raise ValueError("Input array must have shape (N, N, 4, 4)")

    N = array.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    # Find global vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    # Plot each 4x4 slice
    ims = []
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            im = ax.imshow(array[:, :, i, j], vmin=vmin, vmax=vmax, cmap=cmap, origin='upper')
            ims.append(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({i},{j})")
    # Add a single colorbar
    fig.colorbar(ims[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    if title is not None:
        fig.suptitle(title)
    plt.show()
