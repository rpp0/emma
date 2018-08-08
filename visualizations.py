import matplotlib.pyplot as plt


def plot_colormap(inputs, show=True, cmap='inferno', **kwargs):
    """
    Plot signals given in the inputs numpy array in a colormap.
    :param inputs:
    :param show:
    :param cmap:
    :param kwargs:
    :return:
    """
    vmin = inputs.min()
    vmax = inputs.max()
    colorplot = plt.imshow(inputs,
                           vmin=vmin,
                           vmax=vmax,
                           interpolation='nearest',
                           cmap=cmap,
                           **kwargs)
    plt.colorbar(colorplot)
    plt.tight_layout()
    if show:
        plt.show()
