import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


def plt_save_pdf(filename):
    pp = PdfPages(filename)
    pp.savefig()
    pp.close()


def plot_colormap(inputs,
                  show=True,
                  cmap='inferno',
                  draw_axis=True,
                  title='',
                  xlabel='',
                  ylabel='',
                  save=False,
                  **kwargs):
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
    if draw_axis:
        plt.colorbar(colorplot)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    if save:
        if title:
            plt_save_pdf('/tmp/%s.pdf' % title)
        else:
            plt_save_pdf('/tmp/%s.pdf' % str(datetime.now()))
    if show:
        plt.show()
