import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from traceset import TraceSet


def plt_save_pdf(path):
    """
    Save plot as pdf to path
    :param path:
    :return:
    """
    pp = PdfPages(path)
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
    :param draw_axis:
    :param title:
    :param cmap:
    :param xlabel:
    :param ylabel:
    :param save:
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


def plot_trace_set(reference_signal, trace_set, params=None):
    """
    Plot a trace set using matplotlib
    """
    maxplots = 32
    saveplot = False

    if params is not None:
        if len(params) == 1:
            if 'save' in params:
                saveplot = True
            else:
                maxplots = int(params[0])
        elif len(params) == 2:
            maxplots = int(params[0])
            saveplot = params[1] == 'save'

    if not isinstance(trace_set, TraceSet):
        raise ValueError("Expected TraceSet")

    count = 0
    for trace in trace_set.traces:
        plt.plot(range(0, len(trace.signal)), trace.signal)
        count += 1
        if count >= maxplots:
            break
    plt.plot(range(0, len(reference_signal)), reference_signal, linewidth=2, linestyle='dashed')

    plt.title(trace_set.name)

    if saveplot:
        plt_save_pdf('/tmp/%s.pdf' % trace_set.name)
        plt.clf()
    else:
        plt.show()
