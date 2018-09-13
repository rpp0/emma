import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from traceset import TraceSet
import numpy as np


def plt_save_pdf(path):
    """
    Save plot as pdf to path
    :param path:
    :return:
    """
    pp = PdfPages(path)
    pp.savefig()
    pp.close()


def plot_spectogram(trace_set,
                    sample_rate,
                    nfft=128,
                    noverlap=64,
                    cmap='inferno',
                    params=None):
    if params is not None:
        max_traces = int(params[0])
    else:
        max_traces = len(trace_set.traces)

    for trace in trace_set.traces[0:max_traces]:
        plt.specgram(trace.signal, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap=cmap)
        plt.tight_layout()
        plt.show()


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
    if inputs.dtype == np.complex64 or inputs.dtype == np.complex128:
        inputs = np.real(inputs)
        print("Warning: converting colormap to np.real(complex)")
    vmin = inputs.min()
    vmax = inputs.max()
    colorplot = plt.imshow(inputs,
                           vmin=vmin,
                           vmax=vmax,
                           interpolation='nearest',
                           cmap=cmap,
                           **kwargs)
    if draw_axis:
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        axis = plt.gca()
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(colorplot, cax=cax)
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
    colormap = False

    if params is not None:
        if len(params) == 1:
            if 'save' in params:
                saveplot = True
            elif '2d' in params:
                colormap = True
                maxplots = 2048
            else:
                maxplots = int(params[0])
        elif len(params) == 2:
            maxplots = int(params[0])
            saveplot = params[1] == 'save'
            colormap = params[1] == '2d'

    if not isinstance(trace_set, TraceSet):
        raise ValueError("Expected TraceSet")

    if colormap:
        plot_colormap(np.array([trace.signal for trace in trace_set.traces[0:maxplots]]), show=False)
    else:
        count = 0
        for trace in trace_set.traces:
            plt.plot(range(0, len(trace.signal)), trace.signal)
            count += 1
            if count >= maxplots:
                break
        plt.plot(range(0, len(reference_signal)), reference_signal, linewidth=2, linestyle='dashed')

    title = trace_set.name
    if reference_signal.dtype == np.complex64 or reference_signal.dtype == np.complex128:
        title += " (complex, only real values plotted)"
    plt.title(title)

    if saveplot:
        plt_save_pdf('/tmp/%s.pdf' % trace_set.name)
        plt.clf()
    else:
        plt.show()
