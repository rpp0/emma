import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from emma.io.traceset import TraceSet
from emma.utils.utils import MaxPlotsReached, EMMAException


#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('flag').colors)  # Use different cycling colors
#plt.style.use('bmh')  # Use different style


def plt_save_pdf(path):
    """
    Save plot as pdf to path
    :param path:
    :return:
    """
    pp = PdfPages(path)
    pp.savefig(dpi=300)
    pp.close()
    plt.clf()
    plt.cla()


def plot_spectogram(trace_set,
                    sample_rate,
                    nfft=2**10,
                    noverlap=0,
                    cmap='plasma',
                    params=None,
                    num_traces=1024):

    if not trace_set.windowed:
        raise EMMAException("Trace set should be windowed")

    # Check params
    if params is not None:
        if len(params) == 1:
            nfft = int(params[0])
        elif len(params) == 2:
            nfft = int(params[0])
            noverlap = int(nfft * int(params[1]) / 100.0)

    all_signals = np.array([trace.signal for trace in trace_set.traces[0:num_traces]]).flatten()

    """
    # Old style
    for trace in trace_set.traces[0:num_traces]:
        plt.specgram(trace.signal, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap=cmap)
    """

    plt.specgram(all_signals, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap=cmap, mode='psd', scale='dB')
    plt.tight_layout()
    plt.show()


def plot_colormap(inputs,
                  show=True,
                  cmap='inferno',
                  draw_axis=True,
                  title='',
                  xlabel='',
                  ylabel='',
                  colorbar_label='',
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
    :param colorbar_label:
    :param save:
    :param kwargs:
    :return:
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if inputs.dtype == np.complex64 or inputs.dtype == np.complex128:
        inputs = np.real(inputs)
        print("Warning: converting colormap to np.real(complex)")
    #inputs += 0.01
    vmin = inputs.min()
    vmax = inputs.max()
    colorplot = plt.imshow(inputs,
                           vmin=vmin,
                           vmax=vmax,
                           interpolation='nearest',
                           # norm=LogNorm(vmin=vmin, vmax=vmax),
                           cmap=cmap,
                           **kwargs)
    if draw_axis:
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        axis = plt.gca()
        figure = plt.gcf()
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = figure.colorbar(colorplot, cax=cax)
        cbar.set_label(colorbar_label)
    plt.tight_layout()
    if save:
        if title:
            plt_save_pdf('/tmp/%s.pdf' % title)
        else:
            plt_save_pdf('/tmp/%s.pdf' % str(datetime.now()))
    if show:
        plt.show()


def _get_x_axis_values(signal, time_domain=True, sample_rate=1.0):
    if not time_domain:
        freqs = np.fft.fftfreq(len(signal), d=1.0/sample_rate)
        x = np.fft.fftshift(freqs)
    else:
        x = range(0, len(signal))

    return x


def plot_trace_sets(reference_signal,
                    trace_sets,
                    params=None,
                    no_reference_plot=False,
                    num_traces=1024,
                    title='',
                    xlabel='',
                    ylabel='',
                    colorbar_label='',
                    time_domain=True,
                    sample_rate=1.0):
    """
    Plot num_traces signals from a list of trace sets using matplotlib
    """
    saveplot = False
    colormap = False

    # Check params
    if params is not None:
        if len(params) >= 1:
            if 'save' in params:
                saveplot = True
            if '2d' in params:
                colormap = True

    if not isinstance(trace_sets, list) or isinstance(trace_sets, TraceSet):
        raise ValueError("Expected list of TraceSets")
    if len(trace_sets) == 0:
        return

    # Make title
    common_path = os.path.commonprefix([trace_set.name for trace_set in trace_sets])
    if title == '':
        title = "%d trace sets from %s" % (len(trace_sets), common_path)
    if reference_signal.dtype == np.complex64 or reference_signal.dtype == np.complex128:
        title += " (complex, only real values plotted)"

    # Make plots
    count = 0
    all_signals = []
    try:
        for trace_set in trace_sets:
            for trace in trace_set.traces:
                all_signals.append(trace.signal)
                count += 1
                if count >= num_traces:
                    raise MaxPlotsReached
    except MaxPlotsReached:
        pass
    finally:
        if xlabel == '':
            if time_domain:
                xlabel = 'Samples'
            else:
                xlabel = 'Frequency (assuming sample rate %.2f)' % sample_rate

        if colormap:
            plot_colormap(np.array(all_signals),
                          show=False,
                          title=title,
                          xlabel=xlabel,
                          ylabel=ylabel,
                          colorbar_label=colorbar_label)
        else:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            for signal in all_signals:
                x = _get_x_axis_values(signal, sample_rate=sample_rate, time_domain=time_domain)
                plt.plot(x, signal)
            if not no_reference_plot:
                x = _get_x_axis_values(reference_signal, sample_rate=sample_rate, time_domain=time_domain)
                plt.plot(x, reference_signal, linewidth=2, linestyle='dashed')

    if saveplot:
        plt_save_pdf('/tmp/plotted_trace_sets.pdf')
        plt.clf()
    else:
        plt.show()


def plot_correlations(values1, values2, label1="", label2="", show=False):
    values1 = np.reshape(values1, (-1,))  # TODO doesnt account for numkeys. Use only for a single key byte!
    values2 = np.reshape(values2, (-1,))
    correlation = np.corrcoef(values1, values2, rowvar=False)[1, 0]
    mean_values1 = np.mean(values1, axis=0)
    mean_values2 = np.mean(values2, axis=0)
    plt.title("Correlation: " + str(correlation))
    plt.plot(values1, "o", label=label1, markersize=5.0)
    plt.plot(values2, "o", label=label2, markersize=5.0)
    #plt.plot(values1, values2, "o", label=label2, markersize=5.0)
    plt.gca().legend()
    if show:
        plt.show()


def plot_keyplot(keyplot, time_domain=True, sample_rate=1.0, show=False):
    plt.title("Keyplot")
    if time_domain:
        plt.xlabel("Samples")
    else:
        plt.xlabel("Frequency assuming sample rate of %.2f" % sample_rate)
    plt.ylabel("Amplitude")

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for value, mean_signal in sorted(keyplot.items()):
        color = color_cycle[int(value.rpartition(',')[2].strip(')'), 16) % len(color_cycle)]
        x = _get_x_axis_values(mean_signal, sample_rate=sample_rate, time_domain=time_domain)
        plt.plot(x, mean_signal, label=value, color=color)
    plt.legend()

    if show:
        plt.show()
