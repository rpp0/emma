from emma.utils import visualizations
import numpy as np
import matplotlib.pyplot as plt


def get_gradients(conf, model, examples_batch, kerasvis=False):
    if kerasvis:
        from vis.visualization import visualize_saliency
    else:
        visualize_saliency = None  # Get rid of PyCharm warning
    subkey_gradients = []

    # Get number of outputs
    num_outputs = model.model.output.shape[1]

    # Get gradients for each subkey
    for neuron_index in range(0, num_outputs):
        if kerasvis is False:
            gradients = model.get_output_gradients(neuron_index,
                                                   examples_batch,
                                                   square_gradients=True,
                                                   mean_of_gradients=conf.saliency_mean_gradient)
        else:
            gradients = np.zeros(examples_batch.shape)
            for i in range(0, examples_batch.shape[0]):
                gradients[i, :] = visualize_saliency(model.model, -1, filter_indices=neuron_index, seed_input=examples_batch[i, :])

        subkey_gradients.append(gradients)
    return subkey_gradients


def plot_saliency_2d_overlay(conf, salvis_result):
    """
    Plot saliency over a gray colormap of the EM traces, seperately for each subkey.
    :param conf:
    :param salvis_result:
    :return:
    """
    for neuron_index in range(len(salvis_result.gradients)):
        # Plot the result
        visualizations.plot_colormap(salvis_result.examples_batch,
                                     cmap='gray',
                                     show=False,
                                     draw_axis=False,
                                     alpha=1.0)
        visualizations.plot_colormap(salvis_result.gradients[neuron_index],
                                     cmap='inferno',
                                     show=True,
                                     draw_axis=False,
                                     alpha=0.8,
                                     title='Neuron: %d' % neuron_index,
                                     xlabel='Time (samples)',
                                     ylabel='Trace index')


def plot_saliency_2d(conf, salvis_result):
    """
    First plots the input batch using a color map, and then plots the saliency color maps
    for each subkey separately.
    :param conf:
    :param salvis_result:
    :return:
    """
    visualizations.plot_colormap(salvis_result.examples_batch, cmap='plasma')

    for neuron_index in range(len(salvis_result.gradients)):
        visualizations.plot_colormap(salvis_result.gradients[neuron_index])


def plot_saliency_1d(conf, salvis_result):
    """
    Takes the mean signal of a batch and then plots a time series of this signal, overlayed
    with the saliency for each subkey.
    :param conf:
    :param salvis_result:
    :return:
    """
    from emma.processing.dsp import normalize

    # Get mean signal of examples batch
    mean_signal = np.mean(salvis_result.examples_batch, axis=0)

    plt.plot(normalize(mean_signal), color='tab:blue', label='Mean signal (normalized)')

    for neuron_index in range(len(salvis_result.gradients)):
        # Get gradient of mean signal
        gradients = salvis_result.gradients[neuron_index]
        mean_gradient = gradients[0]

        # Visualize mean gradients
        plt.plot(normalize(mean_gradient), label='Mean neuron %d gradient (normalized)' % neuron_index, alpha=0.6)
    plt.legend()
    plt.show()


def plot_saliency_kerasvis(conf, salvis_result):
    for neuron_index in range(len(salvis_result.gradients)):
        visualizations.plot_colormap(salvis_result.gradients[neuron_index])


def plot_saliency_2d_overlayold(conf, salvis_result):
    from matplotlib.colors import ListedColormap

    visualizations.plot_colormap(salvis_result.examples_batch, cmap='gray', show=False, draw_axis=False)

    colormaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    #  colormaps = ['inferno', 'inferno', 'inferno', 'inferno', 'inferno']
    for neuron_index in range(len(salvis_result.gradients)):
        gradients = salvis_result.gradients[neuron_index]

        # Plot
        cmap_orig = plt.get_cmap(colormaps.pop(0))
        cmap_alpha = cmap_orig(np.arange(cmap_orig.N))
        cmap_alpha[:, -1] = np.linspace(0, 1, cmap_orig.N)  # dim 0: pixel, dim 1: channel
        cmap_alpha = ListedColormap(cmap_alpha)
        visualizations.plot_colormap(gradients, cmap=cmap_alpha, show=False, draw_axis=False)
    plt.show()
