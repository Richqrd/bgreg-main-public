from bgreg.utils.dirtree import save_matrix_to_image_file
from bgreg.utils.visual import utilities as utils
from bgreg.data.io import strip_channel_names
import numpy as np
import matplotlib.pyplot as plt


def plot_matrices(subject, matrices, channels=None, title='', plot=False, savemat=False, mat_type=None, trace_type=None):
    """
    Plot matrices

    :param subject: (str) patient id
    :param matrices: (NumPy 3D array) Numerical matrices
    :param channels: (list of str) List of channel names
    :param title: (str) title of figure and file
    :param plot: (bool) visualize plot operator
    :param savemat: (bool) operator to save matrices to files, default False
    :param mat_type: (str) type of matrix, only used if savemat=True
    :param trace_type: (str) signal trace type, only used if savemat=True
    :return:
    """
    # Set number of subplots
    subset = range(matrices.shape[-1] - 1)
    if len(matrices.shape) == 2:
        subset = range(1)
        matrices = np.atleast_3d(matrices)

    if channels is None:
        channels = []
    ticks = strip_channel_names(channels)

    # Create subplot objects
    fig, ax, ax1D = utils.custom_subplots(len(subset))

    # plot on each subplot
    for subplot_id, subplot in enumerate(subset):
        ax1D[subplot_id].set_title('')
        plot_matrix_single(matrices[:, :, subplot], fig=fig, ax=ax1D[subplot_id])

        # Set ticks
        # ax1D[subplot_id].set_yticks(range(len(ticks)), ticks)
        # ax1D[subplot_id].set_xticks(range(len(ticks)), ticks)
        # ax1D[subplot_id].tick_params(labelsize=6)
        # ax1D[subplot_id].tick_params('x', rotation=90)
        plt.axis('off')

    plt.suptitle(title, fontsize=14)
    if savemat:
        save_matrix_to_image_file(subject, fig, title, mat_type, trace_type)
    if plot:
        plt.show()


def plot_matrix_single(matrix, ax=None, fig=None):
    """
    Plot single matrix in ax within fig
    :param matrix:
    :param ax: (matplotlib.ax)
    :param fig: (matplotlib.fig)
    :return:
    """
    if ax is not None and fig is not None:
        Vm = np.percentile(matrix, 99)
        im = ax.imshow(matrix, clim=[-Vm, Vm], cmap=matrix_cmap(),
                       interpolation='none', aspect='auto')
        ax.set_axis_off()
        # cbar = fig.colorbar(im, extend='neither', spacing='proportional',
        #                     orientation='vertical', format="%.0f", ax=ax)
        # cbar.ax.tick_params(labelsize=6)


def matrix_cmap():
    # FIXME: Parameterized for different colors
    return 'GnBu'
