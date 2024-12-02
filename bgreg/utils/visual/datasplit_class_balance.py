import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


def class_balance(preictal_size, ictal_size, postictal_size, train_indices, val_indices, plot=False, savefig=False):
    """
    Plotting scatter plot to display the distribution of train, validation, and test datasets
    across an iEEG signal spectrum divided in preictal, ictal, and postictal signal traces
    :param preictal_size: int, size of preictal signal trace
    :param ictal_size: int, size of ictal signal trace
    :param postictal_size: int, size of postictal signal trace
    :param train_indices: list of integers, indices of signal entries forming the train dataset
    :param val_indices: list of integers, indices of signal entries forming the validation dataset
    :param plot: bool, show plot operator
    :param savefig: bool, save plot operator
    :return:
    """
    # get total number of indices (signal length)
    indices = [i for i in range(preictal_size + ictal_size + postictal_size)]

    # set font and figures params
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial Narrow"
    plt.rcParams["figure.figsize"] = (9, 4)
    marker_size = 300

    # coloring preictal, ictal, and postictal background
    # setting colors for signal traces
    preictal_color = "pink"
    ictal_color = "salmon"
    postictal_color = "red"

    # point where to display text of signal trace label (i.e., preictal)
    mid_point_preictal = preictal_size / 2 - 10
    # plot text
    plt.text(mid_point_preictal, 0.517, "preictal", color=preictal_color, fontsize='small')
    # color background of signal trace (i.e., preictal)
    plt.axvspan(0, preictal_size, facecolor=preictal_color, alpha=0.4, label='_nolegend_')

    mid_point_ictal = (preictal_size + ictal_size) - (ictal_size / 2) - 10
    plt.text(mid_point_ictal, 0.517, "ictal", color=ictal_color, fontsize='small')
    plt.axvspan(preictal_size, preictal_size + ictal_size, facecolor=ictal_color, alpha=0.4, label='_nolegend_')

    mid_point_postictal = (preictal_size + ictal_size) + postictal_size / 2 - 10
    plt.text(mid_point_postictal, 0.517, "postictal", color=postictal_color, fontsize='small')
    plt.axvspan(preictal_size + ictal_size, len(indices), facecolor=postictal_color, alpha=0.4, label='_nolegend_')

    # assign brown color to train class, lightblue and teal for val and test, respectively
    colors = []
    train_color = "brown"
    val_color = "lightblue"
    test_color = "teal"
    for i in indices:
        if i in train_indices:
            colors.append(train_color)
        elif i in val_indices:
            colors.append(val_color)
        else:
            colors.append(test_color)

    # flags to control labels
    legend_control_train = 0
    legend_control_val = 0
    legend_control_test = 0
    yvalue = 0.504
    # scatter plot sequentially by color
    for i in indices:
        if colors[i] == train_color:
            if legend_control_train == 0:
                plt.scatter(indices[i], yvalue, s=marker_size * 0.6, marker="x", color=colors[i], label="Train dataset")
                legend_control_train = 1
            else:
                plt.scatter(indices[i], yvalue, s=marker_size * 0.6, marker="x", color=colors[i], label="_nolegend_")
        elif colors[i] == val_color:
            if legend_control_val == 0:
                plt.scatter(indices[i], yvalue, s=marker_size * 0.8, marker="s", facecolors=colors[i],
                            edgecolors=colors[i], label="Validation dataset")
                legend_control_val = 1
            else:
                plt.scatter(indices[i], yvalue, s=marker_size * 0.8, marker="s", facecolors=colors[i],
                            edgecolors=colors[i], label="_nolegend_")
        else:
            if legend_control_test == 0 and legend_control_val == 1:
                plt.scatter(indices[i], yvalue, s=marker_size, facecolors=colors[i],
                            edgecolors=colors[i], label="Test dataset")
                legend_control_test = 1
            else:
                plt.scatter(indices[i], yvalue, s=marker_size, facecolors=colors[i],
                            edgecolors=colors[i], label="_nolegend_")

    # seizure onset mark with dashed line
    plt.axvline(preictal_size, color="k", linestyle="--")
    plt.text(preictal_size, 0.51, "  seizure onset", fontsize='medium')
    # seizure offset mark with dashed line
    plt.axvline(preictal_size + ictal_size, color="k", linestyle="--")
    plt.text(preictal_size + ictal_size, 0.51, "  seizure offset", fontsize='medium')

    # customize ticks and limiters
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelleft=False)
    # Hide Y axes tick marks
    ax.set_yticks([])
    ax.set_xlabel('Dataset samples')
    ax.set_ylim(0.49, 0.52)
    ax.set_xlim(0, len(indices))

    # display legend at lower center
    ax.legend(loc="lower center", facecolor="white", edgecolor="k", fancybox=True)

    if plot:
        plt.show()
    if savefig:
        plt.savefig("datasplit_class_balance.pdf", bbox_inches='tight', pad_inches=0.1)

