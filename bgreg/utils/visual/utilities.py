import matplotlib.pyplot as plt
import numpy as np


def get_pairs(list1, list2, includeSame=False, unique=True):
    """
    Get all pairs (unique or not) between elements in list1 and list2.

    If same element is in both lists include pair if includeSame == True.
    """

    pairs = []
    for aa in list1:
        for bb in list2:
            if aa == bb and not includeSame:
                pass
            else:
                if unique:
                    if (bb, aa) not in pairs:
                        pairs.append((aa, bb))
                else:
                    pairs.append((aa, bb))

    return pairs


def custom_subplots(nSubplots, figsize=(16, 16), **kwargs):
    """
    Standard figure for creating n subplots
    """

    sz = int(np.ceil(np.sqrt(nSubplots)))
    if (sz ** 2 - sz) >= nSubplots:
        nRow = sz - 1
        nCol = sz
    else:
        nRow = sz
        nCol = sz

    fig, ax = plt.subplots(nRow, nCol, figsize=figsize, **kwargs)

    # Remove unused axes from figure
    subplotPairs = get_pairs(range(nRow), range(nCol), includeSame=True, unique=False)
    for ss in subplotPairs[nSubplots:]:
        ax[ss[0], ss[1]].axis('off')
    ax1D = np.ravel(ax)

    return fig, ax, ax1D
