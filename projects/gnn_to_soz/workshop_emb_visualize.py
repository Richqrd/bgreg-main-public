import os
import numpy as np

import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
import pickle

from bgreg.data.analysis.graph.graph_representation import get_gres
from bgreg.native.datapaths import openneuro_graph_representations_dir


def emb_visualize(directory, type, embedding, labels, target=None, prediction=None, best_loss=None,
                   technique='tsne', subject=None, seed=1, parameters=None, layer_names=[]):
    """
        Visualize dimensionally reduced embeddings
        :param directory: string, path to save the visualizations"
        :param type: string, specify which features to visualize. Valid entries: node, graph
        :param embedding: list, contains embeddings from different layers as items. For node visualization, each
                                    layer embedding item should contain 1 iteration from a batch. For graph
                                    visualization, each layer embedding item should contain an entire batch.
        :param labels: list, strings of labels
        :param target: float, the ground truth of an iteration. For node classification only.
        :param prediction: float, the model prediction of an iteration. For node classification only.
        :param best_loss: float, loss of model that produced the embeddings. Valid entries: binary, multi.
        :param technique: string, the dimensionality reduction technique to use. Valid entries: tsne, umap, pca
        :param subject: string, ID of subject.
        :param seed: int, randomization seed for the stochastic processes in dimensionality reduction.
        :param parameters: object with parameters 'perplexity' and 'n_neighbours', parameters for dimensionality reduction techniques.
        :param layer_names: list, strings of names of each layer.
        :return:
        """

    # Close all previous plots
    plt.close('all')

    # configure pyplot parameters
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Arial Narrow"
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams['figure.figsize'] = [14, 8]

    fig, axs = plt.subplots(1, len(embedding))
    if len(embedding) == 1:
        axs = [axs]
    red_emb_list = [pd.DataFrame() for i in embedding]
    for index, layer_embedding in enumerate(embedding):
        # Dimensionality reduction here, producing a 2 dimensional list
        # perplexity and n_neighbours can be finetuned
        reduced_dimensions = dimensionality_reduction(layer_embedding, technique, seed=seed, parameters=parameters)

        red_emb_list[index]['x'] = reduced_dimensions[:, 0]
        red_emb_list[index]['x'] = red_emb_list[index]['x'].map(lambda x: round(x, 2))
        red_emb_list[index]['y'] = reduced_dimensions[:, 1]
        red_emb_list[index]['y'] = red_emb_list[index]['y'].map(lambda x: round(x, 2))
        red_emb_list[index]['label'] = labels

        # Create a color map
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colors = cm.rainbow(np.linspace(0, 1, num_labels))

        # Create a dictionary mapping labels to colors
        label_color_map = {label: color for label, color in zip(unique_labels, colors)}

        # plot scatters
        scatters = []
        for k,d in red_emb_list[index].groupby('label'):
            scatters.append( axs[index].scatter(d.x, d.y, color=label_color_map[k], label=k, alpha=0.8) )

        # configure axes and titles
        axs[index].legend()
        axs[index].set_xticks([])
        axs[index].set_yticks([])
        axs[index].set_xlabel(f'{technique.upper()}1')
        axs[index].set_ylabel(f'{technique.upper()}2')

        title = f'Scatter plot of {type} embeddings\n'
        if subject:
            title += f'Run on {subject}'
        if best_loss:
            title += f' with loss {best_loss}'
        if prediction:
            title += f' (Prediction {prediction}; Target {target})'
        fig.suptitle(title)

        if len(layer_names) > 0:
            axs[index].set_title(layer_names[index])
        else:
            axs[index].set_title(f'Layer {index}')

        print(f'Succesfully ran {title}')


    # saving the figure
    dr_path = os.path.join(directory, f'{type}_embedding')
    fig.savefig(dr_path, dpi=200)

    return red_emb_list

def dimensionality_reduction(embedding, technique, parameters=None, seed=1):
    """
        Dimensionally reduce embeddings
        :param embedding: list, contains data points with high dimensional features"
        :param technique: string, the dimensionality reduction technique to use. Valid entries: tsne, umap, pca
        :param parameters: object with parameters 'perplexity' and 'n_neighbours', parameters for dimensionality reduction techniques.
        :param seed: int, randomization seed for the stochastic processes in dimensionality reduction.
        :return: reduced_dimensions, a list of datapoints with dimensions reduced to 2
    """

    # scale the embeddings
    scaled_embedding = StandardScaler().fit_transform(embedding)

    # set default parameters
    if parameters == None:
        parameters = {'perplexity': 5, 'n_neighbors': 15}

    # transform the embeddings based on technique
    if technique == 'tsne':
        reduced_dimensions = TSNE(n_components=2, learning_rate='auto', perplexity=parameters['perplexity'],
                                  random_state=seed).fit_transform(scaled_embedding)
    elif technique == 'umap':
        reducer = umap.UMAP(n_neighbors=parameters['n_neighbors'], random_state=seed)
        reduced_dimensions = reducer.fit_transform(scaled_embedding)

    elif technique == 'pca':
        reducer = PCA(n_components=2, random_state=seed)
        reduced_dimensions = reducer.fit_transform(scaled_embedding)

    else:
        print("Error in selecting dimensionality reduction technique")
        return False

    return reduced_dimensions



if __name__ == "__main__":
    # main variable declaration
    save_directory = "/Users/richardzhang/Desktop/jh101/embedding_visuals" # where to save visualizations
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    emb_directory = "/Users/richardzhang/Desktop/jh101/nf-energy_ef-phase_adj-coh/fold-0/embeddings.pickle" # where to load embeddings

    subject = "pt01" # specify which subject to get channel name labels for
    technique = "tsne" # options: tsne, umap, pca

    # load embedding object
    emb_file = open(emb_directory, 'rb')
    emb_obj = pickle.load(emb_file)
    emb_file.close()

    # get embeddings, and convert to numpy array
    # embedding shape: # batches * # layers of embeddings * batch size * # features in an embedding
    raw_embedding = emb_obj['embeddings']
    num_batches = len(raw_embedding)
    embedding = np.empty((num_batches, len(raw_embedding[0])), dtype=object)
    for batch_idx in range(num_batches):
        array_of_layers = np.empty(len(raw_embedding[batch_idx]), dtype=object)
        for layer_idx in range(len(raw_embedding[0])):
            embedding[batch_idx][layer_idx] = raw_embedding[batch_idx][layer_idx]

    # get targets
    # ground truth shape: #batches * batch size
    ground_truth = emb_obj['ground_truth']

    # ------------------------------ Visualize graph embeddings ------------------------------

    # from all batches, extract only the layers corresponding to graph embeddings
    graph_embeddings = embedding[:, 3:5]
    graph_embeddings = np.swapaxes(graph_embeddings, 0, 1)

    # select only 1 batch from each layer - NOTE: batch is 2nd axis in graph_embeddings
    single_graph_embedding = [graph_embeddings[0][0], graph_embeddings[1][0]]
    # create labels corresponding to the batch
    targets = ground_truth[0]
    labels = ['Ictal' if i==1 else 'Nonictal' for i in targets]

    # visualize embedding
    emb_visualize(save_directory, 'graph', single_graph_embedding, labels, technique=technique,
                    layer_names=['Flatten', 'Fully connected'])

    # ------------------------------ Visualize node embeddings ------------------------------

    # from one batch, extract only the layers corresponding to node embeddings
    node_embeddings = embedding[0, 1:3]

    # select only 1 iteration from a batch - NOTE: iteration is 2nd axis in node_embeddings
    single_node_embedding = [node_embeddings[0][0], node_embeddings[1][0]]

    # get channel names for the subject
    _, _, _, ch_names = get_gres(subject, openneuro_graph_representations_dir, 1)

    # remove numbers from channel names
    labels = [''.join([char for char in string if char.isalpha()]) for string in ch_names]

    # visualize embedding
    emb_visualize(save_directory, 'node', single_node_embedding, labels, technique=technique,
                   layer_names=['ECC', 'GAT'])