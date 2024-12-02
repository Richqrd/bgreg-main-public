import networkx as nx
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from spektral.layers import MessagePassing
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import dot
from spektral.utils.sparse import sp_matrix_to_sp_tensor

import matplotlib.pyplot as plt

import os

class GNNExplainer:
    """
    The GNNExplainer model from the paper:

    > [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)<br>
    > Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik and Jure Leskovec.

    The model can be used to explain the predictions for a single node or for an entire
    graph. In both cases, it returns the subgraph that mostly contributes to the
    prediction.

    **Arguments**

    - `model`: tf.keras.Model to explain;
    - `n_hops`: number of hops from which the GNN aggregates info. If `None`, then the
    number is inferred from the Conv and MessagePassing layers in the model.
    - `preprocess`: a preprocessing function to transform the adjacency matrix before
    giving it as input to the GNN; this is usually the same `preprocess` function of the
    Conv or MessagePassing layers used in the GNN (e.g., `GCNConv.preprocess`).
    - `graph_level`: if True, the GNN is assumed to be for graph-level prediction and
    the explanation is computed for the whole graph (and not just a node).
    - `verbose`: if True, print info during training;
    - `learning_rate`: learning rate when training the model;
    - `a_size_coef`: coefficient to control the number of edges of the subgraph that
    contributes to the prediction;
    - `x_size_coef`: coefficient to control the number of features of the subgraph
    that contributes to the prediction;
    - `a_entropy_coef`: coefficient to control the discretization of the adjacency
    mask;
    - `x_entropy_coef`: coefficient to control the discretization of the features
    mask;
    - `laplacian_coef`: coefficient to control the graph Laplacian loss;
    """

    def __init__(
        self,
        model,
        n_hops=None,
        preprocess=None,
        graph_level=False,
        verbose=False,
        learning_rate=0.01,
        a_size_coef=0.0005,
        x_size_coef=0.1,
        a_entropy_coef=0.1,
        x_entropy_coef=0.1,
        laplacian_coef=0.0,
    ):
        self.model = model

        # Automatically detect the number of hops from which the GNN aggregates info
        if n_hops is None:
            self.n_hops = 0
            for layer in model.layers:
                if isinstance(layer, (Conv, MessagePassing)):
                    self.n_hops += 1
            print(f"n_hops was automatically inferred to be {self.n_hops}")
        else:
            self.n_hops = n_hops

        self.preprocess = preprocess
        self.graph_level = graph_level
        self.verbose = verbose

        self.learning_rate = learning_rate
        self.a_size_coef = a_size_coef
        self.x_size_coef = x_size_coef
        self.a_entropy_coef = a_entropy_coef
        self.x_entropy_coef = x_entropy_coef
        self.laplacian_coef = laplacian_coef

    def explain_node(self, x, a, e, node_idx=None, epochs=100):
        """
        Train the GNNExplainer to explain the given graph.

        :param x: feature matrix of shape `(n_nodes, n_node_features)`;
        :param a: sparse adjacency matrix of shape `(n_nodes, n_nodes)`;
        :param node_idx: index of the node to explain. If `self.graph_level=True`, this
        is ignored;
        :param epochs: number of epochs to train for.
        :return:
        - `a_mask`: mask for the adjacency matrix;
        - `x_mask`: mask for the node features.
        """
        x = tf.cast(x, tf.float32)
        if node_idx is None:
            node_idx = 0

        # Get the computational graph
        if self.graph_level:
            self.comp_graph = tf.cast(a, tf.float32)
            self.comp_graph = tf.sparse.from_dense(self.comp_graph)

            self.i = tf.zeros(x.shape[0], dtype=tf.int32)
            # model input parameters are different FIXME
            self.y_pred = tf.argmax(self.model([a, x, e], training=True), axis=1)
        else:
            self.comp_graph = k_hop_sparse_subgraph(
                a, node_idx, self.n_hops, self.preprocess
            )
            self.y_pred = tf.argmax(self.model([x, a], training=False), axis=1)

        self.node_pred = self.y_pred[node_idx]
        self.y_pred = tf.cast(self.y_pred, tf.float32)

        # NOTE
        # y_pred = [0] tensor
        # node_pred = 0 tensor

        # Optimizer for training
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)

        # Init the trainable masks
        # FIXME: assuming x mask is the mask for features
        x_mask = tf.Variable(
            tf.random.normal((1, 1, x.shape[2]), stddev=0.1),
            dtype=tf.float32,
            trainable=True,
        )
        # FIXME: assuming a mask is the mask for graph - applied to the sparse comp_graph
        # Assuming that this is a sparse mask
        a_mask = tf.Variable(
            tf.random.normal(
                self.comp_graph.values.shape, stddev=(2 / x.shape[0]) ** 0.5
            ),
            dtype=tf.float32,
            trainable=True,
        )


        # Training loop
        for i in range(epochs):
            losses = self._train_step(x, a_mask, x_mask, node_idx, e=e)
            if self.verbose:
                print(f"epoch {i}: "+", ".join([f"{key}: {val}" for key, val in losses.items()]))

        return a_mask, x_mask

    @tf.function
    def _train_step(self, x, a_mask, x_mask, node_idx, e=None):


        with tf.GradientTape() as tape:
            # not comp_graph is a sparse matrix, which is masked by a_mask
            masked_a = tf.sparse.map_values(
                tf.multiply, self.comp_graph, tf.nn.sigmoid(a_mask)
            )
            masked_x = x * tf.nn.sigmoid(x_mask)

            # print('x')
            # print(x)
            # print('masks')
            # print(x_mask)
            # print(a_mask)

            # print('a_mask')
            # print(masked_a)
            # print(masked_x)
            # print(e)
            #
            # print('sparse to dense')
            # n = tf.sparse.to_dense(masked_a)
            # print(n)

            if self.graph_level:
                # convert masked_a to dense matrix - all operations to produce masked_a from comp_garph were done on sparse matrices
                # Need to convert to dense matrix for the GNN pipeline
                dense_masked_a = tf.sparse.to_dense(masked_a)
                pred = self.model([dense_masked_a,masked_x, e], training=True)[
                    0, self.node_pred
                ]

            else:
                pred = self.model([masked_x, masked_a], training=False)[
                    node_idx, self.node_pred
                ]


            loss, losses = self._explain_loss_fn(pred, a_mask, x_mask)
        grad = tape.gradient(loss, [a_mask, x_mask])
        self.opt.apply_gradients(zip(grad, [a_mask, x_mask]))
        return losses

    def _explain_loss_fn(self, y_pred, a_mask, x_mask):
        mask = tf.nn.sigmoid(a_mask)

        # Prediction loss
        pred_loss = -tf.math.log(y_pred + 1e-15)

        # Loss for A
        a_size_loss = self.a_size_coef * tf.reduce_sum(mask)
        entropy = -mask * tf.math.log(mask + 1e-15) - (1 - mask) * tf.math.log(
            1 - mask + 1e-15
        )
        a_entropy_loss = self.a_entropy_coef * tf.reduce_mean(entropy)

        # Graph Laplacian loss
        if self.graph_level:
            smoothness_loss = 0
        else:
            masked_a = tf.sparse.map_values(tf.multiply, self.comp_graph, mask)
            d = tf.linalg.diag(tf.sparse.reduce_sum(masked_a, axis=0))
            masked_a = tf.sparse.map_values(tf.multiply, masked_a, -1)

            laplacian = tf.sparse.add(d, masked_a)
            laplacian = tf.cast(laplacian, tf.float32)
            quad_form = (
                tf.reshape(self.y_pred, (1, -1))
                @ laplacian
                @ tf.reshape(self.y_pred, (-1, 1))
            )
            smoothness_loss = self.laplacian_coef * quad_form

        # Feature loss
        mask = tf.nn.sigmoid(x_mask)
        x_size_loss = self.x_size_coef * tf.reduce_sum(mask)
        entropy = -mask * tf.math.log(mask + 1e-15) - (1 - mask) * tf.math.log(
            1 - mask + 1e-15
        )
        x_entropy_loss = self.x_entropy_coef * tf.reduce_mean(entropy)

        loss = (
            pred_loss
            + a_size_loss
            + a_entropy_loss
            + smoothness_loss
            + x_size_loss
            + x_entropy_loss
        )

        losses = {
            "pred_loss": pred_loss,
            "a_size_loss": a_size_loss,
            "a_entropy_loss": a_entropy_loss,
            "smoothness_loss": smoothness_loss,
            "x_size_loss": x_size_loss,
            "x_entropy_loss": x_entropy_loss,
        }
        return loss, losses

    def _explainer_cleaning(self, a_mask, x_mask, node_idx, a_thresh):
        # Get the masks
        selected_adj_mask = tf.nn.sigmoid(a_mask)
        selected_feat_mask = tf.nn.sigmoid(x_mask)

        # convert into a binary matrix
        if self.preprocess is not None:
            comp_graph_values = tf.ones_like(self.comp_graph.values)
            self.comp_graph = tf.sparse.SparseTensor(
                self.comp_graph.indices, comp_graph_values, self.comp_graph.shape
            )

        # remove the edges which value is < a_thresh
        selected_adj_mask = tf.where(
            selected_adj_mask >= a_thresh, selected_adj_mask, 0
        )

        selected_subgraph = tf.sparse.map_values(
            tf.multiply, self.comp_graph, selected_adj_mask
        )

        is_nonzero = tf.not_equal(selected_subgraph.values, 0)
        selected_subgraph = tf.sparse.retain(selected_subgraph, is_nonzero)


        # impose the symmetry of the adj matrix
        # FIXME: need to add perm because first dimension corresponds to the batch dimension, so don't transpose along that dimension -
        selected_subgraph = (
            tf.sparse.add(selected_subgraph, tf.sparse.transpose(selected_subgraph, perm=[0,2,1])) / 2
        )


        if not self.graph_level:
            # get the final denoised subgraph centerd in the interested node
            selected_subgraph = k_hop_sparse_subgraph(
                selected_subgraph, node_idx, self.n_hops
            )

        # the top_feat relevant feature ids
        selected_features = tf.argsort(
            tf.nn.sigmoid(selected_feat_mask), direction="DESCENDING"
        )[0]
        # returns [[index of top feature]], intended to return [index of top feature]; we have extra batch dimension
        selected_features = selected_features[0] #fixes the above

        # selected subgraph is of shape 1 x node x node, sparsetensor
        # selected features is of shape 1 x 1 x features, tf.Variable

        # fixes the batch dimension for the selected subgraph
        selected_subgraph = tf.sparse.reshape(selected_subgraph, [selected_subgraph.dense_shape[1], selected_subgraph.dense_shape[2]])



        return selected_subgraph, selected_features

    def plot_subgraph(
        self, a_mask, x_mask, logdir, node_idx=None, a_thresh=0.1, return_features=False, showGraph=False, soz_nodes = None, prediction=None, groundtruth=None, sample=0, explain_all_data=False, x_mask_processed=False
    ):
        """
        Plot the subgraph computed by the GNNExplainer.

        **Arguments**
        :param a_mask: the mask for the adjacency matrix computed by `explain_node`;
        :param x_mask: the mask for the node features computed by `explain_node`;
        :param node_idx: the same node index that was given to `explain_node`;
        :param a_thresh: threshold to remove low-importance edges;
        :param return_features: if True, return indices to sort the nodes by their
        importance.
        :return: The subgraph computed by GNNExplainer in Networkx format. If
        `return_features=True`, also returns an indices to sort the nodes by their
        importance.
        """
        adj_mtx, top_ftrs = self._explainer_cleaning(a_mask, x_mask, node_idx, a_thresh)

        edge_list = adj_mtx.indices.numpy()
        weights = adj_mtx.values

        G = nx.Graph()
        fig, ax = plt.subplots()
        for i, (n1, n2) in enumerate(edge_list):
            if weights[i] != 0:
                G.add_edge(n1, n2, w=weights[i].numpy())
                if i % 20 == 19:
                    print(f'Adding edge {i+1} / {edge_list.shape[0]} to graph')

        node_colors = None
        if soz_nodes:
            node_colors = ['firebrick' if node in soz_nodes else '#0697A7' for node in list(G)]
            edges = ['black' for node in list(G)]


        # take the largest component commented out
        # giant = max(nx.algorithms.components.connected_components(G), key=len)

        pos = nx.layout.spring_layout(G, k=None) #k is 0.04
        nx.draw_networkx_nodes(G, pos=pos, node_size=100, nodelist=None, ax=ax, node_color=node_colors, edgecolors=edges) #node list is giant
        nx.draw_networkx_edges(G, pos=pos, edge_color="grey", alpha=0.8, ax=ax)
        # nx.draw_networkx_labels(
        #     G, pos=pos, font_color="black", font_size=5, verticalalignment="bottom", ax=ax
        # )

        ax.set_title(f'Graph of adjacency matrix containing top {edge_list.shape[0]} edges and top {len(list(G))} nodes.\nThreshold for edge importance: {a_thresh}. p={round(prediction)} g={round(groundtruth)}',
                     fontdict={'fontsize': 12})
        if x_mask_processed is not False:

            feature_txt = f'--Node Features--\n'

            feature_labels = ['overall', '0.5-4hz', '4-7hz', '8-12hz', '13-30hz', '30-70hz', '70-100hz', '100-250hz', '250-500hz']

            # only custom for pt3 new graph reps
            # feature_labels = ['overall', '1-4Hz', '4-8Hz', '8-13Hz', '13-20Hz', '20-30Hz', '30-40Hz', '40-50Hz', '50-60Hz', '60-70Hz', '70-80Hz', '80-90Hz', '90-100Hz', '100-130Hz', '130-160Hz', '160-190Hz', '190-220Hz', '220-250Hz', '250-300Hz', '300-350Hz', '350-400Hz', '400-450Hz', '450-500Hz']


            for ft_idx in range(len(x_mask_processed)):
                feature_txt += f'{feature_labels[ft_idx]}: {round(float(x_mask_processed[ft_idx]), 2)}\n'

            ax.text(0.01, 0.4, feature_txt, transform=plt.gcf().transFigure, fontsize=6)

        classification = f'p{round(prediction)}_gt{round(groundtruth)}'
        if not os.path.exists(os.path.join(logdir, 'GNNExplainer')):
            os.mkdir(os.path.join(logdir, 'GNNExplainer'))
        if not os.path.exists(os.path.join(logdir, 'GNNExplainer', classification)):
            os.mkdir(os.path.join(logdir, 'GNNExplainer', classification))


        if explain_all_data:
            title = f'explainer_adj_graph_{sample}.png'
            if not os.path.exists(os.path.join(logdir, 'GNNExplainer', 'all_data')):
                os.mkdir(os.path.join(logdir, 'GNNExplainer', 'all_data'))
            fig.savefig(os.path.join(logdir, 'GNNExplainer', 'all_data', title), dpi=200)
        else:
            title = f'explainer_adj_graph_{sample}.png'
            fig.savefig(os.path.join(logdir, 'GNNExplainer', classification, title), dpi=200)

        if showGraph:
            plt.show()

        if return_features:
            return G, top_ftrs
        else:
            return G, [node_colors.count('red'), len(G)]


def k_hop_sparse_subgraph(a, node_idx, k, transformer=None):
    """
    Computes the subgraph containing all the neighbors of `node_idx` up to the k-th order.
    If `a` is not the binary adjacency matrix a  `transformer` should be passed.
    **Arguments**
    - `a`: sparse `(n_nodes, n_nodes)` graph tensor;
    - `node_idx`: center node;
    - `k`: order of neighbor;
    - `transformer`: one of the functions from the `spektral.transforms` module,
       needed to convert the binary adjacency matrix into the correct format for the model;
    """
    if a.dtype != tf.float32:
        a = tf.cast(a, tf.float32)

    if transformer:
        a = binary_adj_converter(a)

    power_a = tf.sparse.eye(a.shape[0])
    k_neighs = np.zeros(a.shape[0]).astype("float32").reshape(1, -1)
    k_neighs[0, node_idx] = 1

    for _ in range(k - 1):
        power_a = dot(power_a, a)
        temp = tf.sparse.slice(power_a, start=[node_idx, 0], size=[1, power_a.shape[0]])
        k_neighs += tf.sparse.to_dense(temp)

    comp_graph = tf.sparse.add(a * tf.reshape(k_neighs, (-1, 1)), a * k_neighs)
    is_nonzero = tf.not_equal(comp_graph.values, 0)
    comp_graph = tf.sparse.retain(comp_graph, is_nonzero)
    comp_graph = tf.sign(comp_graph)

    if transformer:
        comp_graph = sp_tensor_to_sp_matrix(comp_graph)
        comp_graph = transformer(comp_graph)
        return sp_matrix_to_sp_tensor(comp_graph)
    else:
        return comp_graph


def binary_adj_converter(a_in):
    """
    Transforms a graph matrix into the binary adjacency matrix.
    **Arguments**
    - `a_in`: sparse `(n_nodes, n_nodes)` graph tensor;
    """
    a_idx = a_in.indices
    off_diag_idx = tf.not_equal(a_idx[:, 0], a_idx[:, 1])
    a_idx = a_idx[off_diag_idx]

    a = tf.sparse.SparseTensor(
        a_idx, tf.ones(a_idx.shape[0], dtype=tf.float32), a_in.shape
    )
    return a


def sp_tensor_to_sp_matrix(a):
    """
    Transforms a sparse tensor into a sparse scipy matrix .
    **Arguments**
    - `a`: sparse `(n_nodes, n_nodes)` graph tensor;
    """
    a_idx = a.indices
    a_val = a.values

    row_idx = a_idx[:, 0]
    col_idx = a_idx[:, 1]

    return csr_matrix((a_val, (row_idx, col_idx)), shape=a.shape)



