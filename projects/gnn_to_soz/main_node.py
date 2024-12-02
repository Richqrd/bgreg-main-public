import os
import pickle
from pathlib import Path

import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.utils import set_random_seed

import tensorflow as tf

from bgreg.native.datapaths import datarecord_path
from bgreg.data.analysis.graph.graph_representation import sequential_load_data, load_data_multi_subject
from bgreg.data.model.graph_neural_network.ecc_gat_node import Net as Node_Net
from bgreg.data.model.graph_neural_network.data import get_sequential_data_loader

from projects.gnn_to_soz.evaluation import training_curves, attention_plot, f1_score, eval_get_attn, to_categorical, importance_plot
import time

import json
import numpy as np
import math

from gnnexplainer import GNNExplainer

from embedding_visualize import node_embedding

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering

from torch.utils.data import DataLoader, Subset

def create_log_directories(sub, node_feat, edge_feat, adj,
                           link_cutoff, classify, combined_features,
                           comb_node_feat, comb_edge_feat, logdir, logreg=False):
    """
    Create log directories for under projects/gnn_to_soz/logs/{binary|multi}/{subject}
    :param sub: string, subject ID, i.e., "ummc001_1s"
    :param node_feat: string, node features. Valid entries: ones, energy, band_energy
    :param edge_feat: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classify: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param logdir: string, directory path for log files.
    :return:
    """
    logsdir = Path(datarecord_path, "logs")
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    logsdir = Path(logsdir, classify)
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    logsdir = Path(logsdir, sub)
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    if logdir is None:
        if not combined_features:
            logdir = Path(logsdir,
                          "nf-" + node_feat +
                          "_ef-" + edge_feat +
                          "_adj-" + adj +
                          "_link-" + str(link_cutoff))
        elif not logreg:
            logdir = Path(logsdir,
                          "cnf-" + "_".join(comb_node_feat) +
                          "_cef-" + "_".join(comb_edge_feat) +
                          "_adj-" + adj +
                          "_link-" + str(link_cutoff))
        else:
            logdir = Path(logsdir,
                          "cnf-" + "_".join(comb_node_feat) +
                          "_logreg")

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    return logdir


def run(sub="ummc001", node_feat="ones", edge_feat="ones", adj="corr", normalize=True,
        link_cutoff=0.3, classify="binary", combined_features=True,
        comb_node_feat=None, comb_edge_feat=None, seed=1,
        self_loops=False, logdir=None, batch_size=32, val_size=0.1, test_size=0.1,
        fltrs_out=84, l2_reg=1e-3, dropout_rate=0.5, lr=0.001, epochs=2, es_patience=20, logreg=False,
        training=True, explain_all_data=False, gnn_explainer=False):
    """
    Create and evaluate a GNN model for seizure detection from graph representations of iEEG data.


    :param sub: string, subject ID, i.e., "ummc001"
    :param node_feat: string, node features. Valid entries: ones, energy, band_energy
    :param edge_feat: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classify: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param seed: int, seed for reproducibility
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :param logdir: string, directory path for log files.
    :param batch_size: int, batch size for data split.
    :param val_size: float, percentage of data for validation set.
    :param test_size: float, percentage of data for test set.
    :param fltrs_out: int, filters out in model.
    :param l2_reg: float, l2 regularization.
    :param dropout_rate: float, dropout rate.
    :param lr: float, learning rate.
    :param epochs: int, number of epochs.
    :param es_patience: int, patience threshold.
    :return:
    """

    set_random_seed(seed)

    # FIXME: Fixed configuration
    if combined_features and comb_node_feat is None:
        comb_node_feat = ["energy", "band_energy"]
    if combined_features and comb_edge_feat is None:
        comb_edge_feat = ["coh", "phase"]

    if training:
        print(f'Testing on {sub}')
    else:
        print(f'Running (not testing) on {sub}')

    # create the directories for the logs
    logdir = create_log_directories(sub, node_feat, edge_feat, adj,
                                    link_cutoff, classify, combined_features,
                                    comb_node_feat, comb_edge_feat, logdir, logreg=logreg)

    # get graph representations

    # data, ch_names, preictal_size, ictal_size, postictal_size = load_node_data(sub, node_f=node_feat, edge_f=edge_feat,
    #                                                                       adj_type=adj, normalize=normalize,
    #                                                                       link_cutoff=link_cutoff,
    #                                                                       classification="node",
    #                                                                       combined_features=combined_features,
    #                                                                       comb_node_feat=comb_node_feat,
    #                                                                       comb_edge_feat=comb_edge_feat,
    #                                                                       self_loops=self_loops)

    s = ['ummc002', 'ummc003', 'ummc004', 'ummc005', 'ummc006', 'ummc009', 'pt01', 'pt2', 'pt3', 'pt8', 'pt13', 'pt16', 'umf001']
    # s = ['ummc002', 'ummc003', 'ummc004']

    data, preictal_size, ictal_size, postictal_size, test_idx = load_data_multi_subject(s, node_f=node_feat, edge_f=edge_feat,
                                                                               adj_type=adj, normalize=normalize,
                                                                               link_cutoff=link_cutoff,
                                                                               classification="node",
                                                                               combined_features=combined_features,
                                                                               comb_node_feat=comb_node_feat,
                                                                               comb_edge_feat=comb_edge_feat,
                                                                               self_loops=self_loops,
                                                                              test_sub=[sub])


    # split in train, validation, and test loads
    data_indices = [i for i in range(len(data)) if i not in test_idx]
    np.random.shuffle(data_indices)
    val_idx = data_indices[0: int(len(data_indices) * val_size)]
    train_idx = data_indices[int(len(data_indices) * val_size):]
    np.random.shuffle(test_idx)

    print(f'Total data size: {len(data)}')
    print(f'Train-val data size: {len(data_indices)}')
    print(f'Test data size (leave one out): {len(test_idx)}')

    train_sampler = Subset(data, train_idx)
    val_sampler = Subset(data, val_idx)
    test_sampler = Subset(data, test_idx)
    train_loader = DataLoader(train_sampler, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_sampler, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_sampler, batch_size=batch_size, shuffle=False)


    # get model and compile
    model = Node_Net(fltrs_out, l2_reg, dropout_rate)
    optimizer = Adam(lr)

    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC", f1_score, TruePositives(name="true_positives"), TrueNegatives(name="true_negatives"),
                           FalsePositives(name="false_positives"), FalseNegatives(name="false_negatives") ])

    # ------------------------------ training ------------------------------
    best_val_loss = 10000
    patience = es_patience
    train_loss = []
    train_acc = []
    train_auc = []
    train_f1_score = []
    val_loss = []
    val_acc = []
    val_auc = []
    val_f1_score = []
    computation_times = []
    saved_model = False  # flag for saving model into disk
    epochs = epochs if training else 1

    train_confusion_matrices = []
    val_confusion_matrices = []

    start_time_train = time.perf_counter()
    for epoch in range(epochs):
        # train
        metrics = None
        for b in train_loader:
            inputs, targets = b
            if classify == "multi":
                targets = to_categorical(targets, num_classes=3)
            outs = model.train_on_batch(inputs, targets)
            for i, k in enumerate(model.metrics_names):
                if metrics is None:
                    metrics = {k: 0 for k in model.metrics_names}
                metrics[k] += outs[i] / len(train_loader)
        train_loss.append(metrics["loss"])
        train_acc.append(metrics["accuracy"])
        train_auc.append(metrics["auc"])
        train_f1_score.append(metrics["f1_score"])

        confusion_matrix = {
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
            "tn": metrics["true_negatives"],
        }
        train_confusion_matrices.append(confusion_matrix)

        # ------------------------------ validating ------------------------------
        metrics_val = None
        for b_val in val_loader:
            inputs, targets = b_val
            inputs = [inp.numpy() for inp in inputs]
            if classify == "multi":
                targets = to_categorical(targets, num_classes=3)
            else:
                targets = targets.numpy()

            outs = model.evaluate(inputs, targets)
            for i, k in enumerate(model.metrics_names):
                if metrics_val is None:
                    metrics_val = {k: 0 for k in model.metrics_names}
                metrics_val[k] += outs[i] / len(val_loader)
        val_loss.append(metrics_val["loss"])
        val_acc.append(metrics_val["accuracy"])
        val_auc.append(metrics_val["auc"])
        val_f1_score.append(metrics_val["f1_score"])

        confusion_matrix = {
            "tp": metrics_val["true_positives"],
            "fp": metrics_val["false_positives"],
            "fn": metrics_val["false_negatives"],
            "tn": metrics_val["true_negatives"],
        }
        val_confusion_matrices.append(confusion_matrix)

        epoch_time = time.perf_counter() - start_time_train
        computation_times.append(epoch_time)


        # ------------------------------ print stats for epoch ------------------------------
        desc = "Epoch {:d}".format(epoch)
        for k in model.metrics_names:
            desc_metrics = " - {}: " + "{:.4f}"
            desc += desc_metrics.format(k, metrics[k])
        for k in model.metrics_names:
            desc_metrics = " - val_{}: " + "{:.4f}"
            desc += desc_metrics.format(k, metrics_val[k])
        print(desc)

        # ------------------------------ early stopping ------------------------------
        if metrics_val["loss"] < best_val_loss:
            best_val_loss = metrics_val["loss"]
            patience = es_patience
            if training:
                model.save_weights(os.path.join(logdir, "best_model.h5"))
            saved_model = True
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
    if not saved_model:
        if training:
            model.save_weights(os.path.join(logdir, "best_model.h5"))
    # ------------------------------ log best model ------------------------------
    model.load_weights(os.path.join(logdir, "best_model.h5"))

    # ------------------------------ log statistical measurements ------------------------------
    if training:
        train_val_stats = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'train_f1_score': train_f1_score,
            'train_confusion': train_confusion_matrices,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1_score': val_f1_score,
            'val_confusion': val_confusion_matrices,
            'computation_time': computation_times
        }
        pickle.dump(train_val_stats, open(os.path.join(logdir, "train_val_stats.pickle"), "wb"))
        # ------------------------------ evaluating ------------------------------
        # plot and save figures for training stats
        training_curves(train_loss, val_loss, train_acc, val_acc, logdir)

    # evaluate and get attention FIXME
    test_stats, attns, embeddings = eval_get_attn(model, test_loader, classify)
    pickle.dump(attns, open(os.path.join(logdir, "attention_scores.pickle"), "wb"))
    pickle.dump(embeddings, open(os.path.join(logdir, "embeddings.pickle"), "wb"))
    pickle.dump(test_stats, open(os.path.join(logdir, "test_stats.pickle"), "wb"))

    # FIXME temporarily disabled
    # attention_plot(attns, ch_names, sub, logdir)

    print(model.summary())



    # gnn explainer
    if gnn_explainer:

        if explain_all_data:

            sequential_data = sequential_load_data(sub, node_f=node_feat, edge_f=edge_feat,
                                                   adj_type=adj, normalize=normalize,
                                                   link_cutoff=link_cutoff,
                                                   classification=classify,
                                                   combined_features=combined_features,
                                                   comb_node_feat=comb_node_feat,
                                                   comb_edge_feat=comb_edge_feat,
                                                   self_loops=self_loops)

            all_loader = get_sequential_data_loader(sequential_data)

            i = 0
            for b in all_loader:
                if i == 0:
                    inputs, targets = b
                else:
                    print('If you see this there is an error')
                i = i + 1

        else:
            i = 0
            for b in test_loader:
                if i == 1:
                    inputs, targets = b
                i = i + 1

        # Print ictal zones
        ictal_indices = [i for i, j in enumerate(targets) if j == 1.0]
        print('Ictal Sample Indices')
        print(ictal_indices)


        SOZ_freq = pd.DataFrame(columns=['subject', 'seen_SOZ_nodes', 'total_SOZ_nodes', 'seen_nodes', 'total_nodes', 'pred', 'truth'])

        adj_masks = []
        x_masks = []

        nonictal_adj_masks = []
        nonictal_x_masks = []
        ictal_adj_masks = []
        ictal_x_masks = []
        # for sample in range(0,len(inputs[0]),100):
        for sample in range(ictal_indices[0]-25, ictal_indices[0]+25, 1):
        # for sample in range(375,385,1):
            print(f'Running {sample} / {len(inputs[0])} GNNExplainer for subject {subject}')


            adj = inputs[0][sample]
            x = inputs[1][sample]
            e = inputs[2][sample]

            adj = np.expand_dims(adj, axis=0)
            x = np.expand_dims(x, axis=0)
            e = np.expand_dims(e, axis=0)

            prediction = model([adj, x, e], training=True)[0, 0]

            explainer = GNNExplainer(model, graph_level=False, verbose=False)
            original_a_mask, original_x_mask = explainer.explain_node(x=x, a=adj, e=e, epochs=700)

            # process masks
            a_mask = tf.nn.sigmoid(original_a_mask)
            x_mask = tf.nn.sigmoid(original_x_mask)


            num_nodes = int(math.sqrt(a_mask.shape[0]))
            reshaped_a_mask = tf.reshape(a_mask, [num_nodes, num_nodes])
            a_mask_mean = tf.reduce_mean(reshaped_a_mask, 0)

            # print('a_mask_mean')
            # print(a_mask_mean)

            adj_masks.append(a_mask_mean)
            x_masks.append(x_mask.numpy()[0][0])
            # print(x_masks)

            # separate nonictal masks
            if round(float(prediction)) == 0:
                nonictal_adj_masks.append(a_mask_mean)
                nonictal_x_masks.append(x_mask.numpy()[0][0])
            elif round(float(prediction)) == 1:
                ictal_adj_masks.append(a_mask_mean)
                ictal_x_masks.append(x_mask.numpy()[0][0])

            # print('Adj Mask')
            # print(reshaped_a_mask)
            # print('X Feature Mask')
            # print(x_mask)


            # print(f'Prediction: {prediction}')
            # print(f'Ground truth: {targets[sample]}')
            #
            # print('Adj Mask (after S(X)) Descriptive Statistics')
            # processed_adj_mask = tf.nn.sigmoid(a_mask).numpy()
            # print(f'Mean ± STD = {processed_adj_mask.mean()} ± {np.std(processed_adj_mask)}')
            #
            # with np.printoptions(threshold=np.inf):
            #     print('X Mask (after S(X)) Descriptive Statistics')
            #     processed_x_mask = tf.nn.sigmoid(x_mask).numpy()[0][0]
            #     print(f'Mean ± STD = {processed_x_mask.mean()} ± {np.std(processed_x_mask)}')
            #
            #     print('Full X Mask (after S(X))')
            #     print(processed_x_mask)

            # print(ch_names)

            with open(os.path.join("projects/gnn_to_soz", "soz_contacts.json"), 'r') as f:
                soz_contacts_json = json.load(f)
                soz_contacts = soz_contacts_json[subject]['contacts']
                # print(soz_contacts)

                # testing with spectral clustering
                cluster = SpectralClustering(n_clusters=5, random_state=0, affinity="precomputed").fit(reshaped_a_mask)
                # print(list(cluster.labels_))
                # print(ch_names)
                changed_ch_names = ['SOZ' if x in soz_contacts else '' for x in ch_names]
                show_graph_with_labels(reshaped_a_mask, cluster.labels_, changed_ch_names, sample, logdir, a_thresh=0.6)


                soz_indices = []
                for i, name in enumerate(ch_names):
                    if name in soz_contacts:
                        soz_indices.append(i)

                plt.close()
                plt.close()

                print(x_mask.numpy()[0][0])
                G, [SOZ_nodes, seen_nodes] = explainer.plot_subgraph(original_a_mask, original_x_mask, logdir, a_thresh=0.95,
                                                                     soz_nodes=soz_indices, showGraph=False,
                                                                     prediction=float(prediction),
                                                                     groundtruth=float(targets[sample]), sample=sample, explain_all_data=explain_all_data, x_mask_processed=x_mask.numpy()[0][0])

                # logging SOZ frequency
                SOZ_freq.loc[len(SOZ_freq.index)] = [subject, SOZ_nodes, len(soz_contacts), seen_nodes, len(ch_names),
                                                     round(float(prediction)), round(float(targets[sample]))]

        print(SOZ_freq)
        SOZ_freq.to_csv(os.path.join(logdir, 'GNNExplainer', 'soz_freqs.csv'))

        # final adj_mask processing
        adj_masks = [x.numpy() for x in adj_masks]
        adj_masks = np.array(adj_masks)
        x_masks = np.array(x_masks)

        nonictal_adj_masks = [x.numpy() for x in nonictal_adj_masks]
        nonictal_adj_masks = np.array(nonictal_adj_masks)
        nonictal_x_masks = np.array(nonictal_x_masks)

        ictal_adj_masks = [x.numpy() for x in ictal_adj_masks]
        ictal_adj_masks = np.array(ictal_adj_masks)
        ictal_x_masks = np.array(ictal_x_masks)

        # save importance FIXME
        pickle.dump(adj_masks, open(os.path.join(logdir, "imptnce_scores.pickle"), "wb"))
        pickle.dump(x_masks, open(os.path.join(logdir, "xfeature_scores.pickle"), "wb"))

        pickle.dump(nonictal_adj_masks, open(os.path.join(logdir, "nonictal_imptnce_scores.pickle"), "wb"))
        pickle.dump(nonictal_x_masks, open(os.path.join(logdir, "nonictal_xfeature_scores.pickle"), "wb"))

        pickle.dump(ictal_adj_masks, open(os.path.join(logdir, "ictal_imptnce_scores.pickle"), "wb"))
        pickle.dump(ictal_x_masks, open(os.path.join(logdir, "ictal_xfeature_scores.pickle"), "wb"))


        # plot figure for attention ranking FIXME
        importance_plot(adj_masks, ch_names, sub, logdir)
        importance_plot(nonictal_adj_masks, ch_names, sub, logdir, type="nonictal")
        importance_plot(ictal_adj_masks, ch_names, sub, logdir, type="ictal")

def show_graph_with_labels(adjacency_matrix, mylabels, nodes, sample, logdir, a_thresh=0.6):

    fig, ax = plt.subplots(figsize=(12,8))

    color_dict = {0: 'red', 1: 'pink', 2: 'orange', 3: 'aqua', 4: 'teal'}
    node_color_list = []
    label_dict = {}
    rows, cols = np.where(adjacency_matrix > a_thresh)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjacency_matrix.shape[0])
    node_num = 0
    for n in all_rows:
        gr.add_node(n)
        node_color_list.append(color_dict[mylabels[node_num]])
        # print(color_dict[mylabels[node_num]])
        label_dict[n] = nodes[node_num]
        node_num += 1
    gr.add_edges_from(edges)
    pos = nx.spring_layout(gr)
    nx.draw(gr, node_size=50, node_color=node_color_list, with_labels=False, ax=ax, pos=pos)
    nx.draw_networkx_labels(gr, pos=pos, labels=label_dict, font_size=8, verticalalignment="bottom")
    # plt.show()

    if not os.path.exists(os.path.join(logdir, 'GNNExplainer')):
        os.mkdir(os.path.join(logdir, 'GNNExplainer'))
    if not os.path.exists(os.path.join(logdir, 'GNNExplainer', 'spectral')):
        os.mkdir(os.path.join(logdir, 'GNNExplainer', 'spectral'))

    title = f'spectral_adj_graph_{sample}.png'
    fig.savefig(os.path.join(logdir, 'GNNExplainer', 'spectral', title), dpi=200)

if __name__ == "__main__":

    subjects = ['ummc002', 'ummc003', 'ummc004', 'ummc005', 'ummc006', 'ummc009', 'pt01', 'pt2', 'pt3', 'pt8', 'pt13', 'pt16', 'umf001']


    for subject in subjects:
        epochs = 320

        run(sub=subject, epochs=epochs, combined_features=True, node_feat="energy", edge_feat="coh",
            comb_edge_feat=['corr', 'coh'], comb_node_feat=['energy', 'band_energy'],
            logreg=False, training=True, adj="coh", explain_all_data=False, gnn_explainer=False,
            classify="node")

