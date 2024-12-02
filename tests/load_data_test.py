"""
    This script is to test the utils.dataset.load_data(), utils.dataset.get_data_loaders(), and
    utils.visual.datasplit_class_balanc.class_balance() methods.
"""
from bgreg.data.analysis.graph.graph_representation import load_data, load_data_temp
from bgreg.data.model.graph_neural_network.data import get_data_loaders
from bgreg.utils.visual.datasplit_class_balance import class_balance

import pickle

if __name__ == "__main__":
    # Fixed configurations, one would mainly change the 'sub' variable
    sub = "jh101"
    node_feat = "ones"
    edge_feat = "ones"
    adj = "corr"
    normalize = True
    link_cutoff = 0.3
    classify = "binary"
    combined_features = False
    comb_node_feat = None
    comb_edge_feat = None
    seed = 0
    self_loops = False
    batch_size = 32
    val_size = 0.1
    test_size = 0.1

    # Fixed configuration of combined features; check main for updates.
    if combined_features and comb_node_feat is None:
        comb_node_feat = ["energy", "band_energy"]
    if combined_features and comb_edge_feat is None:
        comb_edge_feat = ["coh", "phase"]

    # test load_data()
    data, ch_names, preictal_size, ictal_size, postictal_size = load_data(sub, node_f=node_feat, edge_f=edge_feat,
                                                                          adj_type=adj, normalize=normalize,
                                                                          link_cutoff=link_cutoff,
                                                                          classification=classify,
                                                                          combined_features=combined_features,
                                                                          comb_node_feat=comb_node_feat,
                                                                          comb_edge_feat=comb_edge_feat,
                                                                          self_loops=self_loops
                                                                          )
    pickle.dump(data, open("jh101_grs.pickle", "wb"))
    # test get_data_loaders()
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size, val_size, test_size, seed)
    train_indices = train_loader.sampler.__getattribute__("indices")
    val_indices = val_loader.sampler.__getattribute__("indices")
    test_indices = test_loader.sampler.__getattribute__("indices")

    # test class_balance()
    # Note: this becomes very slow for data size > 1000
    class_balance(preictal_size, ictal_size, postictal_size, train_indices, val_indices, plot=False, savefig=False)
