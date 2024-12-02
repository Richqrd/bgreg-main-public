import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from bgreg.data.analysis.graph.graph_representation import load_data
from bgreg.data.model.graph_neural_network.ecc_gat import Net as Net
from bgreg.data.model.graph_neural_network.data import get_data_loaders

def get_model_info(sub="ummc001", node_feat="ones", edge_feat="ones", adj="corr", normalize=True,
        link_cutoff=0.3, classify="binary", combined_features=True,
        comb_node_feat=None, comb_edge_feat=None, seed=0,
        self_loops=False, batch_size=32, val_size=0.1, test_size=0.1,
        fltrs_out=84, l2_reg=1e-3, dropout_rate=0.5):

    # get graph representations
    data, ch_names, preictal_size, ictal_size, postictal_size = load_data(sub, node_f=node_feat, edge_f=edge_feat,
                                                                          adj_type=adj, normalize=normalize,
                                                                          link_cutoff=link_cutoff,
                                                                          classification=classify,
                                                                          combined_features=combined_features,
                                                                          comb_node_feat=comb_node_feat,
                                                                          comb_edge_feat=comb_edge_feat,
                                                                          self_loops=self_loops)

    # get model and compile
    model = Net(fltrs_out, l2_reg, dropout_rate, classify)

    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size, val_size, test_size, seed)

    i = 0
    for b in train_loader:
        if i == 1:
            inputs, targets = b
        i = i+1

    return model, inputs

def get_flops(model, inputs):
    # Convert the model to a TensorFlow function
    full_model = tf.function(lambda x: model(x))


    dummy_input1 = tf.random.normal([1, inputs[0][0].shape[0], inputs[0][0].shape[1]])
    dummy_input2 = tf.random.normal([1, inputs[1][0].shape[0], inputs[1][0].shape[1]])
    dummy_input3 = tf.random.normal([1, inputs[2][0].shape[0], inputs[2][0].shape[1], inputs[2][0].shape[2]])

    full_model = full_model.get_concrete_function(
    [tf.TensorSpec(dummy_input1.shape, dummy_input1.dtype),
     tf.TensorSpec(dummy_input2.shape, dummy_input2.dtype),
     tf.TensorSpec(dummy_input3.shape, dummy_input3.dtype)]
    )

    # Get the frozen concrete function
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # Calculate FLOPs using the profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # Create the profiler
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    total_flops = flops.total_float_ops

    return total_flops

if __name__ == "__main__":

    subject = 'ummc003'

    model, inputs = get_model_info(sub=subject, combined_features=False, node_feat="ones", edge_feat="ones",
        comb_edge_feat=['corr', 'coh'], comb_node_feat=['energy', 'band_energy'], adj="coh",
        classify="binary")

    flops = get_flops(model, inputs)

    print(model.summary())
    print(f'Total FLOPs: {flops}')
