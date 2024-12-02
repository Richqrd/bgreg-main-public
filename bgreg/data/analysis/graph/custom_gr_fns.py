# FIXME: EDITS FOR THE GRAPH_REPRESENTATION.PY FILE

def get_ef(x, i, normalize, combined_features, fc_measure=None, comb_edge_feat=None):
    """
    Get the edge features corresponding to a location in time 'i' from the EEG signals
    :param x: list of lists, graph abstraction where to read from
    :param i: int, location in the signal sequence
    :param normalize: bool, operator to normalize the data
    :param combined_features: FIXME: Combined features are currently assumed to be a list of strict size 2
    :param fc_measure: string, functional connectivity measurement
    :param comb_edge_feat: FIXME: fixed configuration of combinations for current research purposes
    :return: lists with node features information
    """
    efs = {"ones": 0, "corr": 1, "coh": 2, "coh_band_coh": 3, "phase": 4}

    if combined_features:
        # FIXME: Combined features are currently assumed to be a list of strict size 2
        ef = get_combined_ef(x[i][2][efs[comb_edge_feat[0]]],
                             x[i][2][efs[comb_edge_feat[1]]])
    else:
        ef = x[i][2][efs[fc_measure]]
    if normalize and (combined_features or (fc_measure != "ones")):
        ef = (ef - mean(ef)) / std(ef)  # normalization
    return ef


def save_gres(signal_trace, run, sfreq, window_size, window_step, data_dir, trace_type, adj_window_size=1000*20):
    """
    Create sequences of graph representations of the signal traces by considering
    windows of size 'window_size'

    :param signal_trace: ndarray, EEG signal trace
    :param run: int, run number
    :param sfreq: float, sampling frequency
    :param window_size: int, size of window
    :param window_step: int, step that window takes when iterating through signal trace
    :param data_dir: string, directory where to dump serialized (pickle) graph representations
    :param trace_type: string, preictal, ictal, postictal
    :return:
    """
    last_step = signal_trace.shape[1] - window_size
    # pool = Pool(3) FIXME: changed to be dynamic based on system cpus
    pool = Pool()
    """
    # Regular processes - Adj matrix computed from the same window_size as features
    processes = [pool.apply_async(get_all, args=(signal_trace[:, i:i + window_size], sfreq))
                 for i in range(0, last_step, window_step)]
    """


    # Custom processes - Adj matrix computed from adj_window_size while features are computed from window_size - this version maximizes data
    processes = [pool.apply_async(custom_adj_get_all, args=(signal_trace[:, i:i + window_size],
                                                              signal_trace[:, int (min(max(i - adj_window_size / 2, 0), last_step - adj_window_size)):
                                                                              int( min( max(adj_window_size, i + adj_window_size / 2), last_step ))],
                                                              sfreq, i, last_step))
                 for i in range(0, last_step, window_step)]

    # # Custom processes - Adj matrix computed from adj_window_size while features are computed from window_size - this version is the most consistent
    # adj_window_size = 1000 * 20
    # processes = [pool.apply_async(custom_adj_get_all, args=(signal_trace[:, i:i + window_size],
    #                                                         signal_trace[:, int(i - adj_window_size / 2):
    #                                                                         int(i + adj_window_size / 2)],
    #                                                         sfreq, i, last_step))
    #              for i in range(int(adj_window_size / 2), int(last_step - adj_window_size / 2), window_step)]



    result = [p.get() for p in processes]
    pool.close()
    pool.join()
    file_name = trace_type + "_" + str(run) + ".pickle"
    data_file = Path(data_dir, file_name)
    with open(data_file, 'wb') as save_file:
        print('dumping some file')
        pickle.dump(result, save_file)





def custom_adj_get_all(x_features, x_adj, sfreq, i, last_step):
    """
        Compute adjacency matrix (from the window x_adj), node and edge features (both from the window x_features)
        :param x_features: ndarray, EEG signal trace
        :param x_adj: ndarray, EEG signal trace
        :param sfreq: float, sampling frequency of signal
        :param i: float, step index for tracking progress
        :param last_step: float, last step for tracking progress
        :return: ndarrays of adj_matrices
    """

    print(f'On step {i} / {last_step}')

    # get adj_matrices
    adj_matrices = custom_adj_matrices(x_adj, sfreq)

    # get all other features
    node_features, edge_features = get_features(x_features, sfreq)

    return adj_matrices, node_features, edge_features


# Function to compute node and edge features
def get_features(x, sfreq):
    """
    Get all combinations of node and edge features, WITHOUT adjacency matrices
    :param x: ndarray, EEG signal trace
    :param sfreq: float, sampling frequency of signal
    :return: ndarrays of node_features, edge_features
    """
    # ---------- edge features ----------
    # "ones"
    edge_features = [ones((x.shape[0], x.shape[0], 1), dtype=x.dtype)]

    # "corr"
    corr_x = corr(x)
    edge_features.append(expand_dims(corr_x, -1))

    # "coh"
    coherence_result = calculate_coherence_signal_trace(x, sfreq)
    coherences_dict = coherence_result['coherence']
    coherences = []
    for key in coherences_dict.keys():
        coherences.append(coherences_dict[key])
    coherences_mean = mean(coherences, axis=0)
    edge_features.append(expand_dims(coherences_mean, -1))

    # "coh_band_coh" expanded with extra features for each band
    coherences.insert(0,mean(coherences, axis=0))
    combined_coherences = [
        [[sublists[i][j] for sublists in coherences] for j in range(len(coherences[0][0]))]
        for i in range(len(coherences[0]))
    ]
    combined_coherences = np.array(combined_coherences)
    edge_features.append(combined_coherences)

    # "phase"
    phases_dict = coherence_result['phase_dict']
    phases = []
    for key in phases_dict.keys():
        phases.append(phases_dict[key])
    phases = mean(phases, axis=0)
    edge_features.append(expand_dims(phases, -1))

    # ---------- node features ----------
    # "ones"
    node_features = [ones((x.shape[0], 1), dtype=x.dtype)]

    # "energy"
    nf = energy(x)
    node_features.append(nf)

    # "band_energy"
    freq_dicts = coherence_result['freq_dicts']
    freqs = coherence_result['freqs']
    nf = [[] for _i in range(x.shape[0])]
    for i in range(x.shape[0]):
        for band in freq_dicts.keys():
            lowcut = freqs[min(freq_dicts[band])]
            highcut = freqs[max(freq_dicts[band])]
            if lowcut == highcut:
                lowcut = freqs[min(freq_dicts[band])-1]
            # lowcut frequencies must be greater than 0, so currently set to 0.1
            if lowcut == 0:
                lowcut += 0.1
            # highcut frequencies must be less than sfreq / 2, so subtract 1 from max
            if highcut == sfreq / 2:
                highcut -= 0.1
            freq_sig = butter_bandpass_filter(x[i], lowcut, highcut, sfreq)
            nf[i].append(sum([j ** 2 for j in freq_sig]))
        nf[i] = nf[i]
    nf /= linalg.norm(nf, axis=0, keepdims=True)
    node_features.append(nf)

    return node_features, edge_features


def custom_adj_matrices(x, sfreq):
    """
        Compute adjacency matrix
        :param x: ndarray, EEG signal trace
        :param sfreq: float, sampling frequency of signal
        :return: ndarrays of adj_matrices
    """

    # "corr"
    corr_x = corr(x)

    # "coh"
    coherence_result = calculate_coherence_signal_trace(x, sfreq)
    coherences_dict = coherence_result['coherence']
    coherences = []
    for key in coherences_dict.keys():
        coherences.append(coherences_dict[key])
    coherences = mean(coherences, axis=0)

    # "phase"
    phases_dict = coherence_result['phase_dict']
    phases = []
    for key in phases_dict.keys():
        phases.append(phases_dict[key])
    phases = mean(phases, axis=0)

    adj_matrices = [ones((x.shape[0], x.shape[0]), dtype=x.dtype),  # "ones"
                    corr_x,  # "corr"
                    coherences,  # "coh"
                    phases]  # "phase"

    return adj_matrices


# FIXME: EDITS FOR EVALUATION.PY FILE
def eval_get_attn(model, test_loader, classify):
    metrics = None
    attns = []
    test_loss = []
    test_acc = []
    test_auc = []
    test_f1_score = []
    test_confusion_matrices = []
    embeddings = []
    ground_truth = []

    start_time_train = time.perf_counter()
    for b in test_loader:
        inputs, targets = b
        inputs = [inp.numpy() for inp in inputs]
        if classify == "multi":
            targets = to_categorical(targets, num_classes=3)
        else:
            targets = targets.numpy()
        # get evaluation stats
        outs = model.evaluate(inputs, targets, workers=3)
        for i, k in enumerate(model.metrics_names):
            if metrics is None:
                metrics = {k: 0 for k in model.metrics_names}
            metrics[k] += outs[i] / len(test_loader)
        test_loss.append(metrics["loss"])
        test_acc.append(metrics["accuracy"])
        test_auc.append(metrics["auc"])
        test_f1_score.append(metrics["f1_score"])

        confusion_matrix = {
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
            "tn": metrics["true_negatives"],
        }
        test_confusion_matrices.append(confusion_matrix)


        # get attention - attention is size (batch_size, num_channels, 1, num_channels)
        # squeeze and average across each column to get attention for each channel

        # FIXME attn temporary disable
        _, attn, emb = model.predict(inputs, workers=3, use_multiprocessing=True)
        attn = squeeze(attn, 2)
        attn = mean(attn, 1)
        attns.extend(attn)

        embeddings.append(emb)
        ground_truth.append(targets)


    test_stats = {
        'test_loss': test_loss[-1],
        'test_acc': test_acc[-1],
        'test_auc': test_auc[-1],
        'test_f1_score': test_f1_score[-1],
        'computation_time': computation_times[-1],
        'test_confusion': test_confusion_matrices[-1]
    }

    emb_object = {
        'embeddings': embeddings,
        'ground_truth': ground_truth
    }

    # stats on test set
    desc = "Evaluation"
    for k in model.metrics_names:
        desc_metrics = " - {}: " + ("{:.4f}" if k == "loss" else "{:.2f}")
        desc += desc_metrics.format(k, metrics[k])
    print(desc)

    return test_stats, attns, emb_object