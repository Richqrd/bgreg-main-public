import os
import pickle
from pathlib import Path
from numpy import mean, std, random, asarray, ones, expand_dims, sum, \
    linalg, array, nan_to_num, isnan, ones_like, floor
from multiprocessing import Process, Pool

from bgreg.native.datapaths import graph_representation_elements_dir, preprocessed_data_dir
from bgreg.data.io import get_runs, get_run_from_filename
from bgreg.data.preprocess import get_preprocessed_data
from bgreg.data.analysis.fcn.metrics import corr, calculate_coherence_signal_trace, coh, \
    phase, energy, band_energy, butter_bandpass_filter

import numpy as np
from scipy import io

# save preprocessed data for patient sub with all combinations

def graph_representation_elements_ethz(sub, w_sz=None, w_st=0.125):
    """
    Compute and save graph representation elements
    Creates three subprocesses, one for each signal trace

    After getting the preprocessed signal traces (see projects.gnn_to_soz.preprocess), the function
    creates 3 subprocesses that call the function save_gres() for the preictal, ictal, and postictal traces.
    save_gres() will compute all co-activity measurements (currently correlation, coherence, and phase-lock value),
    to then create the graph, node and edge features. These metrics are then stored in pickle files at the directory
    graph_representation_elements_dir/{patientID} (see native.datapaths).

    Preprocessed signal traces are stored in the preprocessed_data_dir directory declared in native.datapaths.
    This function will save gres for every subject run. For instance, if there are 2 subject runs in
    datarecord_path/{subject}, then there will be 2 gres for every signal trace.

    :param sub: string, patient ID (i.e., pt2).
    :param w_sz: float, window size to analyze signal
    :param w_st: float, percentage of window size to be used as
                window step to analyze signal. 0 < w_st <= 1
    :return: nothing, it saves the signal traces into pickle files.
    """
    dirpath = '/Users/richardzhang/Desktop/GNN Research/bgreg-main/data_files/swec-ethz-ieeg-testing'
    subpath = Path(dirpath, sub)
    sf = 512
    for filepath in subpath.iterdir():
        if filepath.is_file() and filepath.suffix == '.mat':
            # obtain precomputed signals for the patient
            preictal_trace, ictal_trace, postictal_trace = read_ethz(filepath)

            run = str(filepath.name).split('.')[0]

            # default to sampling frequency
            window_size = w_sz

            # default to sampling frequency, floor
            window_step = int(floor(window_size * w_st))

            # create directory to store data
            data_dir = Path('/Users/richardzhang/Desktop/GNN Research/bgreg-main/data_files/ds003029-processed/graph_representation_elements_ethz', sub)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            p_save_preictal_trace = Process(target=save_gres,
                                            args=(preictal_trace, run, sf, window_size,
                                                  window_step, data_dir, "preictal"))
            p_save_preictal_trace.daemon = False
            p_save_ictal_trace = Process(target=save_gres,
                                         args=(ictal_trace, run, sf, window_size,
                                               window_step, data_dir, "ictal"))
            p_save_ictal_trace.daemon = False
            p_save_postictal_trace = Process(target=save_gres,
                                             args=(postictal_trace, run, sf, window_size,
                                                   window_step, data_dir, "postictal"))
            p_save_postictal_trace.daemon = False

            p_save_preictal_trace.start()
            p_save_ictal_trace.start()
            p_save_postictal_trace.start()

            p_save_preictal_trace.join()
            p_save_ictal_trace.join()
            p_save_postictal_trace.join()


def read_ethz(filename):
    eeg = io.loadmat(filename)['EEG']
    sz_on = 92160
    sz_off = eeg.shape[0] - 92160

    preictal = eeg[:sz_on, :]
    ictal = eeg[sz_on:sz_off, :]
    postictal = eeg[sz_off:, :]

    return preictal.T, ictal.T, postictal.T



def save_gres(signal_trace, run, sfreq, window_size, window_step, data_dir, trace_type):
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
    # pool = Pool(3) TODO: changed
    pool = Pool()
    """
    # Regular processes - Adj matrix computed from the same window_size as features
    processes = [pool.apply_async(get_all, args=(signal_trace[:, i:i + window_size], sfreq))
                 for i in range(0, last_step, window_step)]
    """


    # Custom processes - Adj matrix computed from adj_window_size while features are computed from window_size
    # TODO: Fix logic to input signal_trace start point and end point as a parameter rather than the entire signal_trace variable
    adj_window_size = 1000 * 20
    processes = [pool.apply_async(custom_adj_get_all, args=(signal_trace[:, i:i + window_size],
                                                            # [int(i - min(max(i - adj_window_size / 2, 0), last_step - adj_window_size)), int(i + window_size - min(max(i - adj_window_size / 2, 0), last_step - adj_window_size))],
                                                              signal_trace[:, int (min(max(i - adj_window_size / 2, 0), last_step - adj_window_size)): int( min( max(adj_window_size, i + adj_window_size / 2), last_step ))],
                                                              sfreq, i, last_step))
                 for i in range(0, last_step, window_step)]



    result = [p.get() for p in processes]
    pool.close()
    pool.join()
    file_name = trace_type + "_" + str(run) + ".pickle"
    data_file = Path(data_dir, file_name)
    with open(data_file, 'wb') as save_file:
        print('dumping some file')
        pickle.dump(result, save_file)


def save_adj_whole_signal(sub, run):
    """
    TODO:
     AD, Apr-4, This is a temporary function to save adjacency matrices representing the
     functional connectivity network of the WHOLE signal. Needs to be removed after running tests.
     AD, Apr-10, Not removing it yet as the obtained results were not bad.
    """
    file_name = "pp_signal_" + str(run) + ".pickle"
    pp_signal = pickle.load(open(Path(preprocessed_data_dir, sub, file_name), "rb"))
    adjs = get_gr_adj_whole(pp_signal.get_data(), pp_signal.info["sfreq"])

    data_dir = Path(graph_representation_elements_dir, sub)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    file_name = "signal_" + str(run) + ".pickle"
    data_file = Path(data_dir, file_name)
    with open(data_file, 'wb') as save_file:
        pickle.dump(adjs, save_file)


def get_fc_graphs(x, fc_measure, nf_mode, adj_type):
    # edge features
    if fc_measure == "ones":
        ef = ones((x.shape[0], x.shape[0], 1), dtype=x.dtype)
    elif fc_measure == "corr":
        ef = corr(x)
        ef = expand_dims(ef, -1)
    elif fc_measure == "coh":
        ef = coh(x)
    elif fc_measure == "phase":
        ef = phase(x)
    else:
        raise ValueError("Invalid fc_measure")
    ef = array(ef)
    if any(isnan(ef)):
        nan_to_num(ef, copy=False)

    # node features
    if nf_mode == "energy":
        nf = energy(x)
    elif nf_mode == "band_energy":
        nf = band_energy(x)
    elif nf_mode == "ones":
        nf = ones(x.shape[0], dtype=x.dtype)
        nf = expand_dims(nf, -1)
    else:
        raise ValueError("Invalid nf_mode")
    nf = array(nf)

    # adjacency matrix
    if adj_type == fc_measure:
        adj = mean(ef, axis=-1)
    elif adj_type == "ones":
        adj = ones_like(ef)
        adj = mean(adj, axis=-1)
    elif adj_type == "corr":
        adj_ef = corr(x)
        adj_ef = expand_dims(adj_ef, -1)
        adj = mean(adj_ef, axis=-1)
    elif adj_type == "coh":
        adj_ef = coh(x)
        adj = mean(adj_ef, axis=-1)
    elif adj_type == "phase":
        adj_ef = phase(x)
        adj = mean(adj_ef, axis=-1)
    else:
        raise ValueError("Invalid adj_type")

    return adj, nf, ef

def get_all(x, sfreq):
    """
    Get all combinations of node and edge features, and adjacency matrices
    :param x: ndarray, EEG signal trace
    :param sfreq: float, sampling frequency of signal
    :return: ndarrays of adj_matrices, node_features, edge_features
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
    coherences = mean(coherences, axis=0)
    edge_features.append(expand_dims(coherences, -1))

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

    # ---------- adjacency matrix ----------
    adj_matrices = [ones((x.shape[0], x.shape[0]), dtype=x.dtype),  # "ones"
                    corr_x,  # "corr"
                    coherences,  # "coh"
                    phases]  # "phase"

    return adj_matrices, node_features, edge_features

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
    # node_features, edge_features = get_ones(x_features, sfreq)

    return adj_matrices, node_features, edge_features


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

# Function to compute node and edge features, but only use ones
def get_ones(x, sfreq):
    # ---------- edge features ----------
    edge_features = [ones((x.shape[0], x.shape[0], 1), dtype=x.dtype)]

    # ---------- node features ----------
    node_features = [ones((x.shape[0], 1), dtype=x.dtype)]

    return node_features, edge_features




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

    # "coh" FIXME default
    # coherence_result = calculate_coherence_signal_trace(x, sfreq)
    # coherences_dict = coherence_result['coherence']
    # coherences = []
    # for key in coherences_dict.keys():
    #     coherences.append(coherences_dict[key])
    # coherences = mean(coherences, axis=0)
    # edge_features.append(expand_dims(coherences, -1))

    # "coh" custom, with extra features for each band FIXME
    coherence_result = calculate_coherence_signal_trace(x, sfreq)
    coherences_dict = coherence_result['coherence']
    coherences = []
    for key in coherences_dict.keys():
        coherences.append(coherences_dict[key])
    coherences.insert(0,mean(coherences, axis=0))
    combined_coherences = [
        [[sublists[i][j] for sublists in coherences] for j in range(len(coherences[0][0]))]
        for i in range(len(coherences[0]))
    ]
    combined_coherences = np.array(combined_coherences)
    # edge_features.append(expand_dims(coherences, -1))
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

    # ---------- adjacency matrix ----------
    adj_matrices = [ones((x.shape[0], x.shape[0]), dtype=x.dtype),  # "ones"
                    corr_x,  # "corr"
                    coherences,  # "coh"
                    phases]  # "phase"

    return node_features, edge_features


def get_gr_adj_whole(signal_trace, sfreq, adjs=None):
    """
    TODO: This function is used for a very specific test; to be removed
    """
    if adjs is None:
        adjs = []
    adjs.append(corr(signal_trace))

    # "coh"
    coherence_result = calculate_coherence_signal_trace(signal_trace, sfreq)
    coherences_dict = coherence_result['coherence']
    coherences = []
    for key in coherences_dict.keys():
        coherences.append(coherences_dict[key])
    adjs.append(mean(coherences, axis=0))

    # "phase"
    phases_dict = coherence_result['phase_dict']
    phases = []
    for key in phases_dict.keys():
        phases.append(phases_dict[key])
    adjs.append(mean(phases, axis=0))

    return adjs


def get_adj_whole(x, adj_type, run, normalize=True):
    adjs = {"corr": 0, "coh": 1, "phase": 2}
    for adj_obj in x:
        if adj_obj[1] == run:
            adj = adj_obj[0][adjs[adj_type]]
            if normalize and (adj_type != "ones"):
                adj = (adj - mean(adj)) / std(adj)  # normalization
            return adj
    print("get_adj_whole: it didn't enter for", adj_type, run)
    return -1


def get_adj(x, i, adj_type, normalize, link_cutoff, self_loops):
    """
    Get the adjacency matrix corresponding to a location in time 'i' from the EEG signals
    :param x: list of lists, graph abstraction where to read from
    :param i: int, location in the signal sequence
    :param adj_type: string, adjacency matrix type
    :param normalize: bool, operator to normalize the data
    :param link_cutoff: TODO: thresholding with link_cutoff must follow an algorithm TBD
    :param self_loops: currently disabled, enabling fill-out of the diagonal by the coactivity algorithms
    :return: lists with adjacency matrix information
    """
    adjs = {"ones": 0, "corr": 1, "coh": 2, "phase": 3}
    adj = x[i][0][adjs[adj_type]]
    if normalize and (adj_type != "ones"):
        adj = (adj - mean(adj)) / std(adj)  # normalization
    # adj[abs(adj) < link_cutoff] = 0
    # if self_loops:
    #    fill_diagonal(adj, 1.0)
    # else:
    #    fill_diagonal(adj, 0.0)
    return adj


def get_nf(x, i, normalize, combined_features, nf_mode=None, comb_node_feat=None):
    """
    Get the node features corresponding to a location in time 'i' from the EEG signals
    :param x: list of lists, graph abstraction where to read from
    :param i: int, location in the signal sequence
    :param normalize: bool, operator to normalize the data
    :param combined_features: FIXME: Combined features are currently assumed to be a list of strict size 2
    :param nf_mode: string, node feature type
    :param comb_node_feat: FIXME: fixed configuration of combinations for current research purposes
    :return: lists with node features information
    """
    nfs = {"ones": 0, "energy": 1, "band_energy": 2}

    # FIXME: added temporarily for ones
    # combined_features = False
    if combined_features:
        # print(comb_node_feat[0])
        if len(comb_node_feat) == 2:
            nf = get_combined_nf(x[i][1][nfs[comb_node_feat[0]]],
                                 x[i][1][nfs[comb_node_feat[1]]])
        else:
            nf = x[i][1][nfs[comb_node_feat[0]]]
    else:
        nf = x[i][1][nfs[nf_mode]]
    if normalize and (combined_features or (nf_mode != "ones")):
        nf = (nf - mean(nf)) / std(nf)  # normalization
    return nf


def get_ef(x, i, normalize, combined_features, fc_measure=None, comb_edge_feat=None):
    """
    Get the edge features corresponding to a location in time 'i' from the EEG signals
    :param x: list of lists, graph abstraction where to read from
    :param i: int, location in the signal sequence
    :param normalize: bool, operator to normalize the data
    :param combined_features: FIXME: Combined features are currently assumed to be a list of strict size 2
    :param fc_measure: string, functional connectivity masurement
    :param comb_edge_feat: FIXME: fixed configuration of combinations for current research purposes
    :return: lists with node features information
    """
    efs = {"ones": 0, "corr": 1, "coh": 2, "phase": 3}

    # FIXME: added temporarily for ones
    # combined_features = False
    if combined_features:
        # print(comb_edge_feat[0])
        # print(comb_edge_feat[1])
        # FIXME: Combined features are currently assumed to be a list of strict size 2
        ef = get_combined_ef(x[i][2][efs[comb_edge_feat[0]]],
                             x[i][2][efs[comb_edge_feat[1]]])
    else:
        ef = x[i][2][efs[fc_measure]]
    if normalize and (combined_features or (fc_measure != "ones")):
        ef = (ef - mean(ef)) / std(ef)  # normalization
    return ef


def get_combined_nf(tnf1, tnf2):
    """
    Concatenate 2 arrays of node features, tnf1 and tnf2
    :param tnf1: array of node features 1
    :param tnf2: array of node features 2
    :return: NumPy array with concatenated node features
    """
    nff = []
    for i in range(tnf1.shape[0]):
        nff.append(asarray(tnf1[i].tolist() + tnf2[i].tolist()))
    return asarray(nff)


def get_combined_ef(tef1, tef2):
    """
    Concatenate 2 arrays of edge features, tef1 and tef2
    :param tef1: array of edge features 1
    :param tef2: array of edge features 2
    :return: NumPy array with concatenated edge features
    """
    eff = []
    for i in range(tef1.shape[0]):
        eff.append([])
        for j in range(tef1.shape[1]):
            eff[i].append(asarray(tef1[i][j].tolist() + tef2[i][j].tolist()))
    return asarray(eff)


def process_gres(gre, data_dir, file):
    """
    Initialize GRE containaer
    :param gre: list with GRE elements
    :param data_dir: string, directory with GRE for all runs
    :param file: filename to process
    :return: initialized or concatenated GRE
    """
    if len(gre) == 0:
        gre = pickle.load(open(Path(data_dir, file), 'rb'))
    else:
        gre += pickle.load(open(Path(data_dir, file), 'rb'))
    return gre


def get_gres(sub):
    """
    Concatenate GREs from all subject runs
    :param sub: string, subject ID
    :return: 4 lists, gre_preictal, gre_ictal, gre_postictal, ch_names
    """
    gre_preictal = []
    gre_ictal = []
    gre_postictal = []

    data_dir = Path(graph_representation_elements_dir, sub)
    for file in os.listdir(data_dir):
        if "postictal" in file:
            gre_postictal = process_gres(gre_postictal, data_dir, file)
        elif "preictal" in file:
            gre_preictal = process_gres(gre_preictal, data_dir, file)
        elif "ictal" in file:
            gre_ictal = process_gres(gre_ictal, data_dir, file)

    ch_names = pickle.load(open(Path(data_dir, "ch_names.pickle"), 'rb'))
    return gre_preictal, gre_ictal, gre_postictal, ch_names

def sequential_get_gres(sub):
    runs = []
    # runs x ictal status, data

    data_dir = Path(graph_representation_elements_dir, sub)

    for i in range(9):
        exists = False
        for file in os.listdir(data_dir):
            if str(i) in file:
                exists = True
                if "postictal" in file:
                    gre_postictal = process_gres([], data_dir, file)
                elif "preictal" in file:
                    gre_preictal = process_gres([], data_dir, file)
                elif "ictal" in file:
                    gre_ictal = process_gres([], data_dir, file)
        if exists:
            runs.append([gre_preictal, gre_ictal, gre_postictal])

    ch_names = pickle.load(open(Path(data_dir, "ch_names.pickle"), 'rb'))
    return runs, ch_names




    for file in os.listdir(data_dir):
        for i in range(7):
            if str(i) in file:

                if len(runs) <= i:
                    run = process_gres([], data_dir, file)
                    runs.append




def process_gres_whole(gre, data_dir, file):
    """
    Initialize GRE containaer for gres using only ONE adjacency matrix
    :param gre: list with GRE elements
    :param data_dir: string, directory with GRE for all runs
    :param file: filename to process
    :return: tuple, (concatenated GRE, run)
    """
    run = get_run_from_filename(file)
    # these contain the 3 adjacency matrices (for the signal)
    gre_arr = pickle.load(open(Path(data_dir, file), 'rb'))
    run_list = [run] * len(gre_arr)
    if len(gre) == 0:
        if "signal" in file:
            gre = [(gre_arr, run)]
        else:
            gre = [(i, j) for i, j in zip(gre_arr, run_list)]
    else:
        if "signal" in file:
            gre += [(gre_arr, run)]
        else:
            gre += [(i, j) for i, j in zip(gre_arr, run_list)]
    return gre


def get_gres_whole(sub):
    gre_preictal = []
    gre_ictal = []
    gre_postictal = []
    gre_signal = []

    data_dir = Path(graph_representation_elements_dir, sub)
    for file in os.listdir(data_dir):
        if "postictal" in file:
            gre_postictal = process_gres_whole(gre_postictal, data_dir, file)
        elif "preictal" in file:
            gre_preictal = process_gres_whole(gre_preictal, data_dir, file)
        elif "ictal" in file:
            gre_ictal = process_gres_whole(gre_ictal, data_dir, file)
        elif "signal" in file:
            gre_signal = process_gres_whole(gre_signal, data_dir, file)

    ch_names = pickle.load(open(Path(data_dir, "ch_names.pickle"), 'rb'))
    return gre_preictal, gre_ictal, gre_postictal, gre_signal, ch_names


def load_data(sub, node_f="ones", edge_f="corr", adj_type="ones", normalize=True,
              link_cutoff=0.1, classification="binary", combined_features=False,
              comb_node_feat=None, comb_edge_feat=None, self_loops=False):
    """
    Load preprocessed graph representations data from directory 'data/ds-003029-processed/preprocessed_graph'
    for any subject 'sub'
    This function creates the graph representations with their labeled values.
    For instance, for the seizure detection formulated as a binary problem, graph representations of
    signal traces corresponding to the preictal signal trace have a label of 0, and those from the ictal
    signal traces have a label of 1.
    For the multi-class classification problem of distinguishing between preictal, ictal, and postictal
    signal traces, graph representations are labeled with 0, 1, and 2, respectively.
    :param sub: string, subject ID, i.e., "ummc001"
    :param node_f: string, node features. Valid entries: ones, energy, band_energy
    :param edge_f: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj_type: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classification: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :return: array with sequence of graph representations, and array of EEG channel names
    """
    # obtain precomputed graphs for the patient
    gre_preictal, gre_ictal, gre_postictal, ch_names = get_gres(sub)

    # FIXME: cleaning gre_ictal from channel names inserted into it at save_gres,
    #  bug has been fixed above but needs to be pushed.
    #  This for/if-clause needs to be removed after development.
    for i in gre_ictal:
        if isinstance(i, str):
            gre_ictal.remove(i)

    # use all ictal samples
    ictal_samples = len(gre_ictal)
    # split equally nonictal samples taken from preictal and postictal data
    nonictal_samples = int(ictal_samples / 2)

    # shuffle gres
    # FIXME: randomization commented out to get sequential signal data.
    random.shuffle(gre_preictal)
    random.shuffle(gre_ictal)
    random.shuffle(gre_postictal)

    graph_representations = []
    if classification == "binary":
        for i in range(0, min(len(gre_preictal), nonictal_samples)):
            adj = get_adj(gre_preictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_preictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_preictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
            Y = 0.0
            graph_representations.append([[adj, nf, ef], Y])
        for i in range(0, ictal_samples):
            adj = get_adj(gre_ictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_ictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_ictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
            Y = 1.0
            graph_representations.append([[adj, nf, ef], Y])
        for i in range(0, min(len(gre_postictal), nonictal_samples)):
            adj = get_adj(gre_postictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_postictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_postictal, i, normalize, combined_features, fc_measure=edge_f,
                        comb_edge_feat=comb_edge_feat)
            Y = 0.0
            graph_representations.append([[adj, nf, ef], Y])
    elif classification == "multi":
        for i in range(0, min(len(gre_preictal), nonictal_samples)):
            adj = get_adj(gre_preictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_preictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_preictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
            Y = 0
            graph_representations.append([[adj, nf, ef], Y])
        for i in range(0, ictal_samples):
            adj = get_adj(gre_ictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_ictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_ictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
            Y = 1
            graph_representations.append([[adj, nf, ef], Y])
        for i in range(0, min(len(gre_postictal), nonictal_samples)):
            adj = get_adj(gre_postictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_postictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_postictal, i, normalize, combined_features, fc_measure=edge_f,
                        comb_edge_feat=comb_edge_feat)
            Y = 2
            graph_representations.append([[adj, nf, ef], Y])
    else:
        raise ValueError("Invalid classification type. Valid entries are binary or multi.")

    return graph_representations, ch_names, len(gre_preictal), len(gre_ictal), len(gre_postictal)

def load_node_data(sub, node_f="ones", edge_f="corr", adj_type="ones", normalize=True,
              link_cutoff=0.1, classification="binary", combined_features=False,
              comb_node_feat=None, comb_edge_feat=None, self_loops=False):
    """
    Load preprocessed graph representations data from directory 'data/ds-003029-processed/preprocessed_graph'
    for any subject 'sub'
    This function creates the graph representations with their labeled values.
    For instance, for the seizure detection formulated as a binary problem, graph representations of
    signal traces corresponding to the preictal signal trace have a label of 0, and those from the ictal
    signal traces have a label of 1.
    For the multi-class classification problem of distinguishing between preictal, ictal, and postictal
    signal traces, graph representations are labeled with 0, 1, and 2, respectively.
    :param sub: string, subject ID, i.e., "ummc001"
    :param node_f: string, node features. Valid entries: ones, energy, band_energy
    :param edge_f: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj_type: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classification: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :return: array with sequence of graph representations, and array of EEG channel names
    """
    # obtain precomputed graphs for the patient
    gre_preictal, gre_ictal, gre_postictal, ch_names = get_gres(sub)

    # FIXME: cleaning gre_ictal from channel names inserted into it at save_gres,
    #  bug has been fixed above but needs to be pushed.
    #  This for/if-clause needs to be removed after development.
    for i in gre_ictal:
        if isinstance(i, str):
            gre_ictal.remove(i)

    # use all ictal samples
    ictal_samples = len(gre_ictal)
    # split equally nonictal samples taken from preictal and postictal data
    nonictal_samples = int(ictal_samples / 2)

    # shuffle gres
    # FIXME: randomization commented out to get sequential signal data.
    random.shuffle(gre_preictal)
    random.shuffle(gre_ictal)
    random.shuffle(gre_postictal)

    import json
    with open(os.path.join("projects/gnn_to_soz", "soz_contacts.json"), 'r') as f:
        soz_contacts_json = json.load(f)
        soz_contacts = soz_contacts_json[sub]['contacts']
        node_labels = [1 if x in soz_contacts else 0 for x in ch_names]

        # node_labels = []
        # fake_annotated = []
        # for x in ch_names:
        #     n = round(random.uniform(0,1))
        #     if n == 1:
        #         node_labels.append(1)
        #         fake_annotated.append(x)
        #     else:
        #         node_labels.append(0)
        print(node_labels)
        # print(fake_annotated)

    graph_representations = []
    if classification == "node":
        # for i in range(0, min(len(gre_preictal), nonictal_samples)):
            # adj = get_adj(gre_preictal, i, adj_type, normalize, link_cutoff, self_loops)
            # nf = get_nf(gre_preictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            # ef = get_ef(gre_preictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
            # Y = np.array(node_labels)
            # graph_representations.append([[adj, nf, ef], Y])

        for i in range(0, ictal_samples):
            adj = get_adj(gre_ictal, i, adj_type, normalize, link_cutoff, self_loops)
            nf = get_nf(gre_ictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
            ef = get_ef(gre_ictal, i, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)

            Y = np.array(node_labels)

            graph_representations.append([[adj, nf, ef], Y])

        # for i in range(0, min(len(gre_postictal), nonictal_samples)):
        #     adj = get_adj(gre_postictal, i, adj_type, normalize, link_cutoff, self_loops)
        #     nf = get_nf(gre_postictal, i, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
        #     ef = get_ef(gre_postictal, i, normalize, combined_features, fc_measure=edge_f,
        #                 comb_edge_feat=comb_edge_feat)
        #     Y = np.array(node_labels)
        #     graph_representations.append([[adj, nf, ef], Y])
    else:
        raise ValueError("Invalid classification type. Valid entries are node.")

    return graph_representations, ch_names, len(gre_preictal), len(gre_ictal), len(gre_postictal)

def sequential_load_node_data(sub, node_f="ones", edge_f="corr", adj_type="ones", normalize=True,
              link_cutoff=0.1, classification="binary", combined_features=False,
              comb_node_feat=None, comb_edge_feat=None, self_loops=False):
    """
    Load preprocessed graph representations data from directory 'data/ds-003029-processed/preprocessed_graph'
    for any subject 'sub'
    This function creates the graph representations with their labeled values.
    For instance, for the seizure detection formulated as a binary problem, graph representations of
    signal traces corresponding to the preictal signal trace have a label of 0, and those from the ictal
    signal traces have a label of 1.
    For the multi-class classification problem of distinguishing between preictal, ictal, and postictal
    signal traces, graph representations are labeled with 0, 1, and 2, respectively.
    :param sub: string, subject ID, i.e., "ummc001"
    :param node_f: string, node features. Valid entries: ones, energy, band_energy
    :param edge_f: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj_type: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classification: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :return: array with sequence of graph representations, and array of EEG channel names
    """
    # obtain precomputed graphs for the patient
    data, ch_names = sequential_get_gres(sub)

    print("NUMBER OF RUNS")
    print(len(data))

    # FIXME: cleaning gre_ictal from channel names inserted into it at save_gres,
    #  bug has been fixed above but needs to be pushed.
    #  This for/if-clause needs to be removed after development.
    # for i in gre_ictal:
    #     if isinstance(i, str):
    #         gre_ictal.remove(i)

    # shuffle gres
    # FIXME: randomization commented out to get sequential signal data.
    # random.shuffle(gre_preictal)
    # random.shuffle(gre_ictal)
    # random.shuffle(gre_postictal)

    import json
    with open(os.path.join("projects/gnn_to_soz", "soz_contacts.json"), 'r') as f:
        soz_contacts_json = json.load(f)
        soz_contacts = soz_contacts_json[sub]['contacts']
        node_labels = [1 if x in soz_contacts else 0 for x in ch_names]

    graph_representations = []
    if classification == "node":
        for run in data:
            preictal, ictal, postictal = run
            for i in range(0, len(preictal)):
                adj = get_adj(preictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(preictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(preictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = np.array(node_labels)
                graph_representations.append([[adj, nf, ef], Y])
            for i in range(0, len(ictal)):
                adj = get_adj(ictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(ictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(ictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = np.array(node_labels)
                graph_representations.append([[adj, nf, ef], Y])
            for i in range(0, len(postictal)):
                adj = get_adj(postictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(postictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(postictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = np.array(node_labels)
                graph_representations.append([[adj, nf, ef], Y])

    else:
        raise ValueError("Invalid classification type. Valid entries are binary for sequential data loading.")

    return graph_representations

def sequential_load_data(sub, node_f="ones", edge_f="corr", adj_type="ones", normalize=True,
              link_cutoff=0.1, classification="binary", combined_features=False,
              comb_node_feat=None, comb_edge_feat=None, self_loops=False):
    """
    Load preprocessed graph representations data from directory 'data/ds-003029-processed/preprocessed_graph'
    for any subject 'sub'
    This function creates the graph representations with their labeled values.
    For instance, for the seizure detection formulated as a binary problem, graph representations of
    signal traces corresponding to the preictal signal trace have a label of 0, and those from the ictal
    signal traces have a label of 1.
    For the multi-class classification problem of distinguishing between preictal, ictal, and postictal
    signal traces, graph representations are labeled with 0, 1, and 2, respectively.
    :param sub: string, subject ID, i.e., "ummc001"
    :param node_f: string, node features. Valid entries: ones, energy, band_energy
    :param edge_f: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj_type: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classification: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :return: array with sequence of graph representations, and array of EEG channel names
    """
    # obtain precomputed graphs for the patient
    data, ch_names = sequential_get_gres(sub)

    print("NUMBER OF RUNS")
    print(len(data))

    # FIXME: cleaning gre_ictal from channel names inserted into it at save_gres,
    #  bug has been fixed above but needs to be pushed.
    #  This for/if-clause needs to be removed after development.
    # for i in gre_ictal:
    #     if isinstance(i, str):
    #         gre_ictal.remove(i)

    # shuffle gres
    # FIXME: randomization commented out to get sequential signal data.
    # random.shuffle(gre_preictal)
    # random.shuffle(gre_ictal)
    # random.shuffle(gre_postictal)

    graph_representations = []
    if classification == "binary":
        for run in data:
            preictal, ictal, postictal = run
            for i in range(0, len(preictal)):
                adj = get_adj(preictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(preictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(preictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = 0.0
                graph_representations.append([[adj, nf, ef], Y])
            for i in range(0, len(ictal)):
                adj = get_adj(ictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(ictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(ictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = 1.0
                graph_representations.append([[adj, nf, ef], Y])
            for i in range(0, len(postictal)):
                adj = get_adj(postictal, i, adj_type, normalize, link_cutoff, self_loops)
                nf = get_nf(postictal, i, normalize, combined_features, nf_mode=node_f,
                            comb_node_feat=comb_node_feat)
                ef = get_ef(postictal, i, normalize, combined_features, fc_measure=edge_f,
                            comb_edge_feat=comb_edge_feat)
                Y = 0.0
                graph_representations.append([[adj, nf, ef], Y])

    else:
        raise ValueError("Invalid classification type. Valid entries are binary for sequential data loading.")

    return graph_representations


def load_data_temp(sub, node_f="ones", edge_f="corr", adj_type="ones", normalize=True,
                   link_cutoff=0.1, classification="binary", combined_features=False,
                   comb_node_feat=None, comb_edge_feat=None, self_loops=False):
    """
    AD, Apr-4: This function is created to create sequences of graph representations with the same
    adjacency matrix for ALL the graph representations. The adjacency matrix consists in the functional
    connectivity network of the WHOLE signal.
    It is copied from load_data()

    Load preprocessed graph representations data from directory 'data/ds-003029-processed/preprocessed_graph'
    for any subject 'sub'
    This function creates the graph representations with their labeled values.
    For instance, for the seizure detection formulated as a binary problem, graph representations of
    signal traces corresponding to the preictal signal trace have a label of 0, and those from the ictal
    signal traces have a label of 1.
    For the multi-class classification problem of distinguishing between preictal, ictal, and postictal
    signal traces, graph representations are labeled with 0, 1, and 2, respectively.
    :param sub: string, subject ID, i.e., "ummc001"
    :param node_f: string, node features. Valid entries: ones, energy, band_energy
    :param edge_f: string, edge features. Valid entries: ones, coh, corr, phase
    :param adj_type: string, adjacency matrix type. Valid entries:corr, coh, phase
    :param normalize: bool, operator to normalize data
    :param link_cutoff: float, threshold for link cutoff to create functional networks
    :param classification: string, classification type. Valid entries: binary, multi.
    :param combined_features: bool, operator to consider combined features.
                                If True, node_feat and edge_feat are overridden.
    :param comb_node_feat: object, combined node features.
                                Currently supporting lists, i.e., ["energy", "band_energy"]
    :param comb_edge_feat:object, combined edge features.
                                Currently supporting lists, i.e., ["coh", "phase"]
    :param self_loops: bool, operator consider self loops in adjacency matrix.
    :return: array with sequence of graph representations, and array of EEG channel names
    """
    # obtain precomputed graphs for the patient
    gre_preictal, gre_ictal, gre_postictal, gre_signal, ch_names = get_gres_whole(sub)

    # use all ictal samples
    ictal_samples = len(gre_ictal)
    # split equally nonictal samples taken from preictal and postictal data
    nonictal_samples = int(ictal_samples / 2)

    # shuffle gres
    random.shuffle(gre_preictal)
    random.shuffle(gre_ictal)
    random.shuffle(gre_postictal)

    graph_representations = []
    # the adj matrix will depend on the run
    for i in range(0, min(len(gre_preictal), nonictal_samples)):
        adj = get_adj_whole(gre_signal, adj_type, gre_preictal[i][1], normalize)
        nf = get_nf(gre_preictal[i], 0, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
        ef = get_ef(gre_preictal[i], 0, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
        Y = 0.0
        graph_representations.append([[adj, nf, ef], Y])
    for i in range(0, len(gre_ictal)):
        adj = get_adj_whole(gre_signal, adj_type, gre_ictal[i][1], normalize)
        nf = get_nf(gre_ictal[i], 0, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
        ef = get_ef(gre_ictal[i], 0, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
        Y = 1.0
        graph_representations.append([[adj, nf, ef], Y])
    for i in range(0, min(len(gre_postictal), nonictal_samples)):
        adj = get_adj_whole(gre_signal, adj_type, gre_postictal[i][1], normalize)
        nf = get_nf(gre_postictal[i], 0, normalize, combined_features, nf_mode=node_f, comb_node_feat=comb_node_feat)
        ef = get_ef(gre_postictal[i], 0, normalize, combined_features, fc_measure=edge_f, comb_edge_feat=comb_edge_feat)
        Y = 0.0
        graph_representations.append([[adj, nf, ef], Y])

    return graph_representations, ch_names, len(gre_preictal), len(gre_ictal), len(gre_postictal)
