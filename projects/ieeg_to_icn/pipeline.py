"""
    Pipeline:
        1) split the iEEG signal in pre-ictal, ictal, and postictal traces.
        2) for each signal trace, compute the coherence matrices and store them for later processing.
        3) identify ICNs with PCA/ICA and store them for later processing.
        4) analyze ICNs and their link to the epileptogenic zone.
"""
import numpy as np

from bgreg.native.datapaths import *

from bgreg.data.io import get_raw_signal_bids, get_regions, get_regions_dict_inv
from bgreg.data.analysis.fcn.extensions.icn import icnetworks
from bgreg.data.analysis.fcn.extensions.icn.icn import get_icn
from bgreg.data.analysis.fcn.metrics import calculate_coherence_full_signal as cc
from bgreg.data.analysis.fcn.extensions.icn.plotchord import plot_graphs

from bgreg.tools.epilepsy.preprocess import process_bids_epilepsy

from bgreg.utils.dirtree import save_matrices, get_matrices_file
from bgreg.utils.visual import plot_matrices


def run_pipeline(subject, savefile=False):
    """
    Execute sequential steps of pipeline for a given patient
    :param subject: string, patient id
    :param savefile: (boolean), operator to save coh matrices into files, default False
    :return:
    """
    raw_signal = get_raw_signal_bids(subject)
    pp_preictal, pp_ictal, pp_postictal = preprocess_data(raw_signal)
    compute_coherence_matrices(subject, pp_preictal, pp_ictal, pp_postictal, savefile=savefile)
    compute_intrinsic_coherence_networks(subject, savefile=savefile)


def preprocess_data(subject):
    """
    1) split the iEEG signal in pre-ictal and ictal traces; though we also use the original whole signal.
    :return: mne.Raw objects: pp_signal, pp_preictal, pp_ictal
    """
    # (1) Get preprocessed (pp) whole signal and its preictal, ictal and postictal traces
    # pp_signal, pp_preictal, pp_ictal, pp_postictal
    return process_bids_epilepsy(subject)


def compute_coherence_matrices(subject, pp_preictal, pp_ictal, pp_postictal,
                               savefile=False, fmt='.npy', window_len=10, win_slide_len='None'):
    """
    2) for each signal trace, compute the coherence matrices and store them for later processing.

    :param subject: string, patient id
    :param pp_preictal: mne.Raw object, preictal iEEG signal trace
    :param pp_ictal: mne.Raw object, ictal iEEG signal trace
    :param pp_postictal: mne.Raw object, postictal iEEG signal trace
    :param savefile: (boolean), operator to save coh matrices into files, default False
    :param fmt: (str), format to save files in, default .npy
    :param window_len: (int), time bin in seconds for calculating coherence
    :param win_slide_len: (int), sliding window for moving coherence time bin
    :return:
    """
    # (2) Compute the coherence matrices
    # coh_x: { coherence: {}, phase: {}}
    preictal_coh_mat = cc(pp_preictal.get_data(), pp_preictal.info['sfreq'],
                          window_len=window_len, win_slide_len=win_slide_len)
    if savefile:
        _iterate_through_freq_bands(subject, preictal_coh_mat, mat_type='coh', trace_type='preictal', fmt=fmt)

    ictal_coh_mat = cc(pp_ictal.get_data(), pp_ictal.info['sfreq'],
                       window_len=window_len, win_slide_len=win_slide_len)
    if savefile:
        _iterate_through_freq_bands(subject, ictal_coh_mat, mat_type='coh', trace_type='ictal', fmt=fmt)

    postictal_coh_mat = cc(pp_postictal.get_data(), pp_postictal.info['sfreq'],
                           window_len=window_len, win_slide_len=win_slide_len)
    if savefile:
        _iterate_through_freq_bands(subject, postictal_coh_mat, mat_type='coh', trace_type='postictal', fmt=fmt)


def _iterate_through_freq_bands(subject, matrices, mat_type='coh', trace_type='whole', fmt='.npy'):
    """
    iterate through frequency bands.
    Note: by default, coherence matrices are computed by each different frequency band.
    :param subject: (str), patient id
    :param matrices:
    :param mat_type:
    :param trace_type:
    :param fmt:
    :return:
    """
    for band in matrices['coherence'].keys():
        filename = os.path.join(subject, band + fmt)
        save_matrices(filename, matrices['coherence'][band], mat_type=mat_type, trace_type=trace_type)


def compute_intrinsic_coherence_networks(subject, savefile=False):
    """
    3) identify ICNs with PCA/ICA and store them for later processing.
    :param subject: string, patient id
    :param savefile: (boolean), operator to save ICNs into files, default False
    :return:
    """
    # Compute ICNs for coherence matrices of 'preictal' signal trace
    preictal_dir_path = os.path.join(coh_mat_dir_preictal, f'{subject}')
    _iterate_through_coh_mat_dir(subject, preictal_dir_path, trace_type='preictal', savefile=savefile)

    # Compute ICNs for coherence matrices of 'ictal' signal trace
    ictal_dir_path = os.path.join(coh_mat_dir_ictal, f'{subject}')
    _iterate_through_coh_mat_dir(subject, ictal_dir_path, trace_type='ictal', savefile=savefile)

    # Compute ICNs for coherence matrices of 'ictal' signal trace
    postictal_dir_path = os.path.join(coh_mat_dir_postictal, f'{subject}')
    _iterate_through_coh_mat_dir(subject, postictal_dir_path, trace_type='postictal', savefile=savefile)


def _iterate_through_coh_mat_dir(subject, dir_path, trace_type='preictal', savefile=False):
    """
    3 (cont'd) Iterate through the files inside dir_path
    then compute ICN networks and (optionally) save them into files
    :param subject: string, patient id
    :param dir_path: (str), native.datapaths.coh_mat_dir_{whole|preictal|ictal}
    :param trace_type: (str) signal trace type, supoorted values: whole, preictal, ictal and postictal
    :param savefile: (boolean), operator to save ICNs into files, default False
    :return:
    """
    for coh_mat_filename in os.listdir(dir_path):
        # # Get coherence matrices file
        coh_mat_file = os.path.join(dir_path, coh_mat_filename)
        coh_matrices = np.load(coh_mat_file)

        # # get ICNs from coherence matrices
        icn_mat, _icn_proj = get_icn(coh_matrices)
        if savefile:
            filename = os.path.join(subject, coh_mat_filename)
            save_matrices(filename, icn_mat, mat_type='icn', trace_type=trace_type)


def visualize_matrix(subject, filename, mat_type='coh', trace_type='preictal', channels=None, title='', savemat=False):
    """
    Plot coherence matrices and ICNs.
    The patient is defaulted in main.

    :param subject: (str) patient id
    :param filename: (str) file name including file format (i.e., .npy)
    :param mat_type: (str) matrix type, supported values: coh and icn
    :param trace_type: (str) signal trace type, supported values: preictal, ictal and postictal
    :param channels: (list) list of strings with channel names
    :param title: (str) title of plot
    :param savemat: (bool) operator to save matrices to files, default False
    :return:
    """
    mat_file = get_matrices_file(filename, mat_type=mat_type, trace_type=trace_type)
    plot_matrices.plot_matrices(subject, np.load(mat_file), channels=channels, title=title,
                                savemat=savemat, mat_type=mat_type, trace_type=trace_type)


def visualize_icn(subject, filename, trace_type='preictal', title=None, soz=None, savefigs=False):
    """
    Plot chord diagrams and binary matrices of the ICNs
    :param subject: (str) patient id
    :param filename: (str) file name including file format (i.e., .npy)
    :param trace_type: (str) signal trace type, supported values: whole, preictal, ictal and postictal
    :param title: (str) title of plot
    :param soz: (list of strings) list of channels that belong to the clinically annotated SOZ
    :param savefigs: (bool) operator to save graphs and matrices to files, default False
    :return:
    """
    # get the brain regions from the patient
    regions = get_regions(subject)
    regions_dict_inv = get_regions_dict_inv(subject)

    # get ICN file and load the ICN matrix
    icn_mat_file = get_matrices_file(filename, mat_type='icn', trace_type=trace_type)
    icn_matrices = np.load(icn_mat_file)

    # create an IcnGraph object
    icn_graph = icnetworks.IcnGraph(subject, icn_matrices, regions, quantileThresh=98, weightThresh=0.3)
    # plot
    plot_graphs(subject, icn_graph.graphs, regions_dict=icn_graph.regions_dict, regions_dict_inv=regions_dict_inv,
                title=title, nodefill=icn_graph.unfill_soz(soz), savegraph=savefigs,
                trace_type=trace_type)
