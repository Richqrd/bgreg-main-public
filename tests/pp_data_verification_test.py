"""
    This script is to verify that the pickle files saved at the preprocessed_graph directory are not trunctated,
    and that the entries do not contain NaN values.
    The verification process consists in attempting to open the files with pickle.load(f).
    In case of an exception, a RuntimeError is shown.
"""
import os
import pickle
import numpy as np

from bgreg.native.datapaths import preprocessed_data_dir, graph_representation_elements_dir


def check_file(path, signal_or_graph):
    """

    :param path:
    :param signal_or_graph: string, indicate if verifying pp_signals ("s") or pp_graphs ("g")
    :return:
    """
    # Check if file is truncated by attempting to open it (pickle.load).
    print("\u001b[32m Verifying pickle files for {}".format(path.split("/")[-1]))
    for filename in os.listdir(path):
        if "pickle" in filename:
            try:
                with open(os.path.join(path, filename), "rb") as f:
                    # Check for truncation
                    unpickled = pickle.load(f)
                    # ch_names.pickle is just a list of strings (electrode labels)
                    if filename != "ch_names.pickle":
                        if signal_or_graph == "s":
                            check_pp_signal(unpickled, filename)
                        else:
                            check_for_nans_pp_graph(unpickled, filename)
                    print("\u001b[37m Preprocessed file {} saved successfully.".format(filename))
            except pickle.UnpicklingError:
                print("\u001b[31m An error occurred with preprocessed graph file {} for {}.".format(filename,
                                                                                                    path))


def check_for_nans_pp_graph(data_lists, filename):
    """
    preictal.pickle, ictal.pickle, and postistal.pickle contain the graph representations in list form.
    See the method save_data() from datautils.dataproc.dataset for more information on these objects.
    Given the structure of these objects, the coactivity measurements are in the 3 level:
        data_lists[level1][level2][level3]
    This 3rd level contains 4 lists of coactivity measurements corresponding to 1) ones, 2) corr, 3) coh, 4) plv,
    each containing 2-D arrays with dimensions (no. electrodes, no. electrodes).
    As such, this function iterates through these 2-D arrays to find NaN values
    :param data_lists: list of lists corresponding to a pickle object
    :param filename: string, filename being read for informational purposes
    :return:
    """
    for data_list in data_lists[0][0][3]:
        # check for NaN values in coactivity measurements
        if any(np.isnan(x) for x in data_list):
            print("\u001b[31m NaN values in {}.".format(filename))
            # Once one is found, there is no need for further checking
            break


def check_pp_signal(raw_signal, filename):
    """
    With this function we can inspect the preictal, ictal, and postictal traces.
    FIXME: Function not developed yet!
    :param raw_signal: RawArray (mne), signal trace
    :param filename: string, name of inspected file
    :return:
    """
    print("\u001b[37m Checking preprocessed signal file {}.".format(filename))
    raw_signal.plot(block=True)


def verify(path=None, signal_or_graph="s"):
    """
    Interface for checking pickle files by subject (passing path parameter) or in all subjects (path=None)
    :param path: string, path to subject
    :param signal_or_graph: string, indicate if verifying pp_signals ("s") or pp_graphs ("g")
    :return:
    """
    # Check for all subjects
    if path is None:
        if signal_or_graph == "s":
            path = preprocessed_data_dir
        else:
            path = graph_representation_elements_dir
        for sub_dir in os.listdir(path):
            sub_path = os.path.join(path, sub_dir)
            check_file(sub_path, signal_or_graph)
    # Check for individual subject
    else:
        check_file(path, signal_or_graph)


if __name__ == "__main__":
    subject = "ummc001_1s"
    t_sub_path_signal = os.path.join(preprocessed_data_dir, subject)
    t_sub_path_graph = os.path.join(graph_representation_elements_dir, subject)
    # replace path=None with t_sub_path for individual subject check-up
    verify(path=t_sub_path_graph, signal_or_graph="g")
