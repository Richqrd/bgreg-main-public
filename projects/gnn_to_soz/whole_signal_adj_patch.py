import os
import pickle

from bgreg.native.datapaths import preprocessed_data_dir
from bgreg.data.io import get_all_raw_signals_bids, get_runs
from bgreg.data.analysis.graph.graph_representation import save_adj_whole_signal
from bgreg.tools.epilepsy.preprocess import preprocessed_filename, preprocess_bids_epilepsy


def preprocess(patient_dirs):
    for patient in patient_dirs:
        try:
            dump_dir = os.path.join(preprocessed_data_dir, patient)
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)

            runs = get_all_raw_signals_bids(patient)
            for run in runs:
                if patient in ["ummc001", "ummc002"]:
                    # these ummc patients were sampled at ~500Hz
                    pp_signal = preprocess_bids_epilepsy(run[0], highcut=249.0)
                elif patient in ["ummc003", "ummc004", "ummc006", "ummc008"]:
                    # these ummc patients were sampled at ~250Hz
                    pp_signal = preprocess_bids_epilepsy(run[0], highcut=124.0)
                else:
                    # pt and jh patients were sampled at 1000Hz (default highcut=499.0)
                    pp_signal = preprocess_bids_epilepsy(run[0])

                pickle.dump(pp_signal, open(os.path.join(dump_dir, preprocessed_filename("signal", run[1])), "wb"))
        except Exception as e:
            print(e)


def save_adj_whole(patient_dirs):
    for patient in patient_dirs:
        try:
            for run in get_runs(patient):
                save_adj_whole_signal(patient, run)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    pt_dirs = ["jh101", "jh103", "jh108",
               "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
               "pt2", "pt3", "pt6", "pt7", "pt8",
               "umf001",
               "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
               "ummc007", "ummc009"]
    preprocess(pt_dirs)
    save_adj_whole(pt_dirs)
