"""
    This script preprocesses signals for one or multiple subjects,
    and splits them into three signal traces: i) preictal, ii) ictal, iii) postictal.

    Signal traces are saved into pickle files in the preprocessed_data_dir (see native.datapaths.py).

    The flow of the script is to:
        - create the preprocessed_data_dir
        - iterate through the data.ds003029-data/ dir
            -- for each subject subdirectory (i.e., sub-ummc001)
                --- create a subdirectory where to dump the pickle files: preprocessed_data_dir/{subjectID}
                --- call bidsproc.process_bids_epilepsy({subjectID})
                --- dump pickle file into preprocessed_data_dir/{subjectID}


    NOTES:
        - the download API from OpenNeuro is buggy (Jan 2023), and not all subject files are downloaded sometimes,
            thus is necessary to manually download missing subject files.
        - some iEEG files within this dataset (OpenNeuro ds003029) are not annotated for seizure onset/offset,
            thus, they need to be excluded. Hence, this script is customized to only create the pickle files
            for those subjects that we have checked for the annotations, 25 subjects in total.
"""
import os
import pickle
from bgreg.native.datapaths import preprocessed_data_dir
from bgreg.data.io import get_all_raw_signals_bids
from bgreg.data.preprocess import preprocessed_filename
from bgreg.tools.epilepsy.preprocess import process_bids_epilepsy


if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)


subject_dirs = ["jh101", "jh103", "jh108",
                "pt01", "pt10", "pt11", "pt12", "pt13", "pt14", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc009"]

for subject in subject_dirs:
    dump_dir = os.path.join(preprocessed_data_dir, subject)
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    print("Processing subject", subject)
    # Read data files for each subject and get raw iEEG signals
    # Note: raw_signal's type is mne.io.brainvision.brainvision.RawBrainVision
    # runs is a tuple (raw_signal, run ID), i.e., (RawBrainVision, 1)
    runs = get_all_raw_signals_bids(subject)
    for run in runs:
        print("Processing run", str(run[1]))
        if subject in ["ummc001", "ummc002"]:
            # these ummc subjects were sampled at ~500Hz
            pp_preictal, pp_ictal, pp_postictal = process_bids_epilepsy(run[0], highcut=249.0)
        elif subject in ["ummc003", "ummc004", "ummc006", "ummc008"]:
            # these ummc subjects were sampled at ~250Hz
            pp_preictal, pp_ictal, pp_postictal = process_bids_epilepsy(run[0], highcut=124.0)
        else:
            # pt and jh subjects were sampled at 1000Hz (default highcut=499.0)
            pp_preictal, pp_ictal, pp_postictal = process_bids_epilepsy(run[0])

        pickle.dump(pp_preictal, open(os.path.join(dump_dir, preprocessed_filename("preictal", run[1])), "wb"))
        pickle.dump(pp_ictal, open(os.path.join(dump_dir, preprocessed_filename("ictal", run[1])), "wb"))
        pickle.dump(pp_postictal, open(os.path.join(dump_dir, preprocessed_filename("postictal", run[1])), "wb"))
