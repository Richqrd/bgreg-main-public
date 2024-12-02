"""
    This script computes statistics about the patients' runs
"""
import os
from numpy import mean, std
from bgreg.native.datapaths import preprocessed_data_dir
from bgreg.data.io import get_all_raw_signals_bids


if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)

subject_dirs = ["jh101", "jh102", "jh103", "jh108",
                "pt01", "pt10", "pt11", "pt12", "pt13", "pt14", "pt15", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc008", "ummc009"]

ch_no = []
duration = []
allruns = 0
perpatientrun = []

for subject in subject_dirs:
    dump_dir = os.path.join(preprocessed_data_dir, subject)
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    print("Processing subject", subject)
    # Read data files for each subject and get raw iEEG signals
    # Note: raw_signal's type is mne.io.brainvision.brainvision.RawBrainVision
    runs = get_all_raw_signals_bids(subject)
    ppr = 0
    for run in runs:
        allruns += 1
        ppr += 1
        print("Processing run", str(run[1]))
        ch_no.append((len(run[0].ch_names)))
        duration.append(run[0].n_times)
    perpatientrun.append(ppr)

print("All runs:", allruns)

print("Average number of runs across patients:", mean(perpatientrun))
print("STD number of runs across patients", std(perpatientrun))

print("Average number of channels", mean(ch_no))
print("STD number of channels", std(ch_no))

print("Average duration", mean(duration)/1000)
print("STD duration", std(duration)/1000)
