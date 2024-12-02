import os
import pickle

from bgreg.data.io import get_raw_signal_edf
from bgreg.tools.epilepsy.preprocess import process_bids_epilepsy, preprocess_bids_epilepsy


def run(datadir, both=False, save_pickle=False):
    twhdata_dir = os.path.join(datadir, "TWH-data")

    dataprocdir = os.path.join(datadir, "TWH-processed")
    datadumpdir = os.path.join(dataprocdir, "preprocessed_data")

    d07_f = "S01-D07-SZ01_deidentified.edf"  # 220 secs
    d26_f = "S01-D26-SZ02_deidentified.edf"  # 1 hours

    if both:
        dfiles = [("d07", d07_f), ("d26", d26_f)]
    else:
        dfiles = [("d07", d07_f)]

    for df in dfiles:
        dumpdir = os.path.join(datadumpdir, df[0])

        if not os.path.exists(dumpdir):
            os.mkdir(dumpdir)

        dsub = get_raw_signal_edf(os.path.join(twhdata_dir, df[1]))

        pp_whole = preprocess_bids_epilepsy(dsub)
        pickle.dump(pp_whole, open(os.path.join(dumpdir, "pp_whole.pkl"), "wb"))

        if df[0] == "d26":
            continue

        pp_preictal, pp_ictal, pp_postictal = process_bids_epilepsy(dsub)

        if save_pickle:
            # Save preprocessed signal traces in serialized object files.
            pickle.dump(pp_preictal, open(os.path.join(dumpdir, "pp_preictal.pkl"), "wb"))
            pickle.dump(pp_ictal, open(os.path.join(dumpdir, "pp_ictal.pkl"), "wb"))
            pickle.dump(pp_postictal, open(os.path.join(dumpdir, "pp_postictal.pkl"), "wb"))
