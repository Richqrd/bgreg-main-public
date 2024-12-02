import os
import pickle
from mne.export import export_raw


subjects = ["d07", "d26"]


datadir = os.path.join(os.path.abspath(os.getcwd()), "data_files")
twhdata_dir = os.path.join(datadir, "TWH-data")

dataprocdir = os.path.join(datadir, "TWH-processed")
ppdir = os.path.join(dataprocdir, "preprocessed_data")

for sub in subjects:
    subdir = os.path.join(ppdir, sub)
    with open(os.path.join(subdir, "pp_whole.pkl"), "rb") as pp_whole_pkl:
        pp_whole = pickle.load(pp_whole_pkl)
        exportf = os.path.join(subdir, "pp_whole.edf")
        export_raw(exportf, pp_whole, fmt="edf", overwrite=True)
