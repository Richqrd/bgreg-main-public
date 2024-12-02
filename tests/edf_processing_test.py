from pathlib import Path
from bgreg.native.datapaths import get_brain_greg_dir
from projects.ieeg_and_dti.preprocess import run


run(Path(get_brain_greg_dir(), "data_files"))
