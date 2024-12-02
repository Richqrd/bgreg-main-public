from bgreg.data.io import get_raw_signal_bids
from bgreg.data.preprocess import filter_bad_channels

import matplotlib

matplotlib.use('Qt5Agg')

# Subjects included in the analysis
subject = 'jh103'
datatype = 'ieeg'
session = 'presurgery'
task = 'ictal'
acq = 'ecog'
run = 3
suffix = 'ieeg'

# Read data files for each patient and get raw iEEG signals
# Note: raw_signal's type is mne.io.brainvision.brainvision.RawBrainVision
raw_signal = get_raw_signal_bids(subject, datatype=datatype, session=session,
                                 task=task, acq=acq, run=run, suffix=suffix)

# Filter bad channels
raw_signal = filter_bad_channels(raw_signal)

# Plot iEEG signals
raw_signal.plot(block=True)

# Plot power spectral density
# raw_signal.plot_psd(average=True)
