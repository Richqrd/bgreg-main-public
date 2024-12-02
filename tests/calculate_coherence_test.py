from bgreg.tools.epilepsy.preprocess import process_bids_epilepsy
from bgreg.data.analysis.fcn.metrics import calculate_coherence_full_signal as cc
from bgreg.data.io import get_raw_signal_bids
from bgreg.utils.visual import plot_matrices


subject = 'pt6'
window_len = 30
win_slide_len = window_len


raw_signal = get_raw_signal_bids(subject)
pp_preictal, pp_ictal, pp_postictal = process_bids_epilepsy(raw_signal)

signaldata = pp_preictal.get_data()

two_signal = signaldata[:6, :]

preictal_processed_dict = cc(two_signal, pp_preictal.info['sfreq'],
                             window_len=window_len, win_slide_len=win_slide_len)
plot_matrices.plot_matrices(subject, preictal_processed_dict['coherence']['delta'])

