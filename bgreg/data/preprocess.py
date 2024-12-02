import pickle
from pathlib import Path

from numpy import floor, arange
from scipy.signal import sosfiltfilt
from mne.io import RawArray
from mne.filter import construct_iir_filter

from bgreg.native.datapaths import preprocessed_data_dir


def filter_bad_channels(raw_signal, channels=None):
    """
    Remove bad channels from mne.Raw object.
    Will first remove channels from raw_signal.info["bads"],
    then also channels that contain symbols in their names,
    and last it will remove the channels in the 'channels' param if passed.

    :param raw_signal: mne.Raw signal object
    :param channels: list of strings (optional), channel names to remove not
        included in raw_signal.info["bads"]
    :return: mne.Raw signal object
    """
    # Get bad channels from raw_signal
    bads = raw_signal.info["bads"]
    goods = []
    for channel in raw_signal.info["ch_names"]:
        if any(not c.isalnum() for c in channel) or channel in bads:
            # Append to bad channels that include symbols in their name (i.e., $G12)
            bads.append(channel)
        else:
            goods.append(channel)
    # Append additional channels
    if channels is not None:
        for channel in channels:
            bads.append(channel)

    # Select only ECoG signals and exclude bads from raw_signal
    return raw_signal.pick(goods)


def remove_power_source_noise(signal, nt_freq=60, nt_notch_widths=4, nt_method='fir',
                              nt_fir_window='hamming'):
    # Apply notch filter to power source noise frequencies and its harmonics
    harmonics = int(floor(signal.info['sfreq'] / 2 / nt_freq))
    freqs = [fq * nt_freq for fq in arange(1, harmonics + 1)]
    freqs = [freqs]
    notchfilt_signal = signal.load_data().notch_filter(freqs=freqs,
                                                       notch_widths=nt_notch_widths,
                                                       method=nt_method,
                                                       fir_window=nt_fir_window,
                                                       verbose=False)
    return notchfilt_signal


def bandpass(signal, iir_order=4, iir_ftype='butter',
             iir_output='sos', lowcut=0.1, highcut=499.0):
    # Apply IIR filter butterworth order 4 for (0.5 - nyquist frequency) Hz with transition bandwidth = 2 Hz
    sfreq = signal.info['sfreq']
    iir_params = dict(order=iir_order, ftype=iir_ftype, output=iir_output)
    iir_filter = construct_iir_filter(iir_params=iir_params,
                                      f_pass=[lowcut, highcut],
                                      sfreq=sfreq,
                                      btype='bandpass',
                                      return_copy=False,
                                      verbose=False)
    iirfilt_signal = RawArray(sosfiltfilt(iir_filter[iir_output],
                                          signal.get_data()),
                              signal.info, verbose=False)
    iirfilt_signal.set_annotations(signal.annotations)
    return iirfilt_signal


def get_preprocessed_data(sub, run):
    """
    Get the preprocessed preictal, ictal, and postictal signal traces from the directory
    'datapaths.preprocessed_data_dir' for any given subject 'sub'
    :param sub: string, subject ID, i.e., "ummc001"
    :param run: int, run ID
    :return: arrays of pp_preictal, pp_ictal, pp_postictal signal traces
    """
    pp_preictal = pickle.load(open(Path(preprocessed_data_dir, sub, preprocessed_filename("preictal", run)), "rb"))
    pp_ictal = pickle.load(open(Path(preprocessed_data_dir, sub, preprocessed_filename("ictal", run)), "rb"))
    pp_postictal = pickle.load(open(Path(preprocessed_data_dir, sub, preprocessed_filename("postictal", run)), "rb"))
    return pp_preictal, pp_ictal, pp_postictal


def preprocessed_filename(trace_type, run_no):
    return "pp_" + trace_type + "_" + str(run_no) + ".pickle"
