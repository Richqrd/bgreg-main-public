import os
import pickle

from numpy import floor, where

from mne import events_from_annotations

from bgreg.data.preprocess import preprocessed_filename, filter_bad_channels, remove_power_source_noise, bandpass


def process_bids_epilepsy(raw_signal, nt_freq=60, nt_notch_widths=4,
                          nt_method='fir', nt_fir_window='hamming', iir_order=4,
                          iir_ftype='butter', iir_output='sos', lowcut=0.1, highcut=499.0):
    """
    Process BIDS raw signal data and split it into preictal, ictal, and postictal signal traces
    :param raw_signal: (mne.Raw) iEEG signal
    :param nt_freq: (int) frequency to attenuate with notch filter
    :param nt_notch_widths: (int) width of notch filter
    :param nt_method: (str) filter method, default 'fir'
    :param nt_fir_window: (str) fir window, default 'hamming'
    :param iir_order: (int) order of bandpass filter
    :param iir_ftype: (str) type of filter, default 'butter'
    :param iir_output: (str) output of filter, default 'sos'
    :param lowcut: (float) low frequency end for bandpass
    :param highcut: (float) high frequency end for bandpass
    :return:
    """
    # Preprocess iEEG signal (filter bad channels, power noise, and bandpass)
    pp_signal = preprocess_bids_epilepsy(raw_signal, nt_freq=nt_freq, nt_notch_widths=nt_notch_widths,
                                         nt_method=nt_method, nt_fir_window=nt_fir_window,
                                         iir_order=iir_order, iir_ftype=iir_ftype, iir_output=iir_output,
                                         lowcut=lowcut, highcut=highcut)
    # Split signal in pre-ictal, ictal, and post-ictal traces
    return split_signal(pp_signal)


def split_signal(raw_signal):
    """
    Get seizure onset and offset time points,
    then slice signal into pre-ictal and ictal traces
    :param raw_signal: (Raw)
    :return: raw_pre_ictal, raw_ictal (whole), and raw_post_ictal signal traces
    """
    # Get onset and offset time points (floats)
    onset_time_sec, offset_time_sec = get_onset_offset_times(raw_signal)
    # Get last time point in signal
    last_time_sec = (raw_signal.n_times - 1) / raw_signal.info['sfreq']

    raw_preictal = raw_signal.copy().crop(tmin=0.0, tmax=onset_time_sec, include_tmax=False, verbose=False)
    raw_ictal = raw_signal.copy().crop(tmin=onset_time_sec, tmax=offset_time_sec, include_tmax=False, verbose=False)
    raw_postictal = raw_signal.copy().crop(tmin=offset_time_sec, tmax=floor(last_time_sec),
                                           include_tmax=False, verbose=False)

    return raw_preictal, raw_ictal, raw_postictal


def in_keyword(event, ktype="onset"):
    """
    Check if the annotation (event) contains a keyword
    :param event: string, annotation from data
    :param ktype: string, onset or offset
    :return: True if keyword is in annotation, False otherwise
    """
    onset_keywords = ['onset', 'sz event', 'sz start']
    offset_keywords = ['offset', 'seizure off', 'post ictal', 'postictal',
                       'over', 'post-ictal', 'sz end', 'end fast', 'z electrographic end',
                       'z stopping', 'definite off']
    if ktype == "onset":
        for kw in onset_keywords:
            if kw in event:
                return True
    else:
        for kw in offset_keywords:
            if kw in event:
                return True
    return False


def get_onset_offset_times(raw_signal):
    """
    Get onset and offset time points in seconds from the Raw signal objects
    :param raw_signal: mne.Raw signal object
    :return: (float) seizure onset and offset time points
    """
    # Read annotations from BIDS-formatted raw signal object
    events_from_annot, event_dict = events_from_annotations(raw_signal, verbose=False)
    # Initialize onset_time and offset_time variables
    onset_time = offset_time = ''

    # Iterate through annotations
    for event in event_dict.keys():
        # Hardcoded keywords tailored to onset time in OpenNeuro ds003029 dataset
        if in_keyword(event.lower()):
            onset_id = event_dict[event]
            onset_index = int(where(events_from_annot == onset_id)[0][0])
            onset_time = events_from_annot[onset_index][0]
        # Hardcoded keywords tailored to offset time in OpenNeuro ds003029 dataset
        elif in_keyword(event.lower(), ktype="offset"):
            offset_id = event_dict[event]
            offset_index = int(where(events_from_annot == offset_id)[0][0])
            offset_time = events_from_annot[offset_index][0]

    # Convert time in samples to seconds, where sampling freq = samples/second, and onset/offset in samples
    onset_time_sec = onset_time / raw_signal.info['sfreq']
    offset_time_sec = offset_time / raw_signal.info['sfreq']
    return onset_time_sec, offset_time_sec


def preprocess_bids_epilepsy(raw_signal, nt_freq=60, nt_notch_widths=4, nt_method='fir',
                             nt_fir_window='hamming', iir_order=4, iir_ftype='butter',
                             iir_output='sos', lowcut=0.1, highcut=499.0):
    """
    Preprocess a raw iEEG signal with the following steps:
        - Remove clinically annotated bad channels.
        - Apply notch filter (nt_ prefix below)
        - Apply bandpass filter (iir_ prefix below)
    Parameters
    ----------
    :param raw_signal: (mne.Raw) patient's raw signal data
    :param nt_freq: (int) frequency to attenuate with notch filter
    :param nt_notch_widths: (int) width of notch filter
    :param nt_method: (str) filter method, default 'fir'
    :param nt_fir_window: (str) fir window, default 'hamming'
    :param iir_order: (int) order of bandpass filter
    :param iir_ftype: (str) type of filter, default 'butter'
    :param iir_output: (str) output of filter, default 'sos'
    :param lowcut: (float) low frequency end for bandpass
    :param highcut: (float) high frequency end for bandpass

    Returns
    -------
    mne.io.RawArray with preprocessed signal with annotations

    """

    # Removing bad channels predetermined in BIDS dataset and reordering the channels such that
    # same channel type names are corresponding to one another
    filtbad_signal = filter_bad_channels(raw_signal)

    # Apply notch filter to power source noise frequencies and its harmonics
    notchfilt_signal = remove_power_source_noise(filtbad_signal, nt_freq=nt_freq,
                                                 nt_notch_widths=nt_notch_widths,
                                                 nt_method=nt_method,
                                                 nt_fir_window=nt_fir_window)

    bandpass_signal = bandpass(notchfilt_signal, iir_order=iir_order, iir_ftype=iir_ftype,
                               iir_output=iir_output, lowcut=lowcut, highcut=highcut)
    return bandpass_signal


def save_bids_epilepsy(pp_preictal, pp_ictal, pp_postictal, dump_dir, filename, run):
    pickle.dump(pp_preictal, open(os.path.join(dump_dir, preprocessed_filename("preictal", run[1])), "wb"))
    pickle.dump(pp_ictal, open(os.path.join(dump_dir, preprocessed_filename("ictal", run[1])), "wb"))
    pickle.dump(pp_postictal, open(os.path.join(dump_dir, preprocessed_filename("postictal", run[1])), "wb"))
