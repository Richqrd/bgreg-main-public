import os

from mne_bids import BIDSPath, read_raw_bids
from bgreg.native.datapaths import datastore_path, preprocessed_data_dir
from bgreg.data.preprocess import filter_bad_channels
import pyabf
from mne.io import read_raw_edf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_raw_signal_bids(subject, datatype='ieeg', session='presurgery',
                        task='ictal', acq='ecog', run=1, suffix='ieeg'):
    """
    Interface for BIDSPath and read_raw_bids
    Return the raw signal in Raw BIDS format
    :param subject: (str) ID of patient to inspect
    :param datatype: (str), BIDS data type
    :param session: (str), acquisition session
    :param task: (str), experimental task
    :param acq: (str), acquisition parameters
    :param run: (int), run number
    :param suffix: (str), filename suffix
    :return: mne.Raw signal object
    """

    # Get one patient's run
    bids_path = BIDSPath(subject=subject, session=session, task=task,
                         acquisition=acq, run=run, suffix=suffix,
                         datatype=datatype, root=datastore_path)
    # BIDS dataset into mne 'raw' object
    return read_raw_bids(bids_path=bids_path, verbose=False)


def get_all_raw_signals_bids(subject):
    """
    Read and return BIDS signals for all runs of a given subject
    """
    sub = "sub-" + subject
    sub_dir = os.path.join(datastore_path, sub, "ses-presurgery", "ieeg")
    runs = []
    for file in sorted(os.listdir(sub_dir)):
        if "run" in file:
            substring = file.split("-")
            run = int(substring[-1].split("_")[0])
            if run not in runs:
                runs.append(run)

    raw_signals = []
    for run in runs:
        raw_signals.append((get_raw_signal_bids(subject, run=run), run))
    return raw_signals


def get_raw_signal_abf(filename):
    return pyabf.ABF(filename)


def get_raw_signal_edf(filename):
    return read_raw_edf(filename)


def get_run_from_filename(file):
    run = file.split("_")[-1]
    run = int(run.split(".")[0])
    return run


def get_runs(sub):
    """
    Return the number of runs according to the files in the preprocessed_data_dir
    directory declared in native.datapaths.

    :param sub: string, patient ID (i.e., pt2).

    :return: list with run ids (i.e., [1, 2, 3])
    """
    sub_dir = os.path.join(preprocessed_data_dir, sub)
    runs = []
    for file in sorted(os.listdir(sub_dir)):
        if any(chr.isdigit() for chr in file):
            run = get_run_from_filename(file)
            if run not in runs:
                runs.append(run)
            else:
                break
    return runs


def get_regions(subject):
    """
    Get regions from channel names
    :param subject: (str), patient id
    :return: list with regions
    """
    ch_names = get_channel_names(subject)
    stp_channels = strip_channel_names(ch_names)
    return list(filter(''.__ne__, stp_channels))


def get_regions_dict(subject):
    """
    Compute the region dictionary hashing region id (i.e., AD)
    to the channel count as found in the signal object (i.e., 0, 1, 2, 3)
    :param subject: (str) patient id
    :return: dictionary in the form {region: [ch count1, ch count2, etc.]}
    """
    channels = get_channel_names(subject)
    reg_dict = {}
    for ch_count, channel in enumerate(channels):
        stp_channel = strip_channel(channel)
        ch_name = stp_channel[0]
        if ch_name not in reg_dict.keys():
            reg_dict.setdefault(ch_name, [])
            reg_dict[ch_name].append(ch_count)
        else:
            reg_dict[ch_name].append(ch_count)
    return reg_dict


def get_regions_dict_inv(subject):
    """
    Compute the region dictionary hashing channel count as found in the signal object (i.e., 0, 1, 2, 3)
    to the region id (i.e., AD)
    :param subject: (str) patient id
    :return: dictionary in the form {region: [ch count1, ch count2, etc.]}
    """
    channels = get_channel_names(subject)
    reg_dict = {}
    for ch_count, channel in enumerate(channels):
        stp_channel = strip_channel(channel)
        ch_name = stp_channel[0]
        reg_dict[ch_count] = ch_name
    return reg_dict


def get_channel_names(subject, filtbad=True):
    """
    Get channel names from patient file
    :param subject: (str) patient id
    :param filtbad: (boolean), operator to filter bad channels
    :return:
    """
    raw_signal = get_raw_signal_bids(subject)
    if filtbad:
        return filter_bad_channels(raw_signal).ch_names
    else:
        return raw_signal.ch_names


def strip_channel_names(channels):
    """
    Strip channel names to separate literal characters
    from numeric characters
    :param channels: (list of strings), list of channels to strip
    :return: list with characters separated
    """
    ticks = []
    for channel in channels:
        stp_channel = strip_channel(channel)
        ch_name = stp_channel[0]
        if ch_name not in ticks:
            ticks.append(ch_name)
        else:
            ticks.append('')
    return ticks


def strip_channel(channel):
    """
    Strip channel names to separate literal characters
    from numeric characters
    :param channel: (str), channel id
    :return: head and tail from channel
    """
    head = channel.rstrip('0123456789')
    tail = channel[len(head):]
    return head, tail
