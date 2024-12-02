from scipy.signal import butter, lfilter
from scipy.signal.spectral import csd
from numpy import sum, mean, corrcoef, array, transpose, zeros, where, \
    angle, floor, linspace, arange, linalg, expand_dims


def corr(x):
    return corrcoef(x)


def coh(x):
    result = calculate_coherence_signal_trace(x, x.shape[1])
    coherences_dict = result['coherence']
    coherences = []
    for key in coherences_dict.keys():
        coherences.append(array(coherences_dict[key]))
    coherences = array(coherences)
    coherences = transpose(coherences, (3, 1, 2, 0))
    return coherences[0]


def phase(x):
    result = calculate_coherence_signal_trace(x, x.shape[1])
    phases_dict = result['phase_dict']
    phases = []
    for key in phases_dict.keys():
        phases.append(array(phases_dict[key]))
    phases = array(phases)
    phases = transpose(phases, (3, 1, 2, 0))
    return phases[0]


def calculate_coherence_signal_trace(signal_trace, fs):
    electrode_no = len(signal_trace)
    # Initialize output matrices for coherence and phase values, and freq_indices:
    coherence_dict = {}
    phase_dict = {}
    freq_bands = fq_bands(fs)
    # Fill-in initialized matrices
    for band in freq_bands.keys():
        coherence_dict[band] = zeros((electrode_no, electrode_no))
        phase_dict[band] = zeros((electrode_no, electrode_no))

    # get initialized 3D arrays (matrices) of the coherence, phase, and ij_pairs
    ij_pairs = get_ij_pairs(electrode_no)

    # initialize Cxy_dict and phase dictionaries, and list of frequencies
    # Cxy_dict is a dictionary in the form of: (0, 1): [coh-freq1, coh-freq2, ..., coh-freqNyq]
    # for all pairs of electrodes
    # Cxy_phase_dict is also a dictionary in the form of: (0, 1): [time1, time2, ..., timeN] for all electrodes
    # fqs is a list of frequencies
    Cxy_dict = {}
    Cxy_phase_dict = {}
    freqs = []
    # check every electrode pair only once
    for electrode_pair in ij_pairs:
        # initialization of dictionaries to electrode_pair key
        Cxy_dict.setdefault(electrode_pair, {})
        Cxy_phase_dict.setdefault(electrode_pair, {})
        # get signals by index
        x = signal_trace[electrode_pair[0]]
        y = signal_trace[electrode_pair[1]]
        # compute coherence
        nperseg = _nperseg(fs)
        freqs, Cxy, ph, Pxx, Pyy, Pxy = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=16)
        # x and y are the first and second signal we compare against
        # freqs = frequencies that are returned by the coherence function
        # in coherence function computing cross spectral density which gives us this evaluation
        # cross spectral density is a function that looks at what are the frequencies that compose the signal in x

        Cxy_dict[electrode_pair] = Cxy
        Cxy_phase_dict[electrode_pair] = ph

    # Create numpy array of keys and values:
    Cxy_keys = array(list(Cxy_dict.keys()))
    Cxy_values = array(list(Cxy_dict.values()))
    phase_keys = array(list(Cxy_phase_dict.keys()))
    phase_values = array(list(Cxy_phase_dict.values()))

    # Create dictionary with freq-band as keys and list of frequency indices from freqs as values, i.e.
    # freq_indices = {'delta': [1, 2]}
    freq_indices = {}
    for band in freq_bands.keys():
        freq_indices[band] = list(where((freqs >= freq_bands[band][0]) & (freqs <= freq_bands[band][1]))[0])

    # filter for signals that are present that correspond to different bands
    # For each freq band (delta...) is a range, here we are filtering using freqs, which contains frequencies found
    # in signal when converted to frequency domain

    # average over the frequency bands; row averaging
    coh_mean = {}
    phase_mean = {}
    for band in freq_bands.keys():
        coh_mean[band] = mean(Cxy_values[:, freq_indices[band]], axis=1)
        phase_mean[band] = mean(phase_values[:, freq_indices[band]], axis=1)

    for band in freq_bands.keys():
        # Fill coherence_dict matrices:
        # Set diagonals = 1
        coherence_dict[band][range(electrode_no), range(electrode_no)] = 1
        # Fill in rest of the matrices
        for pp, pair in enumerate(Cxy_keys):
            coherence_dict[band][pair[0], pair[1]] = coh_mean[band][pp]
            coherence_dict[band][pair[1], pair[0]] = coh_mean[band][pp]

        # Fill phase matrices:
        # Set diagonals = 1
        phase_dict[band][range(electrode_no), range(electrode_no)] = 1
        # Fill in rest of the matrices
        for pp, pair in enumerate(phase_keys):
            phase_dict[band][pair[0], pair[1]] = phase_mean[band][pp]
            phase_dict[band][pair[1], pair[0]] = phase_mean[band][pp]

    return {'coherence': coherence_dict, 'phase_dict': phase_dict, 'freq_dicts': freq_indices, 'freqs': freqs}


def calculate_coherence_full_signal(signal_array, fs, window_len=10, win_slide_len='None'):
    """
    TODO: decide what to do with this function
    Calculates signal coherence_dict between all electrode pairs, over pre-defined frequency bands.
    Return a dictionary in the form:
        {'coherence_dict': {}, 'phase_dict': {}, 'deltaT': int, 'tAxis': ndarray}
    :param signal_array: array, ecog signals array (n electrodes x t timepoints)
    :param fs: float, sampling frequency
    :param window_len: int, time bin in seconds for calculating coherence_dict
    :param win_slide_len: int or 'None', sliding window for moving coherence_dict time bin
    """

    if win_slide_len == 'None':
        win_slide_len = window_len

    window_len = float(window_len)
    win_slide_len = float(win_slide_len)

    electrode_no, signal_len, signal_timepoints, segment_intervals, nsegments = number_of_segments(signal_array,
                                                                                                   fs,
                                                                                                   window_len,
                                                                                                   win_slide_len)

    # Initialize output matrices for coherence and phase values, and freq_indices:
    coherence_dict = {}
    phase_dict = {}
    freq_indices = {}
    freq_bands = fq_bands(fs)
    # Fill-in initialized matrices
    for band in freq_bands.keys():
        coherence_dict[band] = zeros((electrode_no, electrode_no, nsegments))
        phase_dict[band] = zeros((electrode_no, electrode_no, nsegments))

    # get initialized 3D arrays (matrices) of the coherence, phase, and ij_pairs
    ij_pairs = get_ij_pairs(electrode_no)

    # Loop through each segment (matrix):
    for segment_ind, segment_start in enumerate(segment_intervals[:-1]):
        segment_end = segment_intervals[segment_ind + 1]
        # get segment indices from signal indices
        segment_inter_temp = where((signal_timepoints >= segment_start) & (signal_timepoints <= segment_end))[0]

        # Only perform coherence calculation if have >= 1 second of data
        if len(segment_inter_temp) >= int(fs):
            segment_signal_array = signal_array[0:electrode_no, segment_inter_temp[0]:(segment_inter_temp[-1] + 1)]

            # initialize Cxy_dict and phase dictionaries, and list of frequencies
            # Cxy_dict is a dictionary in the form of: (0, 1): [coh-freq1, coh-freq2, ..., coh-freqNyq]
            # for all pairs of electrodes
            # Cxy_phase_dict is also a dictionary in the form of: (0, 1): [time1, time2, ..., timeN] for all electrodes
            # fqs is a list of frequencies
            Cxy_dict = {}
            Cxy_phase_dict = {}
            freqs = []
            # check every electrode pair only once
            for electrode_pair in ij_pairs:
                # initialization of dictionaries to electrode_pair key
                Cxy_dict.setdefault(electrode_pair, {})
                Cxy_phase_dict.setdefault(electrode_pair, {})
                # get signals by index
                x = segment_signal_array[electrode_pair[0]]
                y = segment_signal_array[electrode_pair[1]]
                # compute coherence
                freqs, Cxy, ph, Pxx, Pyy, Pxy = coherence(x, y, fs=fs, nperseg=int(fs / 2))

                Cxy_dict[electrode_pair] = Cxy
                Cxy_phase_dict[electrode_pair] = ph

            # Create numpy array of keys and values:
            Cxy_keys = array(list(Cxy_dict.keys()))
            Cxy_values = array(list(Cxy_dict.values()))
            phase_keys = array(list(Cxy_phase_dict.keys()))
            phase_values = array(list(Cxy_phase_dict.values()))

            # Create dictionary with freq-band as keys and list of frequency indices from freqs as values, i.e.
            # freq_indices = {'delta': [1, 2]}
            freq_indices = {}
            for band in freq_bands.keys():
                freq_indices[band] = list(where((freqs >= freq_bands[band][0]) & (freqs <= freq_bands[band][1]))[0])

            # average over the frequency bands; row averaging
            coh_mean = {}
            phase_mean = {}
            for band in freq_bands.keys():
                coh_mean[band] = mean(Cxy_values[:, freq_indices[band]], axis=1)
                phase_mean[band] = mean(phase_values[:, freq_indices[band]], axis=1)

            for band in freq_bands.keys():
                # Fill coherence_dict matrices:
                # Set diagonals = 1
                coherence_dict[band][range(electrode_no), range(electrode_no), segment_ind] = 1
                # Fill in rest of the matrices
                for pp, pair in enumerate(Cxy_keys):
                    coherence_dict[band][pair[0], pair[1], segment_ind] = coh_mean[band][pp]
                    coherence_dict[band][pair[1], pair[0], segment_ind] = coh_mean[band][pp]

                # Fill phase matrices:
                # Set diagonals = 1
                phase_dict[band][range(electrode_no), range(electrode_no), segment_ind] = 1
                # Fill in rest of the matrices
                for pp, pair in enumerate(phase_keys):
                    phase_dict[band][pair[0], pair[1], segment_ind] = phase_mean[band][pp]
                    phase_dict[band][pair[1], pair[0], segment_ind] = phase_mean[band][pp]

    return {'coherence': coherence_dict, 'phase_dict': phase_dict, 'freq_dicts': freq_indices}


def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    """
    Overriding scipy.signal.spectral.coherence method
    to calculate phase lag from CSD
    :return: freqs (ndarray), Cxy (ndarray), phase (ndarray), Pxx (ndarray), Pyy (ndarray), Pxy (ndarray)
    """


    # power spectral density = signal in frequency domain
    # pxx and pyy are the PSD lines
    freqs, Pxx = csd(x, x, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    _, Pyy = csd(y, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                 nfft=nfft, detrend=detrend, axis=axis)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)

    ph = angle(Pxy, deg=False)

    # formula for coherence
    Cxy = abs(Pxy) ** 2 / (Pxx * Pyy)

    return freqs, Cxy, ph, Pxx, Pyy, Pxy


"""
    Helper functions
"""


def fq_bands(fs):
    """
    Pass segments of frequency bands constrained to the sampling frequency 'fs'
    :param fs: int, sampling frequency
    :return: dictionary with frequency bands
    """
    if fs < 499:
        return {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 70),
                'gammaHi': (70, 100)}
    elif fs < 999:
        return {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 70),
                'gammaHi': (70, 100), 'ripples': (100, 250)}
    # Define frequency oscillation bands, range in Hz:
    return {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 70),
            'gammaHi': (70, 100), 'ripples': (100, 250), 'fastRipples': (250, 500)}


    # return {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'b1': (13, 20), 'b2': (20,30), 'g1': (30, 40),
    #         'g2': (40,50), 'g3': (50,60), 'g4': (60,70),
    #         'gh1': (70, 80), 'gh2': (80, 90), 'gh3': (90, 100),
    #         'r1': (100, 130), 'r2': (130, 160), 'r3': (160, 190), 'r4': (190, 220), 'r5': (220, 250),
    #         'fr1': (250, 300), 'fr2': (300, 350), 'fr3': (350, 400), 'fr4': (400, 450), 'fr5': (450, 500)
    #         }


def _nperseg(fs):
    if fs <= 250:
        return 128
    return 256


def number_of_segments(signal_array, fs, window_len, win_slide_len):
    """
    Initialization of constants
    :param signal_array: ndarray, electrodes x timepoints
    :param fs: float, sampling frequency
    :param window_len: float, length of segment window
    :param win_slide_len: float, length of sliding window
    :return: electrode_no (int), signal_len (float), signal_timepoints (ndarray),
                segment_intervals (ndarray), nsegments (int)
    """
    # get number of electrodes
    electrode_no = len(signal_array)
    # calculate signal duration to closest int (floor)
    signal_len = floor(len(signal_array[0]) / float(fs))
    # obtain the number of timepoints within the signal ,
    signal_timepoints = linspace(0, signal_len, len(signal_array[0]))
    # calculate maximum time point
    max_time = signal_len - window_len + win_slide_len
    # obtain indices from signal_inds that correspond to each segment (matrix) signal chunk
    segment_intervals = arange(0, max_time + 1, win_slide_len)
    # the number of segments (matrices) = the length of timepoints_split
    nsegments = len(segment_intervals)

    return electrode_no, signal_len, signal_timepoints, segment_intervals, nsegments


def get_ij_pairs(electrode_no):
    """
    Get list of tuples with i, j pairs of electrodes;
        i, j are indices
    :param electrode_no: int, number of electrodes
    :return: ij_pairs (list of tuples(ints))
    """
    # Define electrode pairs over which to calculate
    # the coherence and save it as list of tuples
    ij_pairs = []
    for i in range(electrode_no):
        for j in range(i + 1, electrode_no):
            ij_pairs.append((i, j))

    return ij_pairs


def energy(x):
    nf = zeros(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        nf[i] = sum([j ** 2 for j in x[i]])
    nf /= linalg.norm(nf)
    nf = expand_dims(nf, -1)
    return nf


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def band_energy(x):
    result = calculate_coherence_signal_trace(x, x.shape[1])
    freq_dicts = result['freq_dicts']
    nf = [[] for i in range(x.shape[0])]
    for i in range(x.shape[0]):
        for band in freq_dicts.keys():
            freq_sig = butter_bandpass_filter(x[i], min(freq_dicts[band]), max(freq_dicts[band]) + 1, x.shape[1])
            nf[i].append(sum([j ** 2 for j in freq_sig]))
        nf[i] = array(nf[i])
    nf = array(nf)
    nf /= linalg.norm(nf, axis=0, keepdims=True)
    return nf
