from bgreg.utils.dataproc.bidsproc import get_channel_names
from projects.ieeg_to_icn.pipeline import visualize_matrix


def inspect_single_band(subject, filename, mat_type='coh', trace_type='whole', channels=None, title='', savemat=False):
    """
    Plot coherence matrix for patient s in band b

    :param subject: (str) patient id
    :param filename: (str) file name including file format (i.e., .npy)
    :param mat_type: (str) matrix type to inspect
    :param trace_type: (str) signal trace type, supported values: whole, preictal, ictal and postictal
    :param channels: (list) list of strings with channel names
    :param title: (str) title of plot
    :param savemat:
    :return:
    """
    # plot
    visualize_matrix(subject, filename, mat_type=mat_type, trace_type=trace_type,
                     channels=channels, title=title, savemat=savemat)


if __name__ == "__main__":
    """
    Inspect only one band for one patient
    """
    # # patient to inspect
    sbj = 'pt6'
    # Get all channels from the signal
    ch_names = get_channel_names(sbj)
    # visualize coherence (coh) or icn (icn) matrices/graphs
    mat_type_ = 'coh'
    # type of matrix type (signal trace) to visualize (whole, preictal, or ictal)
    trace_type_ = 'preictal'
    # # band to inspect
    # band = 'alpha'
    # inspect_single_band(patient, band)

    """
    Inspect all bands (multiple plots will be generated)
    Example of for loop:
    """
    # bands = ['alpha', 'beta', 'delta', 'gamma', 'gammaHi', 'theta']
    bands = ['alpha']  # , 'beta', 'delta', 'gamma', 'gammaHi', 'theta']
    for band in bands:
        filename_ = sbj + '/' + band + '.npy'
        title_ = '{} for {} - {} band - {} signal'.format(mat_type_, sbj, band, trace_type_)
        inspect_single_band(sbj, filename_, mat_type=mat_type_, trace_type=trace_type_, channels=ch_names, title=title_)
