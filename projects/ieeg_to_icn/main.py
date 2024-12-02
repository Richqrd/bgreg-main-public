from bgreg.utils.dirtree import ieeg_to_icn, coh_icn
from bgreg.data.io import get_channel_names
from projects.ieeg_to_icn.pipeline import *


def run():
    # initialize data patient_dir tree
    ieeg_to_icn()

    subject = 'pt6'

    # initialize patient_dir tree for coherence and ICN matrices
    coh_icn(subject)

    # run pipeline
    run_pipeline(subject, savefile=True)

    # get all channels from the signal
    ch_names = get_channel_names(subject)

    # visualize coherence (coh) or icn (icn) matrices/graphs
    mat_type = 'coh'

    # type of matrix type (signal trace) to visualize (preictal, ictal, or postictal)
    trace_type = 'preictal'

    # list of frequency bands to visualize
    bands = ['alpha', 'theta', 'delta', 'beta', 'gamma', 'gammaHi']

    # clinically annotated (cla) epileptogenic zone for pt6
    pt6_soz_cla = ['LA1', 'LA2', 'LA3', 'LA4', 'LAH1', 'LAH2', 'LAH3', 'LAH4', 'LPH1', 'LPH2', 'LPH3', 'LPH4']
    # pt01_soz_cla = ['PD1', 'PD2', 'PD3', 'PD4', 'AD1', 'AD2', 'AD3', 'AD4', 'ATT1', 'ATT2']

    # # plot ICN chord diagram and binary matrices
    for band in bands:
        title = '{} for {} - {} band - {} signal'.format(mat_type, subject, band, mat_type)
        filename = os.path.join(subject, band + '.npy')
        visualize_icn(subject, filename, trace_type=trace_type, title=title, soz=pt6_soz_cla, savefigs=True)

    # To visualize the matrix objects uncomment the code below
    # # plot coherence matrices or ICNs for all bands with visualize_matrix
    # for band in bands:
    #     # Title for each plot,
    #     # default format:
    #     #   {mat_type} for {patient} - {freq-band} band - {preictal/ictal/preictal} signal
    #     # example:
    #     #   icn for pt01 - delta band - preictal signal
    #     title = '{} for {} - {} band - {} signal'.format(mat_type, subject, band, trace_type)
    #     filename = os.path.join(subject, band + '.npy')
    #     visualize_matrix(subject,
    #                      filename,
    #                      mat_type=mat_type,
    #                      trace_type=trace_type,
    #                      channels=ch_names,
    #                      title=title,
    #                      savemat=False)
run()