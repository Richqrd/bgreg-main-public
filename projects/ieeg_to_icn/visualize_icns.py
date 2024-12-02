from bgreg.utils.dataproc.bidsproc import get_channel_names
from projects.ieeg_to_icn.pipeline import visualize_icn


# # patient to inspect
subject = 'pt6'
# Get all channels from the signal
ch_names = get_channel_names(subject)
# visualize coherence (coh) or icn (icn) matrices/graphs
mat_type_ = 'icn'
# type of matrix type (signal trace) to visualize (whole, preictal, or ictal)
trace_type_ = 'ictal'
# # band to inspect
band = 'alpha'
title_ = '{} for {} - {} band - {} signal'.format(mat_type_, subject, band, trace_type_)
filename = subject + '/' + band + '.npy'
# pt01_soz_cla = ['PD1', 'PD2', 'PD3', 'PD4', 'AD1', 'AD2', 'AD3', 'AD4', 'ATT1', 'ATT2']
pt6_soz_cla = ['LA1', 'LA2', 'LA3', 'LA4', 'LAH1', 'LAH2', 'LAH3', 'LAH4', 'LPH1', 'LPH2', 'LPH3', 'LPH4']
visualize_icn(subject, filename, trace_type=trace_type_,
              title=title_, soz=pt6_soz_cla, savefigs=False)

"""
Inspect all bands (multiple plots will be generated)
Example of for loop:
"""
# bands = ['alpha', 'beta', 'delta', 'gamma', 'gammaHi', 'theta']
# for band in bands:
#     filename_ = patient + '/' + band + '.npy'
#     title_ = '{} for {} - {} band - {} signal'.format(mat_type_, patient, band, trace_type_)
#     visualize_icn(patient, filename_, mat_type=mat_type_, trace_type=trace_type_,
#                   channels=ch_names, title=title_, soz=pt01_soz_cla)
