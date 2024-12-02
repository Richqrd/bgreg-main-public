from bgreg.data.io import get_all_raw_signals_bids


subject = "pt6"
runs = get_all_raw_signals_bids(subject)

for run in runs:
    print("In run {} of patient {} there are {} channels.".format(run[1], subject, len(run[0].info["ch_names"])))
