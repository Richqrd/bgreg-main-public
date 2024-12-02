import os
import pickle
import numpy as np
from pathlib import Path
from bgreg.native.datapaths import graph_representation_elements_dir
from SOZ import get_SOZ
from metrics import apk


subject = "ummc009"
tests = ["binary-normal", "binary-same-adjacency", "multi-normal", "multi-same-adjacency"]

gre_dir = Path(graph_representation_elements_dir, subject)
ch_names = pickle.load(open(Path(gre_dir, "ch_names.pickle"), 'rb'))

for test_label in tests:
    test_dir = Path(os.getcwd(), "bgreg", "projects", "gnn_to_soz", "logs", test_label, subject)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%", "Processing", test_dir, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for results_dir in os.listdir(test_dir):
        if os.path.isdir(Path(test_dir, results_dir)):
            print("Processing test: {}".format(results_dir))
            attns = pickle.load(open(Path(test_dir, results_dir, "attention_scores.pickle"), 'rb'))
            scores = [(attn - attn.min()) / (attn.max() - attn.min()) for attn in attns]
            scores = np.array(scores)

            # get mean and standard deviation for attention scores
            score = scores.mean(0)
            xerr = scores.std(0)
            sorter = np.argsort(score)
            score_sort = score[sorter]
            xerr_sort = xerr[sorter]
            chan_names_sort = [ch_names[i] for i in sorter]

            target_soz = get_SOZ()[subject]

            print("APK: {}".format(apk(target_soz, chan_names_sort, k=5)))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
