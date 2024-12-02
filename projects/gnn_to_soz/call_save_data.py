import sys
from bgreg.data.analysis.graph.graph_representation import graph_representation_elements


if __name__ == '__main__':



    subjects = ["jh108",
                "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
                "pt2", "pt3", "pt6", "pt7", "pt8",
                "umf001",
                "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
                "ummc007", "ummc009"]


    # all_subjects = ["jh101", "jh103", "jh108"
    #             "pt01", "pt10", "pt12", "pt13", "pt14", "pt16",
    #             "pt2", "pt3", "pt6", "pt7", "pt8", "pt11",
    #             "umf001",
    #             "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
    #             "ummc007", "ummc009"]
    # missing jh108, pt11

    # subjects = ["pt3", "pt6", "pt7", "pt8",
    #                 "umf001",
    #                 "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006",
    #                 "ummc007", "ummc009"]

    subjects = ["pt12"]

    # the other variable for adj_matrix size is in graph_representations and is titled adj_window_size
    w_sz = 1000

    for subject in subjects:
        print(f'On subject {subject}' + '#'*12)
        graph_representation_elements(sub=subject, w_sz=w_sz, a_sz=20*1000)
