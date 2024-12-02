from bgreg.data.analysis.graph.graph_representation_ethz import graph_representation_elements_ethz




if __name__ == '__main__':

    subjects = ["ID10"]

    subjects = [
        "ID1", "ID2", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9", "ID10",
        "ID11", "ID12", "ID13", "ID14", "ID15", "ID16"
    ]

    for subject in subjects:
        print(f'On subject {subject}' + '#' * 12)
        graph_representation_elements_ethz(subject, w_sz=1000)