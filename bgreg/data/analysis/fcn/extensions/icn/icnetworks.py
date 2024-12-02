"""
Class for data analysis of Intrinsic Connectivity Networks:
Extract and save ICNs, plot ICNs

"""

import numpy as np
import re
import networkx as nx
from bgreg.data.io import get_regions_dict


class IcnGraph(object):
    """
    Graphical representation of intrinsic connectivity networks
    """

    def __init__(self, subject, icn_matrices, regions, quantileThresh=98, weightThresh=0.3):
        # list of str (region acronym, i.e., G, ATT, PLT)
        self.regions = regions

        # dictionary of lists, hashing region acronym (str) to all-electrode id
        # i.e., {'G': [0, 1, 2, 3]}
        self.regions_dict = get_regions_dict(subject)

        # ndarray with ICN networks
        self.icn_matrices = icn_matrices

        # initialize the icnBin that will be used for establishing edges
        # for the common regions and the binarized representations
        self.icnBin = np.zeros_like(self.icn_matrices)

        # dictionary to hash graph id (int) to graphs
        # i.e., {0: {'pos': graphObject, 'neg': graphObject}}
        # pos and neg graphs correspond to increased and decreased
        # connectivity graphs, respectively
        # FIXME: What is high and low connectivity in this case?
        self.graphs = {}

        # create graphs upon initialization
        self.constructGraphs(quantileThresh=quantileThresh, weightThresh=weightThresh)

    def constructGraphs(self, quantileThresh=98, weightThresh=0.3):
        """
        Creates dictionary of graphs corresponding to each ICN: {ICNint: {'pos': networkx graph, 'neg': networkx graph}}

        Each graphical representation of each ICN is overlay of 'pos' graph (increased connectivity) and
            'neg' graph (decreased connectivity)

        - ICNint key: integer corresponding to ICN number (indexed from 0)
        - 'pos' graph: edges that correspond to increase in connectivity in given ICN
        - 'neg' graph: edges that correspond to decrease in connectivity in given ICN
        - quantileThresh: threshold for constructing graph edges from upper quantile of ICN loading
            (ie if quantileThresh = 98, ICN loading > 98th percentile with count as edge)
        """

        nodes = range(self.icn_matrices.shape[0])
        edges = self.getEdges(quantile=quantileThresh)
        # TODO: When are the weigthed edges used? Can we reuse this for creating different matrices/graphs?
        # edges = self.getWeightedEdges(weightThresh=weightThresh)  # This was used with Ave graphs

        for kk in range(self.icn_matrices.shape[-1]):

            self.graphs.setdefault(kk, {})

            for pp, key in enumerate(['pos', 'neg']):
                G = nx.Graph()
                G.add_nodes_from(nodes)
                if key == 'pos':
                    G.add_edges_from(edges[kk][0])
                    # G.add_weighted_edges_from(edges[kk][0])  # This was used with Ave graphs
                else:
                    G.add_edges_from(edges[kk][1])
                    # G.add_weighted_edges_from(edges[kk][1])  # This was used with Ave graphs

                self.graphs[kk][key] = G

    def unfill_soz(self, soz):
        nodefill = [True] * self.icn_matrices.shape[0]  # [True] * node number
        soz_node_to_indices = {}
        for node in soz:
            # split soz node in location name and index (string and int)
            soz_loc = re.split('(\d+)', node)
            soz_node_to_indices.setdefault(soz_loc[0], [])
            soz_node_to_indices[soz_loc[0]].append(soz_loc[1])

        node_unfill = []
        for node, idx_list in soz_node_to_indices.items():
            real_node_idx_list = self.regions_dict[node]
            for _id in idx_list:
                node_unfill.append(real_node_idx_list[int(_id) - 1])

        for _id in node_unfill:
            nodefill[_id] = False
        return nodefill

    def getEdges(self, quantile=98):
        # TODO: What are the pos and neg? What do they indicate?
        """
        Get edges for all ICNs (edges correspond to ICN loading greater than or less than 'quantile' threshold)

        Returns dictionary of edges:
            - key: ICN number
            - values: 2-element list of edges. First element corresponds to >quantile and second corresponds
                to <100-quantile (ie. connectivity between regions that increase vs. decrease in ICN)
        """
        edges = {}

        for icn_key, icn_matrix in enumerate(np.rollaxis(self.icn_matrices, 2)):
            thresh = np.percentile(abs(icn_matrix), quantile)

            # ICN loading can be + or -. Define threshold based on which abs(percentile) is greatest
            neg = np.percentile(icn_matrix, 100 - quantile)
            pos = np.percentile(icn_matrix, quantile)

            if abs(neg) > pos:
                icnSigned = -icn_matrix
            else:
                icnSigned = icn_matrix

            edges[icn_key] = []
            for tt in (thresh, -thresh):
                # Connections with increased connectivity
                if tt > 0:
                    edge = [ii for ii in zip(np.where(icnSigned > tt)[0], np.where(icnSigned > tt)[1])]
                    for mm, nn in zip(np.where(icnSigned > tt)[0], np.where(icnSigned > tt)[1]):
                        self.icnBin[mm, nn, icn_key] = 1
                # Connections with decreased connectivity
                else:
                    edge = [ii for ii in zip(np.where(icnSigned < tt)[0], np.where(icnSigned < tt)[1])]
                    for mm, nn in zip(np.where(icnSigned < tt)[0], np.where(icnSigned < tt)[1]):
                        self.icnBin[mm, nn, icn_key] = -1
                edges[icn_key].append(edge)
        return edges

    def getWeightedEdges(self, weightThresh=0.3):
        # FIXME: This is not being used, but, can we improve it and reuse it?
        edges = {}

        nodes = []  # self.getNodes()

        edgePairs = []  # utils.get_pairs(range(len(nodes)), range(len(nodes)), includeSame=False, unique=True)

        for kk, icn in enumerate(np.rollaxis(self.icn_matrices, 2)):
            edges[kk] = []
            edges[kk].append([])
            edges[kk].append([])

            if icn.max() >= abs(icn.min()):
                normFac = icn.max()
            else:
                normFac = icn.min()

            icnNorm = icn / normFac

            for pair in edgePairs:
                weight = icnNorm[pair[0], pair[1]]

                if weight > weightThresh:
                    edges[kk][0].append((pair[0], pair[1], weight))

                elif weight < -weightThresh:
                    edges[kk][1].append((pair[0], pair[1], weight))

        return edges
