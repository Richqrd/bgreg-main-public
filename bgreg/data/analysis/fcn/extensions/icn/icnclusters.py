# @kirkbyl, 21 Nov 2016
"""
Analysis class:
    - clustering to find common ICNs: 1) across fq bands 2) across patients
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import collections
import networkx as nx
from sklearn.cluster import DBSCAN
import pygraphviz as pgv
import brewer2mpl as brewcm

from bgreg.utils.opensource.ecogmood import plotchord
from bgreg.utils.visual import plot_matrices


class IcnCluster(object):
    def __init__(self, subjects, icn_dict, graph_dict, node_labels,
                 graphType='common'):
        """
        - pID: patient ID, string or Patient class
        - dataPath: path to saved ICN data
        - bands: frequency band of corresponding Intrinsic Network

        - graphType can be 'full', 'ave', or 'common':
            - full: all electrodes
            - ave: average across electrodes in given brain region
            - common: 'common' network (across patients) defined by these regions:
            ['aSTC', 'mSTC', 'pSTC', 'AM', 'vHPC', 'mHPC', 'dHPC', 'aOFC', 'pOFC', 'iCIN', 'sCIN', 'dCIN', 'INS']
        """

        self.patients = subjects
        self.ICNetworks = icn_dict
        self.ICGraphs = graph_dict
        self.nodeLabels = node_labels
        self.graphType = graphType

        self.clusters = collections.OrderedDict()
        self.CC = collections.OrderedDict()
        self.aCC = collections.OrderedDict()
        self.allICNs_2D = collections.OrderedDict()
        self.allICNs_3D = collections.OrderedDict()
        self.icnLabels = collections.OrderedDict()
        self.allGraphs = collections.OrderedDict()

        self.concatICNs()

    def concatICNs(self):

        if self.graphType == 'common':
            self.icnLabels['all'] = []
            self.allGraphs['all'] = {}

        counter_a = 0

        for pp, pID in enumerate(self.patients):

            self.icnLabels[pID] = []
            self.allGraphs[pID] = {}

            counter_p = 0

            for bb, band in enumerate(self.ICNetworks[pID].keys()):

                icn = self.ICNetworks[pID][band].commonICN
                flatICN = flattenMatrix(icn)
                gph = self.ICGraphs[pID][band].graphs

                if self.graphType == 'common':

                    if pp == 0 and bb == 0:
                        self.allICNs_2D['all'] = flatICN
                        self.allICNs_3D['all'] = icn
                    else:
                        self.allICNs_2D['all'] = np.vstack((self.allICNs_2D['all'], flatICN))
                        self.allICNs_3D['all'] = np.dstack((self.allICNs_3D['all'], icn))

                    for nn in range(icn.shape[-1]):
                        self.icnLabels['all'].append((pID, band[0]))

                    for gg in gph.values():
                        self.allGraphs['all'][counter_a] = gg
                        counter_a += 1

                if bb == 0:
                    self.allICNs_2D[pID] = flatICN
                    self.allICNs_3D[pID] = icn
                else:
                    self.allICNs_2D[pID] = np.vstack((self.allICNs_2D[pID], flatICN))
                    self.allICNs_3D[pID] = np.dstack((self.allICNs_3D[pID], icn))

                for nn in range(icn.shape[-1]):
                    self.icnLabels[pID].append((pID, band[0]))

                for gg in gph.values():
                    self.allGraphs[pID][counter_p] = gg
                    counter_p += 1

    def corrcoeff(self, showPlots=False):
        for key in self.allICNs_2D.keys():
            # Compute Pearson product-moment correlation coefficients
            self.CC[key] = np.corrcoef(self.allICNs_2D[key])
            # Absolute value for Pearson product-moment correlation coefficients
            self.aCC[key] = abs(self.CC[key])
            if showPlots:
                plt.figure()
                plt.imshow(self.CC[key], interpolation='None')
                plt.colorbar()
                plt.title(key)
                plt.figure()
                plt.imshow(self.aCC[key], interpolation='None')
                plt.colorbar()
                plt.title(key)

    def plotSimilarityMatrix(self, minThresh=None):

        if not hasattr(self, 'aCC'):
            self.corrcoeff()

        for key in self.aCC.keys():
            maxCCrow = []

            for rr, row in enumerate(self.aCC[key][:-1]):
                maxCC = np.max(row[(rr + 1):])
                maxCCrow.append(maxCC)

            sortinds = np.argsort(maxCCrow)[::-1]
            allinds = range(self.aCC[key].shape[0])

            inds = np.hstack((sortinds, (set(allinds) - set(sortinds)).pop()))

            plotInds = [inds[0]]

            sortedArray = np.zeros_like(self.aCC[key], dtype=int)

            for ii, ind in enumerate(inds):
                row = self.aCC[key][ind]
                args = np.argsort(row)[::-1]

                sortedArray[ii] = args

            for ss, line1 in enumerate(sortedArray[:-1]):

                line2 = sortedArray[ss + 1]
                thresh = self.aCC[key][line2[0], line2[1]]

                for ll, tt in enumerate(line1):
                    if self.aCC[key][line1[0], tt] >= thresh and tt not in plotInds:
                        plotInds.append(int(tt))

            if len(plotInds) < sortedArray.shape[0]:
                plotInds.append((set(allinds) - set(plotInds)).pop())

            sorted_aCC = np.zeros_like(self.aCC[key])
            for ii, ind in enumerate(plotInds):
                sorted_aCC[ii] = self.aCC[key][plotInds[ii], plotInds]

            self.findClusters()
            nClusters = len(np.unique(self.clusters[key]))

            plotLabels = [pp + 1 for pp in plotInds]

            if minThresh is None:
                minThresh = 0

            if key == 'all':
                plt.subplots(figsize=(14, 5))
                for pp, plotMat in enumerate([self.aCC[key], sorted_aCC]):
                    plt.subplot(1, 2, pp + 1)

                    newPlot = np.copy(plotMat)
                    newPlot[np.where(plotMat < minThresh)] = 0

                    plt.imshow(newPlot, interpolation='None', aspect='auto')
                    if pp == 1:
                        plt.yticks(np.arange(len(plotInds)), plotLabels)
                        plt.xticks(np.arange(len(plotInds)), plotLabels)
                    plt.colorbar()
                plt.suptitle('{0} combined: {1} clusters ({2} ICNs total)'.format(key.capitalize(),
                                                                                  str(nClusters),
                                                                                  len(self.clusters[key])))
                plt.show()

    def findClusters(self, eps=0.5):
        for key in self.aCC.keys():
            self.clusters[key] = DBSCAN(eps=eps, min_samples=1).fit_predict(self.aCC[key])

    def plotClusters(self, showGraph=True, showMatrix=True):
        """
        Plot similar ICNs (ie those in same cluster) either as matrix or as graph
        """
        for key in self.allICNs_3D.keys():
            unique, counts = np.unique(self.clusters[key], return_counts=True)

            for uu, cc in zip(unique, counts):
                if cc >= 2:
                    subset = np.where(self.clusters[key] == uu)[0]

                    title = 'Patient: {0}, cluster {1}: {2}'.format(key,
                                                                    uu,
                                                                    np.array(self.icnLabels[key])[subset])

                    if showMatrix:
                        plot_matrices.plot_matrices(self.allICNs_3D[key], title=title)

                    if showGraph:
                        plotchord.plot_graphs(self.allGraphs[key], title=title)
                        #  nodeLabels=self.nodeLabels[key])  # FIXME: Find a way to get these labels


class TopographicalMap(object):

    def __init__(self, pIDs, edgeThresh=0.55, shuffle=False):

        self.patientIDs = pIDs
        self.Networks = IcnCluster(pIDs)

        self.nodeLabels = self.Networks.nodeLabels[pIDs[0]]

        self.similarityMatrix = self.getSimilarityMatrix()
        self.assignICNs()

        self.createGraph(edgeThresh=edgeThresh, shuffle=shuffle)

    def getSimilarityMatrix(self, showPlots=True, minThresh=None):

        self.Networks.corrcoeff()
        if showPlots:
            self.Networks.plotSimilarityMatrix(minThresh=minThresh)
        return self.Networks.aCC

    def assignICNs(self):
        """
        Assign each graph to a patient and to a fq band
        """

        self.patInds = collections.OrderedDict()
        self.bandInds = collections.OrderedDict()
        self.patICNmap = collections.OrderedDict()

        counter1 = 0
        counter2 = 0

        for pp, pat in enumerate(self.patientIDs):

            nGphs = len(self.Networks.allGraphs[pat])
            self.patInds[pat] = range(counter1, counter1 + nGphs)
            self.patICNmap[pat] = []

            for ii, ind in enumerate(self.patInds[pat]):
                self.patICNmap[pat].append((ii, ind))

            counter1 = counter1 + nGphs

            for band in self.Networks.bands:
                if pp == 0:
                    self.bandInds[band] = []

                nNets = self.Networks.ICNetworks[pat][band].shape[-1]
                self.bandInds[band].extend(range(counter2, counter2 + nNets))

                counter2 = counter2 + nNets

    def createGraph(self, edgeThresh=0.55, patientKey='all', excludeICNs=None, shuffle=False):

        if shuffle:
            ntwrks_all = np.ones_like(self.Networks.aCC[patientKey])

            valsToShuffle = np.copy(self.Networks.aCC[patientKey][np.triu(
                self.Networks.aCC[patientKey], k=1) > 0])
            np.random.shuffle(valsToShuffle)

            # rebuild along diagonal
            inds_up = np.triu_indices(ntwrks_all.shape[0], k=1)
            inds_low = np.tril_indices(ntwrks_all.shape[0], k=-1)

            ntwrks_all[inds_up] = valsToShuffle
            ntwrks_all[inds_low[1], inds_low[0]] = valsToShuffle

        else:
            ntwrks_all = self.Networks.aCC[patientKey]

        grphs_all = self.Networks.allGraphs[patientKey]
        icns_all = self.Networks.allICNs_3D[patientKey]

        if excludeICNs is not None:

            goodNetworks = [gg for gg in range(
                ntwrks_all.shape[0]) if gg not in excludeICNs]
            ntwrks = ntwrks_all[np.ix_(goodNetworks, goodNetworks)]

            # reindex without excluded networks
            patInds_new = collections.OrderedDict()
            counter = 0
            for patID in self.patInds.keys():
                newInds = [pp for pp in self.patInds[patID]
                           if pp not in excludeICNs]
                patInds_new[patID] = range(counter, counter + len(newInds))
                counter = counter + len(newInds)

            del self.patInds
            self.patInds = patInds_new
            self.gphs = {}
            self.icns = np.zeros(
                (icns_all.shape[0], icns_all.shape[0], len(goodNetworks)))
            for gg, goodNetwork in enumerate(goodNetworks):
                self.grphs[gg] = grphs_all[goodNetwork]
                self.icns[:, :, gg] = icns_all[:, :, goodNetwork]

        else:
            ntwrks = ntwrks_all
            self.grphs = grphs_all
            self.icns = icns_all

        self.patientKey = patientKey

        self.G = nx.from_numpy_matrix(ntwrks)

        # Remove low correlated edges
        removeEdges = []

        for aa, bb in zip(np.where(ntwrks <= edgeThresh)[0], np.where(ntwrks <= edgeThresh)[1]):
            removeEdges.append((aa, bb))

        self.G.remove_edges_from(removeEdges)
        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))

        return self.G

    def removeNodesFrom(self, nodes):

        self.G.remove_nodes_from(nodes)

    def getSubgraphs(self, nThresh=1):

        self.subgraphs = list((self.G.subgraph(c) for c in
                               nx.connected_components(self.G)))  # list(nx.connected_component_subgraphs(self.G))

        # n gives the number of sub graphs

        if nThresh == 1:
            addS = ''
        else:
            addS = 's'

        # you can now loop through all nodes in each sub graph
        print('{0} subgraphs total'.format(n))
        print('Subgraphs with > {0} node{1}:'.format(nThresh, addS))
        for i in range(n):
            if len(self.subgraphs[i].nodes()) > nThresh:
                print('\tNodes in subgraph {0}: {1}'.format(i, self.subgraphs[i].nodes()))

        return self.subgraphs

    def getCoreNodes(self, G='default', k=6):

        if G == 'default':
            G = self.G

        # Set k = None so that main core is obtained instead of from hardcoded and data-based order of nodes
        Gcore = nx.k_core(G, k=k)

        return Gcore

    def getCliques(self, G='default', nclique=4):

        if G == 'default':
            G = self.G

        cliques = list(
            nx.algorithms.community.k_clique_communities(G, nclique))  # list(nx.k_clique_communities(G, nclique))

        return cliques

    def plotGraph(self, G='default', colorCode=None, figname='pgv', prog='neato', shape='circle', width=0.2,
                  addLabels=True, style='filled', fillcolor_rgb=[1, 1, 1], penwidth=0.01, nclique=4, ncore=6,
                  plotCliques=False, brewcmap='Set3', evenColorSpace=True):
        """
        colorCode: 'patients', 'fqbands', 'cliques', 'custom' (to choose specific patients, fqbands, cliques) or None
        brewcmap: brewer qualitative colormap, can be 'Paired' or 'Set3'
        """

        if G == 'default':
            G = self.G

        Gcore = self.getCoreNodes(G=G, k=ncore)

        fillcolor_hex = mcolors.rgb2hex(fillcolor_rgb)

        self.Gviz = pgv.AGraph()
        self.Gviz.add_nodes_from(Gcore.nodes(), shape=shape, width=width,
                                 label='', style=style, penwidth=penwidth, fillcolor=fillcolor_hex)
        self.Gviz.add_edges_from(Gcore.edges(), penwidth=penwidth)

        # Find cliques and make clique dictionary
        cliques = self.getCliques(G=Gcore, nclique=nclique)
        cliqueDict = {}
        if len(cliques) > 0:
            for cc, clique in enumerate(cliques):
                cliqueDict[cc + 1] = list(clique)

        # Color code nodes: color by patients, frequency bands or cliques
        if colorCode != None:
            if type(colorCode) == str:
                if colorCode == 'patients':
                    values = [list(vals) for vals in self.patInds.values()]
                    labels = self.patInds.keys()
                elif colorCode == 'fqbands':
                    values = [list(vals) for vals in self.bandInds.values()]
                    labels = self.bandInds.keys()
                elif colorCode == 'cliques':
                    cliques = self.getCliques(G=Gcore, nclique=nclique)
                    labels = range(1, len(cliques) + 1)

                    values = []
                    if len(cliques) > 0:
                        for clique in cliques:
                            values.append(list(clique))

                elif colorCode == 'custom':

                    if len(cliques) > 0:
                        clique_choice = input(
                            "Enter clique number, choose from {0} or 'all': ".format(cliqueDict.keys()))

                    pIDs = self.patInds.keys()
                    fqBands = self.bandInds.keys()

                    pID_choice = input(
                        "Enter patient ID, choose from {0} or 'all': ".format(pIDs))
                    fqBand_choice = input(
                        "Enter fq band, choose from {0} or 'all': ".format(fqBands))

                    # Choose ICNs for specific patient IDs:
                    if pID_choice != 'all':
                        pID_values = self.patInds[pID_choice]
                    elif pID_choice == 'all':
                        pID_values = []
                        for vals in iter(self.patInds.values()):
                            pID_values.extend(vals)
                    else:
                        print('Unknown input for patient ID')

                    # Choose ICNs for specific fq bands:
                    if fqBand_choice != 'all':
                        fqBand_values = self.bandInds[fqBand_choice]
                    elif fqBand_choice == 'all':
                        fqBand_values = []
                        for vals in self.bandInds.itervalues():
                            fqBand_values.extend(vals)
                    else:
                        print('Unknown input for fq bands')

                    # Choose ICNs for specific clique:
                    if clique_choice != 'all':
                        clique_values = cliqueDict[int(clique_choice)]
                    elif clique_choice == 'all':
                        clique_values = []
                        for vals in cliques:
                            clique_values.extend(vals)
                    else:
                        print('Unknown input for cliques')

                    # values correpond to intersection of all of the above:
                    values = list(set(pID_values) & set(
                        fqBand_values) & set(clique_values))
                    labels = None

            elif type(colorCode) == list:
                values = colorCode
                labels = None

            elif type(colorCode) == dict:

                if 'clique' in colorCode:
                    clique_choice = colorCode['clique']
                else:
                    clique_choice = 'all'

                if 'patient' in colorCode:
                    pID_choice = colorCode['patient']
                else:
                    pID_choice = 'all'

                if 'fqBand' in colorCode:
                    fqBand_choice = colorCode['fqBand']
                else:
                    fqBand_choice = 'all'

                # Choose ICNs for specific patient IDs:
                if pID_choice != 'all':
                    pID_values = self.patInds[pID_choice]
                elif pID_choice == 'all':
                    pID_values = []
                    for vals in self.patInds.itervalues():
                        pID_values.extend(vals)
                else:
                    print('Unknown input for patient ID')

                # Choose ICNs for specific fq bands:
                if fqBand_choice != 'all':
                    fqBand_values = self.bandInds[fqBand_choice]
                elif fqBand_choice == 'all':
                    fqBand_values = []
                    for vals in self.bandInds.itervalues():
                        fqBand_values.extend(vals)
                else:
                    print('Unknown input for fq bands')

                # Choose ICNs for specific clique:
                if clique_choice != 'all':
                    clique_values = cliqueDict[int(clique_choice)]
                elif clique_choice == 'all':
                    clique_values = []
                    for vals in cliques:
                        clique_values.extend(vals)
                else:
                    print('Unknown input for cliques')

                # values correpond to intersection of all of the above:
                values = list(set(pID_values) & set(
                    fqBand_values) & set(clique_values))
                labels = None

            if len(values) != 0:
                if not isinstance(values[0], list):
                    values = [values]
                    rgbcolors = [[1, 0, 0]]
                else:
                    try:
                        bmap = brewcm.get_map(
                            brewcmap, 'Qualitative', len(values))
                        rgbcolors = bmap.mpl_colors
                    except:

                        if hasattr(brewcmap, 'colors') and evenColorSpace != True:
                            if len(values) <= len(brewcmap.colors):
                                rgbcolors = brewcmap.colors[0:len(values)]
                            else:  # changed two below scenarios to [] isntead of () to call into str
                                rgbcolors = brewcmap(
                                    np.linspace(0, 1, len(values)))
                        else:
                            rgbcolors = brewcmap(
                                np.linspace(0, 1, len(values)))

                hexcolors = []
                for rgbcolor in rgbcolors:
                    hexcolors.append(mcolors.rgb2hex(rgbcolor))

                if labels is not None:
                    fig, ax = plt.subplots()
                    for ll, label in enumerate(labels):
                        ax.plot([ll], [0], color=hexcolors[ll],
                                marker='o', markersize=10, label=label)
                    ax.set_xticklabels('')
                    ax.set_yticklabels('')
                    plt.title('Legend for colorCode: ' + colorCode)
                    plt.legend()

                nodeLabels = []
                for vv, vals in enumerate(values):

                    for val in vals:

                        if val in Gcore.nodes():

                            if addLabels == True:
                                label = next(
                                    key for key, inds in self.patInds.items() if val in inds)
                                nodeLabels.append(label)
                            else:
                                label = ''

                            self.Gviz.add_node(
                                val, fillcolor=hexcolors[vv], shape=shape, xlabel=label, width=width, style=style,
                                penwidth=penwidth, fontsize=9)

                if colorCode == 'custom' or type(colorCode) == dict:
                    labels = nodeLabels
                    values = values[0]
            else:
                labels = None
                values = None

        else:
            labels = None
            values = None

        # Plot graph to PDF
        self.Gviz.layout(prog=prog)
        self.Gviz.draw('{0}.pdf'.format(figname))

        if plotCliques == True:
            self.plotCliqueICNs(G=Gcore, ncore=ncore, nclique=nclique)

        return {'labels': labels, 'ICN #': values}

    def plotCliqueICNs(self, G='default', ncore=6, nclique=4, plotAll=False, ave='mean', pc=25, colormap=cm.plasma):
        """
        Chord diagram of average ICNs of each clique
        Show all ICNs in given clique if plotAll==True

        ave: average type, can be 'mean', 'median' or 'percentile'
        pc: if ave=='percentile' then need to enter pc
        """

        if G == 'default':
            G = self.G

        Gcore = self.getCoreNodes(G=G, k=ncore)
        cliques = self.getCliques(G=Gcore, nclique=nclique)

        # plot all cliques:
        if plotAll == True:
            for cc, clique in enumerate(cliques):
                plotchord.plot_graphs(self.Networks.allGraphs['all'], colormap=colormap)
                plt.suptitle('Clique ' + str(cc))

        # PLot mean ICN for each clique
        edges = {}
        cliqueGrphs = {}
        for cc, clique in enumerate(cliques):
            if ave == 'mean':
                meanICN = np.mean(
                    abs(self.Networks.allICNs_3D['all'][:, :, list(clique)]), axis=-1)
            elif ave == 'median':
                meanICN = np.median(
                    abs(self.Networks.allICNs_3D['all'][:, :, list(clique)]), axis=-1)
            elif ave == 'percentile':
                meanICN = np.percentile(
                    abs(self.Networks.allICNs_3D['all'][:, :, list(clique)]), pc, axis=-1)
            else:
                print('Unknown average type')

            thresh = np.percentile(abs(meanICN), 96)
            nodes = range(meanICN.shape[0])
            edges[cc] = [ii for ii in zip(np.where(meanICN > thresh)[
                                              0], np.where(meanICN > thresh)[1])]

            cliqueGrphs[cc] = {}
            cliqueGrphs[cc]['pos'] = nx.Graph()
            cliqueGrphs[cc]['pos'].add_nodes_from(nodes)
            cliqueGrphs[cc]['pos'].add_edges_from(edges[cc])
            cliqueGrphs[cc]['neg'] = nx.Graph()
            cliqueGrphs[cc]['neg'].add_nodes_from(nodes)

        fig = plotchord.plot_graphs(cliqueGrphs, colormap=colormap)

        return fig

    def getGraphInfo(self, G='default', nclique=4, ncore=6, plotCliques=True):

        if G == 'default':
            G = self.G

        Gcore = self.getCoreNodes(G=G, k=ncore)

        cliques = self.getCliques(G=Gcore, nclique=nclique)

        numCliques = len(cliques)

        if plotCliques == True:
            self.plotCliqueICNs(G=Gcore, ncore=ncore, nclique=nclique)


def flattenMatrix(mat):
    nElements = int((mat.shape[0] * mat.shape[1]) / 2. + mat.shape[0] / 2.)

    flatMatrix = np.zeros((mat.shape[-1], nElements))

    for mm in range(mat.shape[-1]):
        matrix = mat[:, :, mm]

        triInds = np.triu_indices_from(matrix)
        flatMatrix[mm] = matrix[triInds]

    return flatMatrix
