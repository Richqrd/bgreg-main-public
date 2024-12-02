import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from bgreg.utils.visual import utilities as utils
from bgreg.utils.dirtree import save_icn_to_image_file


def plot_graphs(subject, graph_dict, regions_dict=None, regions_dict_inv=None,
                title='', nodefill=None, colormap=cm.tab20, savegraph=False,
                trace_type=None):
    """
    Plots networkx graphs as a chord diagram

    :param subject: (str) patient id
    :param graph_dict: graphDict is a dictionary of networkx graphs,
        keys correspond to graph index (0, 1, 2, 3 ...).
        Graph is either instance of networkx class or a dictionary with
        keys 'pos' and 'neg' if graph contains both increasing and decreasing
        connections (eg for ICNs)
    :param regions_dict: (list of str) list of channel names
    :param title: (str) title of figure and file
    :param nodefill: (list of bool) operators to color nodes or not
    :param colormap: (matplotlib.cm) colormap to use for CircosPlot
    :param savegraph: (bool) operator to save graphs to file, default False
    :param trace_type: (str) signal trace type, only used if savemat=True
    :return:
    """
    # Get figure object and axis arrays to fit graphs from graph_dict
    fig, ax, ax1D = utils.custom_subplots(len(graph_dict))
    for idx, (graph_key, graph) in enumerate(graph_dict.items()):
        if isinstance(graph, dict) and 'pos' in graph:
            # for kk, key in enumerate(['pos', 'neg']):
            # Connections with increased connectivity: solid line
            linestyle = '-'
            # Connections with decreased connectivity: dashed line
            # if key == 'neg':
            #     linestyle='--'

            circ = CircosPlot(graph['pos'].nodes(), graph['pos'].edges(),
                              nodecolor=get_node_colors(regions_dict, colormap=colormap),
                              regions_dict=regions_dict_inv, nodefill=nodefill, linestyle=linestyle,
                              fig=fig, ax=ax1D[idx])
            circ.draw()

            ax1D[idx].set_title('ICN {0}'.format(str(graph_key + 1)), fontsize=16)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1D[-1].legend(lines[0:len(regions_dict.keys())], regions_dict.keys(), ncol=3, loc='center right')

    fig.suptitle(title, fontsize=20)
    if savegraph:
        save_icn_to_image_file(subject, fig, title, trace_type)
    else:
        plt.show()


def graph_cmap(nColors, colormap=cm.tab20):
    cmap_all = colormap(np.linspace(0.1, 0.9, nColors))
    # # Shuffle so similar colors are not right next to each other:
    cmap = np.vstack((cmap_all[::2], cmap_all[1::2]))

    return cmap


def get_node_colors(channels, colormap=cm.tab20):
    """
    Set a color to each electrode.
    Electrodes are colored by region, that is, there is one color per region.

    :param channels: (list of str) list of channel names
    :param colormap: (matplotlib.cm) range of colors to select from
    """

    nColors = len(channels)
    colors = graph_cmap(nColors, colormap=colormap)
    nodecolor = []

    for ch_count, (ch_label, ch_list) in enumerate(channels.items()):
        for _ch in ch_list:
            nodecolor.append(colors[ch_count])

    return nodecolor


def get_edge_colors(edges, nodeLabels, colormap=cm.plasma):
    # FIXME: do we need this function?
    """
    Define colors for edges in graph

    - nodeLabels: dictionary of {label: [nodes]} (output of get_node_labels)
    - edges: list of edge tuples
    """

    nColors = len(nodeLabels)
    colors = graph_cmap(nColors, colormap=colormap)

    edgecolor1 = []
    edgecolor2 = []

    for edge in edges:
        for ee, elecs in enumerate(nodeLabels.values()):
            if edge[0] in elecs:
                edgecolor1.append(colors[ee])
            if edge[1] in elecs:
                edgecolor2.append(colors[ee])

    return edgecolor1, edgecolor2


class CircosPlot(object):
    def __init__(self, nodes, edges, radius=12, noderadius=0.07,
                 nodecolor=None, edgecolor=None, regions_dict=None,
                 linewidth=1, linestyle='-',
                 nodefill=None, alpha=0.6, ax=None, fig=None):

        self.nodes = nodes  # list of nodes
        self.edges = edges  # list of edge tuples

        # Make sure props are dictionaries if passed in
        # Node props
        self.nodeprops = {}
        # Edge props
        self.edgeprops = {}

        # Set colors. Priority: nodecolor > nodeprops > default
        # Node color
        self.nodecolor = nodecolor

        # Edge color
        if edgecolor is not None:
            self.edgecolor = edgecolor
        else:
            self.edgecolor = 'black'

        self.regions_dict = regions_dict

        self.lw = linewidth
        self.ls = linestyle

        if nodefill is not None:
            self.nodefill = nodefill

        self.alpha = alpha
        self.radius = radius
        self.fig = fig
        self.ax = ax
        self.node_radius = self.radius * noderadius
        self.ax.set_xlim(-radius * (1 + noderadius), radius * (1 + noderadius))
        self.ax.set_ylim(-radius * (1 + noderadius), radius * (1 + noderadius))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        for k in self.ax.spines.keys():
            self.ax.spines[k].set_visible(False)

    def draw(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        """
        Draws nodes onto the canvas with colours.
        """
        curr_color = np.array([0, 0, 0, 0])
        for nn, (node, color) in enumerate(zip(self.nodes, self.nodecolor)):
            x, y = get_cartesian(self.radius, self.node_theta(node))
            if not self.nodefill[nn]:
                self.nodeprops['facecolor'] = 'w'
            else:
                self.nodeprops['facecolor'] = color
            self.nodeprops['edgecolor'] = color

            # if self.regions_dict[nn] != curr_reg:
            if not (curr_color == color).all():
                self.nodeprops['label'] = self.regions_dict[nn]
            else:
                self.nodeprops['label'] = ''
            curr_color = color.copy()

            node_patch = patches.Ellipse((x, y), self.node_radius, self.node_radius,
                                         lw=0.8, **self.nodeprops)
            self.ax.add_patch(node_patch)

    def node_theta(self, node):
        """
        Maps node to Angle.
        """
        if isinstance(self.nodes, list):
            i = self.nodes.index(node)
        else:
            i = [ll for ll in self.nodes][node]
        theta = i * 2 * np.pi / len(self.nodes)

        return theta

    def add_edges(self):
        """
        Draws edges to screen.
        """
        for ii, (start, end) in enumerate(self.edges):
            start_theta = self.node_theta(start)
            end_theta = self.node_theta(end)
            verts = [get_cartesian(self.radius, start_theta),
                     (0, 0),
                     get_cartesian(self.radius, end_theta)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            path = Path(verts, codes)
            self.edgeprops['facecolor'] = 'none'

            if isinstance(self.edgecolor, list) and len(self.edgecolor) > 1:
                self.edgeprops['edgecolor'] = self.edgecolor[ii]
            else:
                self.edgeprops['edgecolor'] = self.edgecolor

            patch = patches.PathPatch(path, lw=self.lw, ls=self.ls, alpha=self.alpha, **self.edgeprops)
            self.ax.add_patch(patch)


def get_cartesian(r, theta):
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    return x, y
