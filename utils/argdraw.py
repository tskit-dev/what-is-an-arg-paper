"""
Some networkx utilities for drawing an ARG
"""
import collections
import itertools

import msprime
import networkx as nx

from constants import NODE_IS_RECOMB, NODE_IS_ALWAYS_UNARY, NODE_IS_SOMETIMES_UNARY
def nx_ts_colour_map(G, flags_attribute_name="flags"):
    """
    The default colour map for nodes in a graph that represents a tree sequence
    Assumes each node has a "flags" atteibute from the tree sequence
    """
    colour_map = []
    for nd in G.nodes(data=True):
        colour = "lightgreen"
        if nd[1][flags_attribute_name] & NODE_IS_RECOMB:
            colour = "red"
        elif nd[1][flags_attribute_name] & msprime.NODE_IS_CA_EVENT:
            assert nd[1][flags_attribute_name] & NODE_IS_ALWAYS_UNARY
            colour = "cyan"
        elif nd[1][flags_attribute_name] & NODE_IS_SOMETIMES_UNARY:
            colour = "lightseagreen"

        colour_map.append(colour)
    return colour_map

def nx_draw_with_curved_multi_edges(G, pos, colour_map, curve_scale=1, node_labels=None):
    """
    networkx matlib plots show all edges as straight lines by default.
    This function curves the edges if there is more than one edge with identical parent/child values
    At the moment curve_scale is a hack to adjust the fact that we don't know the measurement scale on
    the x or the y axis.
    """
    nx.draw_networkx_nodes(G, pos, node_color=colour_map)
    nx.draw_networkx_labels(G, pos, labels=node_labels)


    for (start, end), edge_iter in itertools.groupby(G.edges()):
        # identical node/neighbour pairs placed together according to docs
        edges = list(edge_iter)
        dist = ((pos[start][0] - pos[end][0]) ** 2 + (pos[start][1] - pos[end][1]) ** 2) ** 0.5
        # FIXME - dist won't be accurate if X and Y axes are on different scales
        curvature = curve_scale/dist
        # FIXME - calculate curve_scale from the plot dimensions, rather than requiring it
        max_edge_index = len(edges) - 1
        for i, e in enumerate(edges):
            curve_prop = (2 * i / max_edge_index - 1) if max_edge_index > 0 else 0
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[e],
                connectionstyle=f'arc3, rad = {curve_prop * curvature}')

def nx_get_dot_pos(G, time_attribute_name="time"):
    """
    Layout using graphviz's "dot" algorithm and return a dict of positions in the
    format required by networkx. We assume that the nodes have a "time" attribute
    """
    nodes_at_time = collections.defaultdict(list)
    for nd in G.nodes(data=True):
        nodes_at_time[nd[1][time_attribute_name]].append(nd[0])

    A = nx.nx_agraph.to_agraph(G)
    # First cluster all nodes at the same times (probably mostly samples)
    for t, nodes in nodes_at_time.items():
        if len(nodes) > 1:
            A.add_subgraph(nodes, level="same", name=f"cluster_t{t}")
    
    # We could also cluster nodes from a single individual together here
    A.layout(prog="dot")
    return {n: [float(x) for x in A.get_node(n).attr["pos"].split(",")] for n in G.nodes()}

