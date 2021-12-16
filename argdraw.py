"""
Some networkx utilities for drawing an ARG
"""
import networkx as nx
import itertools

def draw_with_curved_multi_edges(G, pos, colour_map, curve_scale=1):
    """
    networkx matlib plots show all edges as straight lines by default.
    This function curves the edges if there is more than one edge with identical parent/child values
    At the moment curve_scale is a hack to adjust the fact that we don't know the measurement scale on
    the x or the y axis.
    """
    nx.draw_networkx_nodes(G, pos, node_color=colour_map)
    nx.draw_networkx_labels(G, pos)


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