"""
Viz routines for args.
"""
import collections

import tskit
import networkx as nx
import numpy as np


def draw(ts, ax):
    G = convert_nx(ts)
    labels = {j: f"{j}" for j in range(ts.num_nodes)}
    pos = nx_get_dot_pos(G)
    nx.draw(
        G,
        pos,
        # node_color=colour_map,
        node_shape="o",
        node_size=200,
        font_size=9,
        labels=labels,
        ax=ax,
    )


def convert_nx(ts):
    """
    Returns the specified tree sequence as an networkx graph.
    """
    G = nx.Graph()
    edges = collections.defaultdict(list)
    for edge in ts.edges():
        edges[(edge.child, edge.parent)].append((edge.left, edge.right))
    for node in ts.nodes():
        G.add_node(node.id, time=node.time, flags=node.flags)
    for edge, intervals in edges.items():
        G.add_edge(*edge, intervals=intervals)
    return G


def nx_get_dot_pos(G, add_invisibles=False):
    """
    Layout using graphviz's "dot" algorithm and return a dict of positions in the
    format required by networkx. We assume that the nodes have a "time" attribute
    """
    nodes_at_time = collections.defaultdict(list)
    for nd in G.nodes(data=True):
        nodes_at_time[nd[1]["time"]].append(nd[0])

    A = nx.nx_agraph.to_agraph(G)
    # First cluster all nodes at the same times (probably mostly samples)
    for t, nodes in nodes_at_time.items():
        if len(nodes) > 1:
            A.add_subgraph(nodes, level="same", name=f"cluster_t{t}")
    if add_invisibles:
        # Must add in "invisible" edges between different levels, to stop them clustering
        # on the same level
        prev_node = None
        for t, nodes in nodes_at_time.items():
            if prev_node is not None:
                A.add_edge(prev_node, nodes[0], style="invis")
            prev_node = nodes[0]
        # We could also cluster nodes from a single individual together here
    A.layout(prog="dot")
    # multiply the y coord by -1 to get y axis going in the direction we want.
    xy_dir = np.array([1, -1])
    return {
        n: np.fromstring(A.get_node(n).attr["pos"], sep=",") * xy_dir for n in G.nodes()
    }

