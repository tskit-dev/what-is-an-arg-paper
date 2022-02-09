"""
Viz routines for args.
"""
import collections
import itertools
import string

import tskit
import networkx as nx
import numpy as np
import pydot

def draw(ts, ax, use_ranked_times=None, tweak_x=None, arrows=False, pos=None):
    """
    Draw a graphical representation of a tree sequence. If a metadata key
    called "name" exists for the node, it is taken as
    a node label, otherwise the node ID will be used as a label instead.

    If use_ranked times is True, the y axis uses the time ranks, with the
    same times sharing a rank. If False, it uses the true (tree sequence)
    times. If None, times from the tree sequence are not used and the
    standard dot layout is used.
    
    tweak_x is a dict of {u: x_adjustment_percent} which allows
    the x position of node u to be hand-adjusted by adding or
    subtracting a percentage of the total x width of the plot
    
    If pos is passed in, it should be a dictionary mapping nodes to
    positions.
    """
    G = convert_nx(ts)
    labels = {}
    for nd in ts.nodes():
        try:
            labels[nd.id] = str(nd.metadata["name"])
        except (TypeError, KeyError):
            labels[nd.id] = str(nd.id)

    if pos is None:
        pos = nx_get_dot_pos(G)

    if use_ranked_times is not None:
        if use_ranked_times:
            _, inv = np.unique(ts.tables.nodes.time, return_inverse=True)
            ranked_times = (np.cumsum(np.bincount(inv)) - 1)[inv]
            for i, p in pos.items():
                p[1] = ranked_times[i]
        else:
            times = ts.tables.nodes.time
            for i, p in pos.items():
                p[1] = times[i]
    if tweak_x:
        plot_width = np.ptp([x for x, _ in pos.values()])
        for node, tweak_val in tweak_x.items():
            pos[node] = pos[node] + np.array([(tweak_val/100*plot_width), 0])
    # Draw just the nodes
    nx.draw(
        G,
        pos,
        # node_color=colour_map,
        node_shape="o",
        node_size=200,
        font_size=9,
        labels=labels,
        ax=ax,
        edgelist=[],
    )
    # Now add the edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        arrowstyle="-|>" if arrows else "-",
        ax=ax,
    )

    return pos

def label_nodes(ts, labels=None):
    """
    Adds a metadata item called "name" to each node, whose value
    is given by the labels dictionary passed in. If labels is None (default),
    use the dictionary {0: 'A', 1: 'B', 2: 'C', ... 26: 'Z'}.
    Any nodes without a corresponding key in the labels dictionary will
    simply have their metadata value set to their node id.
    
    Note that this means that if no labels are given, nodes 26 onwards
    will be labelled with numbers rather than ascii uppercase letters.
    """
    if labels is None:
        labels = {i: lab for i, lab in enumerate(string.ascii_uppercase)}
    tables = ts.dump_tables()

    tables.nodes.clear()
    for i, nd in enumerate(ts.tables.nodes):
        m = nd.metadata or {}
        # assume we can set metadata to a dict: e.g. that node.metadata_schema is json
        tables.nodes.append(nd.replace(metadata={**m, "name": labels.get(i, i)}))
    return tables.tree_sequence()


def convert_nx(ts):
    """
    Returns the specified tree sequence as an networkx directed graph.
    """
    G = nx.DiGraph()
    edges = collections.defaultdict(list)
    for edge in ts.edges():
        edges[(edge.child, edge.parent)].append((edge.left, edge.right))
    for node in ts.nodes():
        G.add_node(node.id, time=node.time, flags=node.flags)
    for edge, intervals in edges.items():
        G.add_edge(*edge, intervals=intervals)
    return G


def nx_get_dot_pos(G):
    """
    Layout using graphviz's "dot" algorithm and return a dict of positions in the
    format required by networkx. We assume that the nodes have a "time" attribute
    """
    P=nx.drawing.nx_pydot.to_pydot(G) # switch to a Pydot representation
    nodes_at_time = collections.defaultdict(list)
    for nd in P.get_nodes():
        nodes_at_time[float(nd.get("time"))].append(nd)
    
    # First cluster all nodes at the same times (probably mostly samples)
    for t, nodes in nodes_at_time.items():
        if len(nodes) > 1:
            S=pydot.Cluster(f"cluster_t{t}")
            for nd in nodes:
                S.add_node(nd)
            P.add_subgraph(S)
    graphviz_bytes = P.create_dot()
    graphs = pydot.graph_from_dot_data(graphviz_bytes.decode())
    assert len(graphs) == 1
    graph = graphs[0]
    # negate the coords to get the y axis going in the direction we want.
    return {
        # [1:-1] snips off enclosing quotes, negation corrects the y axis direction
        int(n.get_name()): -np.fromstring(n.get_pos()[1:-1], sep=",")
        # need to iterate over this graph and also all the subgraphs to get all the nodes
        for g in itertools.chain([graph], graph.get_subgraphs()) for n in g.get_nodes()
        if n.get_pos() and n.get_name().isdigit()
    }

