import msprime
import networkx as nx
import numpy as np
import pandas as pd
import tskit

from constants import NODE_IS_RECOMB, NODE_IS_SOMETIMES_UNARY, NODE_IS_ALWAYS_UNARY

def convert_to_single_rec_node(msprime_ts):
    """
    Should be unnecessary when https://github.com/tskit-dev/msprime/issues/1942 is fixed
    """
    tables = msprime_ts.dump_tables()
    tables.nodes.clear()
    node_mapping = np.zeros(msprime_ts.num_nodes, dtype=tables.edges.child.dtype)
    node_id = 0
    while node_id < msprime_ts.num_nodes:
        node_mapping[node_id] = tables.nodes.num_rows
        node = msprime_ts.node(node_id)
        if (node.flags & msprime.NODE_IS_RE_EVENT):
            node_id += 1
            assert node_id < msprime_ts.num_nodes
            assert msprime_ts.node(node_id).flags & msprime.NODE_IS_RE_EVENT
            assert msprime_ts.node(node_id).time == node.time
            node_mapping[node_id] = tables.nodes.num_rows
            node.flags &= ~msprime.NODE_IS_RE_EVENT  # Unset the NODE_IS_RE_EVENT bit
            node.flags |= NODE_IS_RECOMB
        tables.nodes.append(node)
        node_id += 1
    tables.edges.parent = node_mapping[tables.edges.parent]
    tables.edges.child = node_mapping[tables.edges.child]
    tables.mutations.node = node_mapping[tables.mutations.node]
    tables.sort()
    return tables.tree_sequence()

def flag_unary_nodes(ts):
    """
    Return a tree sequence in which any nodes which are sometimes unary nodes in the TS
    (i.e. there is a tree in which they node has only one child) is flagged with 
    NODE_IS_SOMETIMES_UNARY, and nodes that are always unary are flagged with
    NODE_IS_ALWAYS_UNARY
    """
    # This is very inefficient - find a proper incremental appraoch
    sometimes_unary = np.zeros(ts.num_nodes, dtype=bool)
    sometimes_not_unary = np.zeros(ts.num_nodes, dtype=bool)
        
    tables = ts.dump_tables()
    for tree in ts.trees():
        for u in tree.nodes():
            if tree.num_children(u) == 1:
                sometimes_unary[u] = True
            if tree.num_children(u) != 1:
                sometimes_not_unary[u] = True
    flags = tables.nodes.flags
    flags &= np.full_like(flags, ~NODE_IS_SOMETIMES_UNARY)  # unset this bit
    flags &= np.full_like(flags, ~NODE_IS_ALWAYS_UNARY)  # unset this bit
    flags[sometimes_unary] |= NODE_IS_SOMETIMES_UNARY
    flags[np.logical_and(sometimes_unary, np.logical_not(sometimes_not_unary))] |= NODE_IS_ALWAYS_UNARY
    tables.nodes.flags = flags
    return tables.tree_sequence()

def to_networkx_graph(ts):
    edges=ts.tables.edges
    G = nx.from_pandas_edgelist(
        pd.DataFrame({'source': edges.parent, 'target': edges.child, 'left': edges.left, 'right': edges.right}),
        edge_attr=True,
        create_using=nx.MultiDiGraph
    )
    nx.set_node_attributes(G, {n.id: {'flags':n.flags, 'time': n.time, 'labels': "foo"} for n in ts.nodes()})
    return G

def add_individuals_to_coalescence_nodes(ts):
    tables = ts.dump_tables()
    nodes_individual = tables.nodes.individual
    for node in ts.nodes():
        if (node.flags & NODE_IS_RECOMB) or (node.flags & msprime.NODE_IS_CA_EVENT):
            continue
        if node.individual == tskit.NULL:
            nodes_individual[node.id] = tables.individuals.add_row()
    tables.nodes.individual = nodes_individual
    return tables.tree_sequence()