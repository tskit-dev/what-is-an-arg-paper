import collections

import pandas as pd
import networkx as nx
import numpy as np
import tskit

import argutils


def convert_argweaver(infile):
    """
    Convert an ARGweaver .arg file to a tree sequence. An example .arg file is at

    https://github.com/CshlSiepelLab/argweaver/blob/master/test/data/test_trans/0.arg
    """
    start, end = next(infile).strip().split()
    assert start.startswith("start=")
    start = int(start[len("start=") :])
    assert end.startswith("end=")
    end = int(end[len("end=") :])
    # the "name" field can be a string. Force it to be so, in case it is just numbers
    df = pd.read_csv(infile, header=0, sep="\t", dtype={"name": str, "parents": str})

    name_to_record = {}
    for _, row in df.iterrows():
        row = dict(row)
        name_to_record[row["name"]] = row
    # We could use nx to do this, but we want to be sure the order is correct.
    parent_map = collections.defaultdict(list)

    # Make an nx DiGraph so we can do a topological sort.
    G = nx.DiGraph()
    time_map = {} # argweaver times to allocated time
    for row in name_to_record.values():
        child = row["name"]
        parents = row["parents"]
        time_map[row["age"]] = row["age"]
        G.add_node(child)
        if isinstance(parents, str):
            for parent in row["parents"].split(","):
                G.add_edge(child, parent)
                parent_map[child].append(parent)

    tables = tskit.TableCollection(sequence_length=end)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    breakpoints = np.full(len(G), tables.sequence_length)
    aw_to_tsk_id = {}
    min_time_diff = min(np.diff(sorted(time_map.keys())))
    epsilon = min_time_diff / 1e6
    for node in nx.lexicographical_topological_sort(G):
        record = name_to_record[node]
        flags = 0
        # Sample nodes are marked as "gene" events
        if record["event"] == "gene":
            flags = tskit.NODE_IS_SAMPLE
            assert record["age"] == 0
            time = record["age"]
        else:
            time = time_map[record["age"]]
            # Argweaver allows age of parent and child to be the same, so we
            # need to add epsilons to enforce parent_age > child_age
            time_map[record["age"]] += epsilon
        tsk_id = tables.nodes.add_row(flags=flags, time=time, metadata=record)
        aw_to_tsk_id[node] = tsk_id
        if record["event"] == "recomb":
            breakpoints[tsk_id] = record["pos"]

    L = tables.sequence_length
    for aw_node in G:
        child = aw_to_tsk_id[aw_node]
        parents = [aw_to_tsk_id[aw_parent] for aw_parent in parent_map[aw_node]]
        if len(parents) == 1:
            tables.edges.add_row(0, L, parents[0], child)
        elif len(parents) == 2:
            # Recombination node.
            # If we wanted a GARG here we'd add an extra node
            x = breakpoints[child]
            tables.edges.add_row(0, x, parents[0], child)
            tables.edges.add_row(x, L, parents[1], child)
        else:
            assert len(parents) == 0
    # print(tables)
    tables.sort()
    # print(tables)
    ts = tables.tree_sequence()
    # The plan here originally was to use the earg_to_garg method to
    # convert the recombination events to two parents (making a
    # standard GARG). However, there are some complexities here so
    # returning the ARG topology as defined for now. There is an
    # argument that we should do this anyway, since that's the structure
    # that was returned and makes very little difference.

    # garg = argutils.earg_to_garg(ts)

    return ts.simplify(keep_unary=True)

def convert_kwarg(infile, num_samples, sequence_length, sample_names=None):
    """
    Convert a KwARG output file to a tree sequence.
    """

    tables = tskit.TableCollection(sequence_length=sequence_length)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    time = 0.0

    node_ids = {}
    for n in range(num_samples):
        node_ids[n] = n
        if sample_names is not None and n in sample_names:
            name = sample_names[n]
        else:
            name = n
        tsk_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=0.0, metadata={"name": name})

    for x in infile:
        line = x.split()
        if (line[0] == "Mutation"):
            site_id = tables.sites.add_row(position=int(line[3]) - 1, ancestral_state='0')
            tables.mutations.add_row(site=site_id, node=node_ids[int(line[6]) - 1], derived_state='1')
        elif (line[0] == "Coalescing"):
            c1 = node_ids[int(line[2]) - 1]
            c2 = node_ids[int(line[4]) - 1]
            time += 1
            tsk_id = tables.nodes.add_row(flags=0, time=time, metadata={"coal": line[2]})
            node_ids[int(line[2]) - 1] = tsk_id
            tables.edges.add_row(0, sequence_length, tsk_id, c1)
            tables.edges.add_row(0, sequence_length, tsk_id, c2)
        elif (line[0] == "---->Recombination"):
            breakpoint = int(line[6][:-1]) - 1
            n1 = node_ids[int(line[3]) - 1]
            time += 1
            tsk_id_1 = tables.nodes.add_row(flags=0, time=time, metadata={"rec": line[3]})
            node_ids[int(line[3]) - 1] = tsk_id_1
            tsk_id_2 = tables.nodes.add_row(flags=0, time=time, metadata={"rec": line[11]})
            node_ids[int(line[11]) - 1] = tsk_id_2
            tables.edges.add_row(0, breakpoint, tsk_id_1, n1)
            tables.edges.add_row(breakpoint, sequence_length, tsk_id_2, n1)

    tables.sort()
    ts = tables.tree_sequence()

    return argutils.simplify_keeping_all_nodes(ts)


def convert_relate_without_times(infile):
    """
    Convert a Relate .anc file to a tree sequence, but with time_units="uncalibrated"
    which allows us to create an proper ARG rather than a JBOT. See
    https://myersgroup.github.io/relate/getting_started.html#Output
    
    NUM_HAPLOTYPES 8
NUM_TREES 7619
0: 8:(607.26519 0.000 0 86) 13:(2702.54805 0.000 0 28) 9:(2641.61779 0.000 0 41) 11:(3724.67317 3.000 0 28) 9:(2641.61779 7.000 0 41) 12:(5155.21459 5.000 0 21) 10:(3088.44253 4.000 0 28) 8:(607.26519 4.000 0 86) 13:(2095.28286 2.000 0 28) 10:(446.82474 0.000 0 28) 11:(636.23064 0.000 0 28) 12:(1430.54142 0.000 0 21) 14:(1247.46463 0.000 0 21) 14:(3700.13116 2.000 0 21) -1:(0.00000 0.000 0 21) 
21: 8:(667.04955 0.000 0 86) 11:(2266.08783 0.000 0 28) 9:(2335.63438 0.000 0 41) 12:(3455.60941 3.000 0 28) 9:(2335.63438 7.000 0 41) 14:(7848.96712 
5.000 21 41) 10:(3236.77021 4.000 0 28) 8:(667.04955 4.000 0 86) 11:(1599.03828 2.000 0 28) 10:(901.13583 0.000 0 28) 12:(218.83920 0.000 0 28) 13:(14
58.13811 0.000 21 28) 13:(268.61652 0.000 21 28) 14:(4124.74119 5.000 21 41) -1:(0.00000 0.000 21 28) 

    """
    # Make an nx DiGraph so we can do a topological sort.
    G = nx.DiGraph()
    haps = next(infile)
    assert haps.startswith("NUM_HAPLOTYPES")
    num_samples = int(haps[len("NUM_HAPLOTYPES"):])
    node_map={}
    for i in range(num_samples):
        G.add_node(i)
        node_map[i] = i
    trees = next(infile)
    assert haps.startswith("NUM_TREES")
    num_trees = int(haps[len("NUM_TREES"):])
    edges = {}
    for line in infile:
        p = line.find(":")
        assert p >= 0
        left_pos = line[:p]
        line = line[(p+1):-1].strip()
        assert line[-1] == ")"
        branches = line.split(")")
        tree = []
        for child, branch in enumerate(branches):
            p = branch.find(":")
            parent = int(branch[:p])
            vals = branch[p:].strip().split() # split on whitespace
            # In Relate, node IDs are not necessarily shared across trees. We need to
            # make node IDs unique by looking at each tree, going through the nodes
            # in postorder, and if the edge to the node hasn't been seen before (i.e.
            # with the same (parent, child, left, right) values, we change the parent
            # node ID to a new unique one (note that this means we must have SMC trees
            # (as we can't know if we have returned to the same node in later trees)
            edge = (child, parent, val[-2], vals[-1])
            tree.push(edge)
            # parent, child, left, right -> new_parent_id
            if edge not in edges:
                i = len(node_map)
                node_map[parent] = i
                G.add_node(i)
                edges[edge] = i  # record that in this edge the parent is actually ID i
            else:
                node_map[parent] = edges[edge]
        for edge in tree:
            G.add_edge(node_map[edge[0]], node_map[edge[1]], left=edge[2], right=edge[3])
         
            
        