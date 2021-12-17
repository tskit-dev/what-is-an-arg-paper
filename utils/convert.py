import re
import graphlib  # requires python >= 3.9

import msprime
import numpy as np
import pandas as pd
import tskit

from constants import NODE_IS_RECOMB

def arg_to_ts(file, epsilon_scale = 1e-5):
    """
    Convert an ARGweaver .arg file to a tree sequence. An example .arg file is at
    
    https://github.com/CshlSiepelLab/argweaver/blob/master/test/data/test_trans/0.arg
    
    Times are adjusted so that parents are older than children by a small amount
    `epsilon` which is calculated by taking the largest number of nodes at a single time
    and dividing the smalles gap between node times by that value, then multiplying by
    a small value given by `epsilon_scale`.
    """
    start, end = next(file).strip().split()
    assert start.startswith("start=")
    start = int(start[len("start="):])
    assert end.startswith("end=")
    end = int(end[len("end="):])
    # the "name" field can be a string. Force it to be so, in case it is just numbers
    df = pd.read_csv(file, header=0, sep="\t", dtype={'name': str, 'parents': str})
    df.set_index('name', inplace=True)
    # sort topologically, so parents are after children
    graph = {
        r.name: set([] if pd.isna(r.parents) else r.parents.split(","))
        for i, r in df.iterrows()
    }
    sorter = graphlib.TopologicalSorter(graph)
    items = tuple(sorter.static_order())
    df.loc[items, 'topo_order'] = np.arange(len(df))[::-1]
    # sort by time first, then put parents after children within a timeslice
    df.sort_values(by=["age", "topo_order"], inplace=True)
    df["tskit_id"] = np.arange(len(df))

    tables = tskit.TableCollection(sequence_length=end)

    # Find a reasonable epsilon value
    epsilon = np.diff(df["age"])
    _, max_in_age = np.unique(df["age"], return_counts=True)
    max_in_age = max(max_in_age)
    epsilon = epsilon_scale * min(epsilon[epsilon > 0]) / max_in_age 

    prev_time = 0
    for i, r in df.iterrows():
        if r.event == "gene":
            # assume "gene" nodes cannot be ancestors, so can be placed at their real time
            tables.nodes.add_row(flags = tskit.NODE_IS_SAMPLE, time=r.age)
        else:
            flags = NODE_IS_RECOMB if r.event == "recomb" else 0
            if prev_time == r.age:
                tables.nodes.add_row(flags=flags, time=tables.nodes[-1].time + epsilon)
            else:
                tables.nodes.add_row(flags=flags, time=r.age)
        prev_time = r.age

    times = tables.nodes.time
    df["tskit_time"] = times

    for i, r in df.iterrows():
        # .arg format contains redundant information, as both children and parents of a node
        # are stored, so most edges are describved twice. Here we only add parents of each node
        if pd.isna(r.children):
            children = []
        else:
            children = [df.loc[name, "tskit_id"] for name in r.children.split(",")]
        if pd.isna(r.parents):
            parents = []
        else:
            parents = [df.loc[name, "tskit_id"] for name in r.parents.split(",")]

        own_id = df.loc[r.name, "tskit_id"]
        if r.event == "recomb":
            assert len(parents) == 2
            assert start <= r.pos <= end
            assert times[own_id] < times[parents[0]], (r, times[own_id],  times[parents[0]])
            tables.edges.add_row(parent=parents[0], child=own_id, left=start, right=r.pos)
            assert times[own_id] < times[parents[1]], (r, times[own_id],  times[parents[1]])
            tables.edges.add_row(parent=parents[1], child=own_id, left=r.pos, right=end)

        elif r.event == "coal":
            assert len(children) == 2
            assert len(parents) < 2
            assert r.pos == 0
            for p in parents:
                tables.edges.add_row(parent=p, child=own_id, left=start, right=end)

        elif r.event == "gene":
            assert r.pos == 0
            assert len(children) == 0
            assert len(parents) == 1
            for p in parents:
                tables.edges.add_row(parent=p, child=own_id, left=start, right=end)

    tables.sort()
    tables.simplify(keep_unary=True)

    return tables.tree_sequence()