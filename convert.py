import re

import msprime
import numpy as np
import pandas as pd
import tskit

from constants import NODE_IS_RECOMB

def arg_to_ts(file):
    start, end = next(file).strip().split()
    assert start.startswith("start=")
    start = int(start[len("start="):])
    assert end.startswith("end=")
    end = int(end[len("end="):])
    df = pd.read_csv(file, header=0, sep="\t", index_col=0)
    df["tskit_id"] = np.arange(len(df))  # allocate temporarily
    # sort by time, then name
    df.sort_values(by=["age", "tskit_id"], inplace=True)
    df["tskit_id"] = np.arange(len(df))


    tables = tskit.TableCollection(sequence_length=end)

    # Find a reasonable epsilon value
    epsilon = np.diff(df["age"])
    _, max_in_age = np.unique(df["age"], return_counts=True)
    max_in_age = max(max_in_age)
    epsilon = min(epsilon[epsilon > 0]) / max_in_age / 1e5

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