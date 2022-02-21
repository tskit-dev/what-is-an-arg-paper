import collections

import pandas as pd
import networkx as nx
import tskit


def convert_argweaver(infile, epsilon_scale=1e-5):
    """
    Convert an ARGweaver .arg file to a tree sequence. An example .arg file is at

    https://github.com/CshlSiepelLab/argweaver/blob/master/test/data/test_trans/0.arg

    Times are adjusted so that parents are older than children by a small amount
    `epsilon` which is calculated by taking the largest number of nodes at a single time
    and dividing the smalles gap between node times by that value, then multiplying by
    a small value given by `epsilon_scale`.
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

    # Make an nx DiGraph so we can do a topological sort.
    G = nx.DiGraph()
    for row in name_to_record.values():
        child = row["name"]
        parents = row["parents"]
        G.add_node(child)
        if isinstance(parents, str):
            for parent in row["parents"].split(","):
                G.add_edge(child, parent)

    tables = tskit.TableCollection(sequence_length=end)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    breakpoints = np.full(len(G), tables.sequence_length)
    for node in nx.topological_sort(G):
        record = name_to_record[node]
        flags = 0
        if node.startswith("n"):
            flags = tskit.NODE_IS_SAMPLE
            assert record["age"] == 0
            assert record["event"] == "gene"
            time = 0
        else:
            # Use topological sort order for age for the moment.
            time += 1
        num_nodes = 2 if record["event"] == "recomb" else 1
        for _ in range(num_nodes):
            tsk_node = tables.nodes.add_row(flags=flags, time=time, metadata=record)
            aw_to_tsk_node[node].append(tsk_node)

    for record in name_to_record.values():
        aw_parent = record["name"]
        print(aw_parent, "->", record["children"])

    print(tables.nodes)
    print(aw_to_tsk_node)




    # G.add_edges_from([e[:2] for e in E])
    # # We have to get the topological sorting of the nodes because we don't
    # # have time ordering.
    # topological = list(nx.topological_sort(G))

    # # sort topologically, so parents are after children
    # graph = {
    #     r.name: set([] if pd.isna(r.parents) else r.parents.split(","))
    #     for i, r in df.iterrows()
    # }
    # sorter = graphlib.TopologicalSorter(graph)

    # items = tuple(sorter.static_order())
    # df.loc[items, "topo_order"] = np.arange(len(df))[::-1]
    # # sort by time first, then put parents after children within a timeslice
    # df.sort_values(by=["age", "topo_order"], inplace=True)
    # df["tskit_id"] = np.arange(len(df))

    # tables = tskit.TableCollection(sequence_length=end)

    # # Find a reasonable epsilon value
    # epsilon = np.diff(df["age"])
    # _, max_in_age = np.unique(df["age"], return_counts=True)
    # max_in_age = max(max_in_age)
    # epsilon = epsilon_scale * min(epsilon[epsilon > 0]) / max_in_age

    # prev_time = 0
    # for i, r in df.iterrows():
    #     if r.event == "gene":
    #         # assume "gene" nodes cannot be ancestors, so can be placed at their real time
    #         tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=r.age)
    #     else:
    #         flags = NODE_IS_RECOMB if r.event == "recomb" else 0
    #         if prev_time == r.age:
    #             tables.nodes.add_row(flags=flags, time=tables.nodes[-1].time + epsilon)
    #         else:
    #             tables.nodes.add_row(flags=flags, time=r.age)
    #     prev_time = r.age

    # times = tables.nodes.time
    # df["tskit_time"] = times

    # for i, r in df.iterrows():
    #     # .arg format contains redundant information, as both children and parents of a node
    #     # are stored, so most edges are describved twice. Here we only add parents of each node
    #     if pd.isna(r.children):
    #         children = []
    #     else:
    #         children = [df.loc[name, "tskit_id"] for name in r.children.split(",")]
    #     if pd.isna(r.parents):
    #         parents = []
    #     else:
    #         parents = [df.loc[name, "tskit_id"] for name in r.parents.split(",")]

    #     own_id = df.loc[r.name, "tskit_id"]
    #     if r.event == "recomb":
    #         assert len(parents) == 2
    #         assert start <= r.pos <= end
    #         assert times[own_id] < times[parents[0]], (
    #             r,
    #             times[own_id],
    #             times[parents[0]],
    #         )
    #         tables.edges.add_row(
    #             parent=parents[0], child=own_id, left=start, right=r.pos
    #         )
    #         assert times[own_id] < times[parents[1]], (
    #             r,
    #             times[own_id],
    #             times[parents[1]],
    #         )
    #         tables.edges.add_row(parent=parents[1], child=own_id, left=r.pos, right=end)

    #     elif r.event == "coal":
    #         assert len(children) == 2
    #         assert len(parents) < 2
    #         assert r.pos == 0
    #         for p in parents:
    #             tables.edges.add_row(parent=p, child=own_id, left=start, right=end)

    #     elif r.event == "gene":
    #         assert r.pos == 0
    #         assert len(children) == 0
    #         assert len(parents) == 1
    #         for p in parents:
    #             tables.edges.add_row(parent=p, child=own_id, left=start, right=end)

    # tables.sort()
    # tables.simplify(keep_unary=True)

    # return tables.tree_sequence()
