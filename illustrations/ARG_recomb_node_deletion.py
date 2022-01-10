import collections
import io
import itertools
import sys
from pathlib import Path


import msprime
import networkx as nx
import numpy as np
import tskit
import matplotlib.pyplot as plt

outfile = "ARG_recomb_node_deletion"

# useful scripts in the top level dir
sys.path.append(str(Path(__file__).parent.parent.absolute() / "utils"))
import convert
import ts_process
import argdraw


def positions_from_ts(ts, pos=None, return_graph=False):
    ts_arg = ts_process.flag_unary_nodes(ts)
    G = ts_process.to_networkx_graph(ts_arg)
    if pos is None:
        pos = argdraw.nx_get_dot_pos(G, add_invisibles=False)
    return (pos, G) if return_graph else pos
    

def plot_ts_arg(ts, ax, labels=None, pos=None, arrows=False):
    pos, G = positions_from_ts(ts, pos, return_graph=True)
    col_map = argdraw.nx_ts_colour_map(G)

    argdraw.nx_draw_with_curved_multi_edges(
        G, pos, col_map, curve_scale=30, ax=ax, labels=labels)

# SIMULATE
ts = ts_process.convert_to_single_rec_node(
    msprime.sim_ancestry(
        3,
        sequence_length=1e4,
        population_size=1e4,
        recombination_rate=1e-8,
        record_full_arg=True, 
        random_seed=76, # by trial and error, a seed of 76 gives a nice small example
    )
)

node_pos = positions_from_ts(ts)

# relabel the nodes to get the order right
tables = ts.dump_tables()
node_map = np.arange(ts.num_nodes, dtype=tables.edges.parent.dtype)
sample_ids = ts.samples()
assert np.all(sample_ids == np.arange(ts.num_samples))
node_map[np.argsort([node_pos[s][0] for s in sample_ids])] = sample_ids
tables.nodes.clear()
for u in node_map:
    tables.nodes.append(ts.tables.nodes[u])
tables.edges.parent = node_map[tables.edges.parent]
tables.edges.child = node_map[tables.edges.child]
tables.sort()
ts=tables.tree_sequence()

# Get the positions again, as the nodes are now if a different order
node_pos = positions_from_ts(ts)

ts_simp, node_map = ts.simplify(map_nodes=True)
labels = {j: i for i, j in enumerate(node_map) if j>=0}

# tweak some node positions for this example
node_pos[22][0] *= 1.05
node_pos[11][0] *= 0.9

simp_node_pos = {j:node_pos[i] for i, j in enumerate(node_map) if j>=0}
fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
plot_ts_arg(ts, ax1, pos=node_pos)
graph1_io = io.StringIO()
plt.savefig(graph1_io, format="svg", bbox_inches='tight')
graph1_svg = graph1_io.getvalue()
graph1_svg = graph1_svg[graph1_svg.find("<svg"):]

fig, ax2 = plt.subplots(1, 1, figsize=(5, 6))
plot_ts_arg(ts_simp, ax2, labels=labels, pos=simp_node_pos)
graph2_io = io.StringIO()
plt.savefig(graph2_io, format="svg", bbox_inches='tight')
graph2_svg = graph2_io.getvalue()
graph2_svg = graph2_svg[graph2_svg.find("<svg"):]

tree_svg = ts_simp.draw_svg(
    size=(900, 200),
    time_scale="rank",
    node_labels=labels,
    style=".x-axis .tick .lab {font-weight: regular; font-size: 12}"
)

with open(Path(__file__).parent / (outfile + ".svg"), "wt") as file:
    figlabelstyle = 'font-family="serif" font-size="30px"'
    print(
        '<svg baseProfile="full" height="700" version="1.1" width="920"',
        'xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events"',
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        file=file)
    print(
        '<g transform="translate(40 0)">' + graph1_svg + '</g>',
        f'<g transform="translate(20 30)"><text {figlabelstyle}>a</text></g>',
        '<g transform="translate(500 0)">' + graph2_svg + '</g>',
        f'<g transform="translate(480 30)"><text {figlabelstyle}>b</text></g>',
        '<g transform="translate(0 500)">' + tree_svg + '</g>',
        f'<g transform="translate(20 490)"><text {figlabelstyle}>c</text></g>',
        file=file)
    print('</svg>', file=file)
