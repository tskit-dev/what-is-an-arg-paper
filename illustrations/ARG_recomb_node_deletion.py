import collections
import io
import itertools
import sys
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tskit

current_dir = Path(__file__).parent

# useful modules in the top level dir: we should be able to import this if the script is
# run from the top level dir, but for some reason it fails, so we just add the path here
sys.path.append(str(current_dir.parent.absolute()))
import argutils

outfile = "ARG_recomb_node_deletion"

def arg_node_simplification():
    seed = 372
    t_x={9: 3, 6: -8, 18: 50, 19: -15, 17: 5, 13: 10, 14: 2, 10: -10, 15: 5, 16: -5, 7: -5, 11: 5, 8: 8}

    ts = argutils.sim_wright_fisher(2, 10, 100, recomb_proba=0.1, seed=seed)
    labels = {i: string.ascii_uppercase[i] for i in range(len(string.ascii_uppercase))}
    # relabel the nodes to get samples reading A B C D
    labels.update({2: 'B', 3: 'C', 1: "D"})
    ts = argutils.viz.label_nodes(ts, labels)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,5))
    ax1.set_title("(a) little ARG")
    ax2.set_title("(b) remove pass through")
    ax3.set_title("(c) simplify (keep unary in coal)")
    ax4.set_title("(d) fully simplified")
    col = mpl.colors.to_hex(plt.cm.tab20(1))
    pos = argutils.viz.draw(
        ts, ax1,
        draw_edge_widths=True,
        use_ranked_times=True, node_color=col, tweak_x=t_x)
    ts2, node_map = argutils.simplify_remove_pass_through(ts, repeat=True, map_nodes=True)
    argutils.viz.draw(
        ts2, ax2,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()}, node_color=col)
    ts3, node_map = argutils.simplify_keeping_unary_in_coal(ts, map_nodes=True)
    argutils.viz.draw(
        ts3, ax3,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()}, node_color=col)
    ts4, node_map = ts.simplify(map_nodes=True)
    argutils.viz.draw(
        ts4, ax4,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()}, node_color=col)
    
    
    graph1_io = io.StringIO()
    plt.savefig(graph1_io, format="svg", bbox_inches='tight')
    graph1_svg = graph1_io.getvalue()
    graph1_svg = graph1_svg[graph1_svg.find("<svg"):]
    
    tree_svg = ts.draw_svg(
        size=(970, 250),
        time_scale="rank",
        node_labels=labels,
        style=".x-axis .tick .lab {font-weight: regular; font-size: 12}"
    )

    # figlabelstyle = 'font-family="serif" font-size="30px"'

    svg = [
        '<svg width="900" height="700" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        '<style>.tree-sequence text {font-family: sans-serif}</style>'
        '<g transform="translate(10, 50)">',
    ]
    svg.append('<g transform="scale(0.87)">' + graph1_svg + "</g>")
    svg.append('<g transform="translate(0 400) scale(0.83)">' + tree_svg + "</g>")
    svg.append("</g></svg>")
    return "\n".join(svg)


svg = (
    '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
    '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
)
svg += arg_node_simplification()
with open(current_dir / f"{outfile}.svg", "wt") as f:
    f.write(svg)
    
