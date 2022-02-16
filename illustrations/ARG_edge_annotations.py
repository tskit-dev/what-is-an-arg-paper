import io
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tskit

current_dir = Path(__file__).parent

# useful modules in the top level dir: we should be able to import this if the script is
# run from the top level dir, but for some reason it fails, so we just add the path here
sys.path.append(str(current_dir.parent.absolute()))
import argutils

outfile = "ARG_edge_annotations"

def arg_edge_annotations():
    def add_edge_labels(ax, ts, G, pos):
        def edge_not_above_recombinant(edge):
            return not argutils.is_recombinant(ts.node(edge.child).flags)
        def left_recombinant_edge(edge):
            if not argutils.is_recombinant(ts.node(edge.child).flags):
                return False
            if edge.right == ts.sequence_length:
                return False
            return True
    
        def right_recombinant_edge(edge):
            if not argutils.is_recombinant(ts.node(edge.child).flags):
                return False
            if edge.left == 0:
                return False
            if (edge.left==4 and edge.right==6):  # Hack for this ts
                return False
            return True
    
        
        for halign, func1 in {
            "center": edge_not_above_recombinant,
            "right": left_recombinant_edge,
            "left": right_recombinant_edge,
        }.items():
            for full_edge, func2 in {
                True: lambda e: e.left==0 and e.right==7,
                False: lambda e: not(e.left==0 and e.right==7),
            }.items():
                nx.draw_networkx_edge_labels(
                    G,
                    pos=pos,
                    ax=ax,
                    rotate=False,
                    font_weight="normal" if full_edge else "bold",
                    alpha=0.5 if full_edge else None,
                    font_size=16,
                    edge_labels={(e.child, e.parent): f"({e.left:.0f},{e.right:.0f}]" for e in ts.edges() if func1(e) and func2(e)},
                    horizontalalignment=halign,
                )        
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,13), sharey=True)
    
    ax1.set_title("(a) eARG with implicit\nencoding (Wiuf & Hein)")
    ts = argutils.viz.label_nodes(argutils.wh99_example())
    pos, G = argutils.viz.draw(
        ts, ax1,
        use_ranked_times=False,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=450,
        max_edge_width=2,
        font_size=14,
        tweak_x={
            0: 10, 3: 28, 4: 2, 7: 20, 11: 10, 12: -18,
            20: -23, 19: 4, 21: -11, 17: -25, 13: -16,
            8: 2, 9: -24, 5: -16, 6: -41, 15: 1, 16: -26, 2: -10
        }
    )
    add_edge_labels(ax1, ts, G, pos)
    
    ax2.set_title("(b) explicit encoding\n(i.e. non-ancestral removed)")
    ts2 = argutils.simplify_keeping_all_nodes(ts)
    pos, G = argutils.viz.draw(
        ts2, ax2, pos=pos,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=450,
        font_size=14,
        #arrows=True,
        draw_edge_widths=True,
    )
    add_edge_labels(ax2, ts2, G, pos)
    

    graph_io = io.StringIO()
    plt.savefig(graph_io, format="svg", bbox_inches='tight')
    graph_svg = graph_io.getvalue()
    
    svg = [
        # Could caoncatenate more SVG stuff here in <g> tags, e.g.
        # if we wanted to draw the 2 plots as 2 separate svg plots\
        # rather than using plt.subplots
        graph_svg[graph_svg.find("<svg"):]
    ]
    return "\n".join(svg)


svg = (
    '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
    '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
)
svg += arg_edge_annotations()
with open(current_dir / f"{outfile}.svg", "wt") as f:
    f.write(svg)
    
