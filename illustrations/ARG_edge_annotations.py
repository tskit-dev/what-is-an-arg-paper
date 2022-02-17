import io
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tskit
import PIL

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
                    bbox=dict(boxstyle="round,pad=0.05", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
                )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,12), sharey=True)
    fig.tight_layout()
    
    ax1.set_title("(a) eARG with implicit\nencoding (Wiuf & Hein)", fontsize="xx-large")
    ts = argutils.viz.label_nodes(argutils.wh99_example())
    pos, G = argutils.viz.draw(
        ts, ax1,
        use_ranked_times=True,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=100,
        max_edge_width=2,
        font_size=14,
        tweak_x={
            0: 10, 3: 29, 4: 2.5, 7: 20, 11: 9, 12: -17,
            20: -22.5, 19: 3.5, 21: -11, 17: -25, 13: -16,
            8: 2, 9: -24, 5: -16, 6: -41, 15: 0.6, 16: -25.6, 2: -10
        },
        tweak_y={
            22: 1, 17: 0.8
        }
    )
    add_edge_labels(ax1, ts, G, pos)
    
    ax2.set_title("(b) explicit encoding\n(i.e. non-ancestral removed)", fontsize="xx-large")
    ts2 = argutils.simplify_keeping_all_nodes(ts)
    pos, G = argutils.viz.draw(
        ts2, ax2, pos=pos,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=200,
        font_size=14,
        #arrows=True,
        draw_edge_widths=True,
    )
    add_edge_labels(ax2, ts2, G, pos)
    
    
    # From https://networkx.org/documentation/stable/auto_examples/drawing/plot_custom_node_icons.html
    icons = {
        "genome_empty": "node_icons/genome_empty.png",
        "genome_empty_hamburger": "node_icons/genome_empty_hamburger.png",
        "genome_full": "node_icons/genome_full.png",
        "genome_0-4": "node_icons/genome_0-4.png",
        "genome_4-7": "node_icons/genome_4-7.png",
        "genome_0-2": "node_icons/genome_0-2.png",
        "genome_2-7": "node_icons/genome_2-7.png",
        "genome_0-2-4-7": "node_icons/genome_0-2-4-7.png",
        "genome_4-6": "node_icons/genome_4-6.png",
        "genome_6-7": "node_icons/genome_6-7.png",
        "genome_0-1": "node_icons/genome_0-1.png",
        "genome_00-11": "node_icons/genome_00-11.png",
        "genome_1-7": "node_icons/genome_1-7.png",
        "genome_00-22-2-7": "node_icons/genome_00-22-2-7.png",
        "genome_11-22-2-7": "node_icons/genome_11-22-2-7.png",
        "genome_22-77": "node_icons/genome_22-77.png",
        "genome_ancestral": "node_icons/genome_ancestral.png",
    }
    # Load images
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}
    # Panel (a)
    tr_figure = ax1.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    icon_size = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.0003
    icon_center = icon_size / 2.0
    for n in G.nodes:
        G.nodes[n]["image"] = images["genome_empty"]
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        a.set_title(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[n], y=0, verticalalignment="bottom", loc="center", fontsize="xx-large")
        a.axis("off")

        
    # Panel (b)
    tr_figure = ax2.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    icon_size = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.0003
    icon_center = icon_size / 2.0
    for n in [0, 1, 2, 7, 14, 18, 22]:
        G.nodes[n]["image"] = images["genome_full"]
    for n in [3]:
        G.nodes[n]["image"] = images["genome_0-4"]
    for n in [15]:
        G.nodes[n]["image"] = images["genome_4-6"]
    for n in [4, 12, 13]:
        G.nodes[n]["image"] = images["genome_4-7"]
    for n in [16]:
        G.nodes[n]["image"] = images["genome_6-7"]
    for n in [9]:
        G.nodes[n]["image"] = images["genome_empty_hamburger"]
    for n in [10]:
        G.nodes[n]["image"] = images["genome_0-2-4-7"]
    for n in [5, 8, 11]:
        G.nodes[n]["image"] = images["genome_0-2"]
    for n in [19]:
        G.nodes[n]["image"] = images["genome_00-11"]
    for n in [20]:
        G.nodes[n]["image"] = images["genome_11-22-2-7"]
    for n in [21]:
        G.nodes[n]["image"] = images["genome_22-77"]
    for n in [6, 17]:
        G.nodes[n]["image"] = images["genome_2-7"]
    for n in [14, 18]:
        G.nodes[n]["image"] = images["genome_00-22-2-7"]
    for n in [22]:
        G.nodes[n]["image"] = images["genome_ancestral"]
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        if n in [2, 4, 6, 7, 13, 21]:
            n_loc = "right"
        else:
            n_loc = "center"
        a.set_title(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[n], verticalalignment="top", loc=n_loc, fontsize="x-large")
        a.axis("off")

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
