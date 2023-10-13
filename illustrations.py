import pathlib
import string
import io
import json
from types import SimpleNamespace

import click
import networkx as nx
import PIL
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats
import tskit

import argutils


@click.group()
def cli():
    pass


@click.command()
def ancestry_resolution():
    """
    Ancestry resolution on the WH99 graph.
    """
    def plot_ellipse(ts, ax, xa, ya, burger_segments=None, trim_left=0, rel_height=1):
        # trim_left is a hack to not show stuff above the root
        icon_size1 = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.00029
        icon_center1 = icon_size1 / 2.0
        icon_size2 = icon_size1 * rel_height
        icon_center2 = icon_size2 / 2.0
        a = plt.axes([xa - icon_center1 - 0.005/2, ya - icon_center2+0.005/2, icon_size1 + 0.005, icon_size2-0.005])
        icon = patches.Ellipse(
            (0.5, 0.5),
            0.96,
            0.96,
            linewidth=2,
            edgecolor='tab:green',
            facecolor='white',
        )
        a.add_patch(icon)
        a.axis("off")
        if burger_segments is not None:
            icon_size1 = icon_size1 * 0.83
            icon_center1 = icon_size1 / 2.0
            icon_size2 = icon_size1 * 0.4
            icon_center2 = icon_size2 / 2.0
            a2 = plt.axes([xa - icon_center1 - 0.005/2, ya - icon_center2+0.005/2, icon_size1 + 0.005, icon_size2-0.005])
            for segment in burger_segments:
                icon = patches.Rectangle(
                    (segment[0].left/ts.sequence_length, 0),
                    segment[0].right/ts.sequence_length,
                    1,
                    facecolor=str(1-segment[1]/ts.num_samples),
                    )
                a2.add_patch(icon)
            a2.add_patch(patches.Rectangle(
                (trim_left, 0),
                1 - trim_left,
                1,
                edgecolor="k",
                facecolor='none',
                linewidth=2))
            a2.axis("off")
            
        return a

    def plot_rect(ax, xa, ya, scale=(1, 1), color="k"):
        icon_size1 = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.0002 * scale[0]
        icon_center1 = icon_size1 / 2.0
        icon_size2 = icon_size1 * 1.3 * scale[1]
        icon_center2 = icon_size2 / 2.0
        a = plt.axes([xa - icon_center1, ya - icon_center2, icon_size1, icon_size2])
        icon = patches.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            linewidth=4,
            edgecolor=color,
            facecolor='white',
        )
        a.add_patch(icon)
        a.axis("off")
        return a

    def add_edge_labels(ax, ts, G, pos):
        # Merge edge labels that have the same parent & child
        label_format = "[{0:.0f},{1:.0f})"
        full_edge = label_format.format(0, ts.sequence_length)
        lab = {}
        for e in ts.edges():
            key = (e.child, e.parent)
            if key in lab:
                lab[key] += (" " + label_format.format(e.left, e.right))
            else:
                lab[key] = label_format.format(e.left, e.right)

        for full in [True, False]:
            nx.draw_networkx_edge_labels(
                G,
                pos=pos,
                ax=ax,
                rotate=False,
                font_weight="normal" if full else "bold",
                alpha=0.5 if full else None,
                font_size=24,
                edge_labels={k: l for k, l in lab.items() if (l == full_edge) is full},
                horizontalalignment="center",
                bbox=dict(
                    boxstyle="round,pad=0.05",
                    ec=(1.0, 1.0, 1.0),
                    fc=(1.0, 1.0, 1.0),
                ),
            )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    fig.tight_layout()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0.2)
    plt.margins(0,-0.04)

    ax1.set_title("A", fontsize=32, family="serif", loc="center")
    ts = argutils.viz.label_nodes(argutils.wh99_example(one_node_recombination=True))
    pos, G = argutils.viz.draw(
        ts,
        ax1,
        tweak_x={7: 15, 3: -12, 4: -8, 6: -6},
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=100,
        max_edge_width=2,
        font_size=12,
    )

    ax2.set_title("B", fontsize=32, family="serif", loc="center")
    pos, G = argutils.viz.draw(
        ts,
        ax2,
        pos=pos,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=100,
        max_edge_width=2,
        font_size=14,
    )
    add_edge_labels(ax2, ts, G, pos)

    ax3.set_title("C", fontsize=32, family="serif", loc="center")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        style=(0, (5, 10)),
        edgelist=list(G.edges()),
        arrowstyle="-",
        ax=ax3,
        node_size=200,
        width=1,
    )

    ts_simp = argutils.remove_edges_above_local_roots(ts).simplify(filter_nodes=False, keep_unary=True)
    pos, G = argutils.viz.draw(
        ts_simp,
        ax3,
        pos=pos,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=200,
        font_size=14,
        # arrows=True,
        draw_edge_widths=True,
    )
    add_edge_labels(ax3, ts_simp, G, pos)


    # Panel (a)
    tr_figure = ax1.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    edges = ts.tables.edges    
    for n in G.nodes:
        if G.nodes[n]["flags"] & argutils.ancestry.NODE_IS_RECOMB:
            col = 'tab:red'
            size = (1.2, 1.2)
            breaks = (edges.left[edges.child == n], edges.right[edges.child == n])
            breaks = np.unique(np.concatenate(breaks))
            assert len(breaks == 3)
            assert breaks[0] == 0
            assert breaks[2] == ts.sequence_length
            breakpoint = breaks[1]
        else:
            col = 'k' if ts.node(n).is_sample() else 'tab:blue'
            size = (0.5, 1)
            breakpoint = None
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plot_rect(ax1, xa, ya, size, col)
        if breakpoint is not None:
            a.text(
                0.4, 0.2,
                str(int(breakpoint)),
                fontname="Trebuchet MS",
                fontsize=40,
            )

    # Panel (b)
    tr_figure = ax2.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plot_ellipse(ts, ax2, xa, ya)
        a.set_title(
            G.nodes[n]["label"],
            y=0.05,
            fontname="Trebuchet MS",
            verticalalignment="bottom",
            loc="center",
            fontsize=24,
        )

    # Panel (c)
    # Find the number of samples under each region for each node
    node_samples = {n.id:[] for n in ts.nodes()}
    for tree in ts_simp.trees():
        intvl = tree.interval
        for u in tree.nodes():
            n_samples = tree.num_samples(u)
            s = node_samples[u]
            if len(s) > 0 and s[-1][0].right == intvl.left and s[-1][1] == n_samples:
                s[-1] = (tskit.Interval(s[-1][0].left, intvl.right), n_samples)
            else:
                s.append((intvl, n_samples))
    
    
    tr_figure = ax3.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plot_ellipse(
            ts_simp,
            ax3,
            xa,
            ya,
            None if n == 16 else node_samples[n],
            2 if n in {13, 14, 15} else 0,
            rel_height=1.25)
        a.set_title(
            G.nodes[n]["label"],
            fontname="Trebuchet MS",
            verticalalignment="bottom",
            y=0.45,
            loc="center",
            fontsize="x-large",
        )
    graph_io = io.StringIO()
    plt.savefig(graph_io, format="svg", bbox_inches="tight", pad_inches=0)
    graph_svg = graph_io.getvalue()

    svg = [
        # Could concatenate more SVG stuff here in <g> tags, e.g.
        # if we wanted to draw the 2 plots as 2 separate svg plots\
        # rather than using plt.subplots
        graph_svg[graph_svg.find("<svg") :]
    ]

    top_svg = (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    )
    top_svg += "\n".join(svg)
    with open(f"illustrations/ancestry-resolution.svg", "wt") as f:
        f.write(top_svg)


def legend_svg():
    # Save legend as a separate SVG
    marker_params = dict(marker='o', linestyle=':', markersize=15, markeredgewidth=4)

    fig, ax = plt.subplots(figsize=(4.3,3))
    label_sep=0.6
    ax.text(2.5, 0.9, "Number of\nparents", fontsize=18, ha="center", linespacing=0.8)
    ax.text(8, 0.9, "Coalescent\nspan", fontsize=18, ha="center", linespacing=0.8)
    
    for parents in range(5):
        ax.plot(
            [1],
            [-parents],
            markeredgecolor=argutils.viz.arity_colors(parents),
            markerfacecolor='none',
            **marker_params
        )
        ax.text(
            1+label_sep,
            -parents,
            str(parents) + (" parents" if parents in (0, 4) else ""),
            va="center_baseline",
            fontsize=15,
        )
    # not quite right, but good enough
    mean_rgb = np.mean(np.array([argutils.viz.arity_colors(p)[:3] for p in range(5)]))
    
    for span in [100, 50, 0]:
        shade = 1-(span/100) * (1-mean_rgb)
        y = (span-100)/35
        ax.plot(
            [7],
            [y],
            markeredgecolor=[mean_rgb]*3,
            markerfacecolor=[shade] * 3,
            **marker_params,
        )
        ax.text(7+label_sep, y, str(span) + "%", va="center_baseline", fontsize=15)
        if span:
            ax.text(7+label_sep, y-0.5, "...", va="center_baseline", fontsize=15)
            
        
    ax.plot(
        [6],
        [-4],
        markeredgecolor=[mean_rgb]*3,
        markerfacecolor=[0] * 3,
        **marker_params,
    )
    ax.text(6+label_sep, -4, "sample", va="center_baseline", fontsize=15)
    
    ax.set_ylabel("Node key", fontsize=20, bbox=dict(facecolor='white'))
    ax.yaxis.set_label_coords(0.035, 0.5)
    ax.set_axisbelow(False)
    ax.set_xlim(-0.5,10.5)
    ax.set_ylim(-4.7,2.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)

    ax.patch.set_facecolor('white')
    fig.patch.set_alpha(0)
    legend_io = io.StringIO()
    plt.savefig(legend_io, format="svg", bbox_inches="tight")
    plt.close()
    legend_svg = legend_io.getvalue()
    return legend_svg[legend_svg.find("<svg") :]

def draw_simplification_stages(ax1, ax2, ax3, ax4):
    seed = 4517  # Chosen to give a diamond, 2 parents + 1 child node and a CA/noncoal
    t_x = {
        2: -30, 3: 30, 4: 15, 7: 10, 8: 4, 9: 20, 10: 10,
        11: 3, 12: 14, 13: 8, 15: 8, 16: 3.5, 17: 15, 18: 20,
    }

    ts = argutils.sim_wright_fisher(2, 10, 100, recomb_proba=0.1, seed=seed)
    tables = ts.dump_tables()
    rank_times = scipy.stats.rankdata(tables.nodes.time, method="dense")
    # tweak rank times here for nicer viz
    rank_times = np.where(rank_times == 1, 0.5, rank_times)
    tables.nodes.time = rank_times
    # reverse coordinates to give space for the key on the plot
    left = tables.edges.left
    tables.edges.left = tables.sequence_length - tables.edges.right
    tables.edges.right = tables.sequence_length - left
    tables.sort()
    ts=tables.tree_sequence()

    labels = {i: a for i, a in enumerate(string.ascii_lowercase)}
    # relabel the nodes to get samples reading A B C D
    labels.update({6: "e", 4: "g", 10: "j", 9: "k", 12: "l", 11: "m", 15: "o", 14: "p"})
    #labels = {i: i for i in range(26)}  # uncommment to get node IDs for re-positioning
    ts1 = argutils.viz.label_nodes(ts, labels)

    pos, G = argutils.viz.draw(
        ts1,
        ax1,
        draw_edge_widths=True,
        use_ranked_times=False,
        node_arity_colors=True,
        tweak_x=t_x,
    )
    ts2, node_map = argutils.simplify_remove_pass_through(
        ts1, repeat=True, map_nodes=True
    )
    argutils.viz.draw(
        ts2,
        ax2,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_arity_colors=True,
    )
    ts3, node_map = argutils.simplify_keeping_unary_in_coal(ts1, map_nodes=True)
    tables = ts3.dump_tables()
    argutils.viz.draw(
        ts3,
        ax3,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_arity_colors=True,
    )
    ts4, node_map = ts1.simplify(map_nodes=True)
    argutils.viz.draw(
        ts4,
        ax4,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_arity_colors=True,
    )
    return ts1, ts2, ts3, ts4

@click.command()
def simplification():
    """
    Sequentially simplifying a WF simulation.
    """

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(3, 12))
    ts1, ts2, ts3, ts4 = draw_simplification_stages(ax1, ax2, ax3, ax4)

    graph1_io = io.StringIO()
    plt.savefig(graph1_io, format="svg", bbox_inches="tight")
    plt.close()
    graph1_svg = graph1_io.getvalue()
    graph1_svg = graph1_svg[graph1_svg.find("<svg") :]    

    svg = [
        '<svg width="900" height="820" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        "<style>.tree-sequence text {font-family: sans-serif}</style>"
    ]
    svg.append('<g transform="translate(40, 0) scale(0.87)">' + graph1_svg + "</g>")

    for i, ts in enumerate([ts1, ts2, ts3, ts4]):
        tree_svg = ts.draw_svg(
            size=(750, 250),
            #time_scale="rank",
            node_labels={n.id: n.metadata["name"] for n in ts.nodes()},
            x_label=None if i == 3 else "",
            style=(
                #'.x-axis .tick .lab {font-weight: regular; font-size: 12; visibility: hidden} ' +
                '.x-axis .tick:first-child .lab, .x-axis .tick:last-child .lab {visibility: visible}' +
                '.subfig3 .x-axis .tick .lab {visibility: visible}'
            ),
        )
        svg.append(
            f'<text font-size="2em" font-family="serif" transform="translate(0, {200 * i + 30})">' +
            f'{string.ascii_uppercase[i]}</text>'
        )
        svg.append(
            f'<g class="subfig{i}" transform="translate(250 {205 * i}) scale(0.83)">' +
            tree_svg +
            "</g>"
        )
    svg.append('<g transform="translate(190, 580) scale(0.5)">' + legend_svg() + "</g>")
    svg.append("</svg>")


    # figlabelstyle = 'font-family="serif" font-size="30px"'


    top_svg = (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    )
    top_svg += "\n".join(svg)
    with open(f"illustrations/simplification.svg", "wt") as f:
        f.write(top_svg)


@click.command()
def arg_in_pedigree():
    """
    The ARG as embedded in diploid pedigree.
    """
    def add_edge_labels(ax, ts, G, pos, n):
        params = {
            "G": G,
            "pos": pos,
            "ax": ax,
            "rotate": False,
            "font_size": 12,
            "bbox": dict(
                boxstyle="round,pad=0.05",
                ec=(1.0, 1.0, 1.0),
                fc=(1.0, 1.0, 1.0)),
        }
        edge_labels={(e.child, e.parent): f"[{e.left:.0f},{e.right:.0f})" for e in ts.edges()}
        full_edges = {k: v for k, v in edge_labels.items() if not argutils.is_recombinant(ts.node(k[0]).flags)}
        nx.draw_networkx_edge_labels(
            **params, font_weight="normal", alpha=0.5, edge_labels=full_edges, label_pos=0.6)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.a and k[1]==n.e}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.35)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.a and k[1]==n.f}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.3)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.c and k[1]==n.e}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.7)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.c and k[1]==n.f}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.55)

    pedigree_svg = pathlib.Path("illustrations/assets/pedigree.svg").read_text()
    pedigree_svg = pedigree_svg[pedigree_svg.find("<svg") :]

    n = SimpleNamespace()  # convenience labels
    l = 10
    tables = tskit.TableCollection(sequence_length=l)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for gen in range(4):
        for individual in range(2):
            i = tables.individuals.add_row()
            for genome in range(2):
                label = string.ascii_lowercase[tables.nodes.num_rows]
                metadata = {
                    "gender": "male" if individual == 0 else "female",
                    "name": label,
                    "genome": "paternal" if genome == 0 else "maternal",
                }
                setattr(n, label, tables.nodes.num_rows)
                flags = 0
                if gen == 0:
                    flags |= tskit.NODE_IS_SAMPLE
                if label == "a" or label == "c":
                    flags |= argutils.NODE_IS_RECOMB
                tables.nodes.add_row(
                    flags=flags,
                    time=gen,
                    metadata=metadata,
                    individual=i,
                )

    bp = [2, 7]
    tables.edges.clear()
    tables.individuals[0] = tables.individuals[0].replace(parents=[2, 3])
    tables.edges.add_row(child=n.a, parent=n.e, left=0, right=bp[0])
    tables.edges.add_row(child=n.a, parent=n.f, left=bp[0], right=l)
    tables.edges.add_row(child=n.b, parent=n.g, left=0, right=l)

    tables.individuals[1] = tables.individuals[1].replace(parents=[2, 3])
    tables.edges.add_row(child=n.c, parent=n.f, left=0, right=bp[1])
    tables.edges.add_row(child=n.c, parent=n.e, left=bp[1], right=l)
    tables.edges.add_row(child=n.d, parent=n.h, left=0, right=l)

    tables.individuals[2] = tables.individuals[2].replace(parents=[4, 5])
    tables.edges.add_row(child=n.e, parent=n.i, left=0, right=l)
    tables.edges.add_row(child=n.f, parent=n.k, left=0, right=l)

    tables.individuals[3] = tables.individuals[3].replace(parents=[4, 5])
    tables.edges.add_row(child=n.g, parent=n.i, left=0, right=l)
    tables.edges.add_row(child=n.h, parent=n.k, left=0, right=l)

    tables.individuals[4] = tables.individuals[4].replace(parents=[6, 7])
    tables.edges.add_row(child=n.i, parent=n.n, left=0, right=l)

    tables.individuals[5] = tables.individuals[5].replace(parents=[6, 7])
    tables.edges.add_row(child=n.k, parent=n.n, left=0, right=l)

    tables.sort()
    ts = tables.tree_sequence()

    ts_used = ts  # argutils.remove_unused_nodes(ts)  # use this to remove unused nodes

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.35))
    col = mpl.colors.to_hex("lightgray")
    pos, G = argutils.viz.draw(
        ts_used, ax,
        reverse_x_axis=True,
        draw_edge_widths=False,
        node_size=500,
        font_size=17,
        node_color=col,
        tweak_x={l:(-13 if l< 12 else -20) for l in range(8,16)},
        max_edge_width=2)
    add_edge_labels(ax, ts_used, G, pos, n)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    with io.StringIO() as f:
        plt.savefig(f, format="svg")
        pedigree_ARG = f.getvalue()
        pedigree_ARG = pedigree_ARG[pedigree_ARG.find("<svg") :]
    plt.close(fig)
    ts_simp = ts_used.simplify(keep_unary=True)
    pedigree_ts = ts_simp.draw_svg(
        size=(350, 305),
        node_labels={n.id: n.metadata["name"] for n in ts_simp.nodes()},
        style=(
            ".x-axis .tick .lab {font-weight: normal; font-size:12px}"
            " .x-axis .title .lab {font-size:10px}"
        )
    )
    # Hack the labels, as inkscape conversion doens't like CSS transforms

    pedigree_ts = pedigree_ts.replace(
        '<line x1="0" x2="0" y1="0" y2="5" /><g transform="translate(0 6)">',
        '<line x1="0" x2="0" y1="0" y2="-5" /><g transform="translate(0 -17)">'
    )
    pedigree_ts = pedigree_ts.replace(
        '<text class="lab" text-anchor="middle" transform="translate(0 -11)">Genome position</text>',
        '<text class="lab" text-anchor="middle" transform="translate(0 -30)">Genome position</text>'
    )
    pedigree_ts = pedigree_ts.replace(
        '<text class="lab rgt" transform="translate(3 -7.0)">g</text>',
        '<text class="lab lft" transform="translate(-3.5 -7.0)">g</text>'
    )
    pedigree_ts = pedigree_ts.replace(
        '<text class="lab lft" transform="translate(-3 -7.0)">g</text>',
        '<text class="lab rgt" transform="translate(3.5 -7.0)">g</text>'
    )
   

    svg = [
        '<svg width="1100" height="435" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        "<style>.tree-sequence text {font-family: sans-serif}</style>"
        '<text font-size="2em" font-family="serif" transform="translate(100, 25)">'
        "A</text>",
        '<text font-size="2em" font-family="serif" transform="translate(380, 25)">'
        "B</text>",
        '<text font-size="2em" font-family="serif" transform="translate(830, 25)">'
        "C</text>",
        #'<text font-size="2em" font-family="serif" transform="translate(1250, 25)">'
        #"D</text>",
        '<g transform="translate(10, 40)">',
    ]
    svg.append('<g transform="scale(0.36)">' + pedigree_svg + "</g>")
    svg.append('<g transform="translate(240) scale(0.82)">' + pedigree_ARG + "</g>")
    svg.append('<g transform="translate(580 -10) scale(1.43)">' + pedigree_ts + "</g>")
    svg.append("</g></svg>")

    top_svg = (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    )
    top_svg += "\n".join(svg)
    with open("illustrations/arg-in-pedigree.svg", "wt") as f:
        f.write(top_svg)


@click.command()
def inference():
    """
    Examples of ARGs produced by various inference algorithms
    """
    def get_edge_colors(ts):
        """
        Return a dict of edge colours for nodes and a dict of edge colors for edges
        """
        # Sort edges by the age of the child node
        edges_argsorted = scipy.stats.rankdata(
            ts.tables.nodes.time[ts.tables.edges.child], method='dense') - 1
        edge_sorted_dict = {edge_id: sorted_edge_id
                            for edge_id, sorted_edge_id in enumerate(edges_argsorted)}
        # Create a dictionary w keys: (parent, child) of each edge; values: edge color
        edge_colors_by_nodes = {}
        edge_colors_by_id = {}
        # Colormap normalization to the number of edges in each tree sequence
        max_rank_by_ts[name] = np.max(edges_argsorted)
        norm = mpl.colors.Normalize(vmin=0, vmax=max_rank_by_ts[name])
        for key_edge, edge in enumerate(ts.edges()):
            color = argutils.viz.make_color(
                    edge_colormap(norm(edge_sorted_dict[key_edge])))
            edge_colors_by_nodes[edge.child, edge.parent] = color
            edge_colors_by_id[key_edge] = color
        return edge_colors_by_nodes, edge_colors_by_id


    tree_seqs = {}
    # Assign colormap for edges
    edge_colormap = plt.cm.viridis

    # KwARG
    ts = tskit.load("examples/Kreitman_SNP_kwarg.trees")
    # ts = argutils.simplify_keeping_unary_in_coal(ts)  # in case we want to compare with tsinfer
    labels = {n.id: n.metadata["name"] if n.is_sample() else "" for n in ts.nodes()}
    tree_seqs["KwARG"] = argutils.viz.label_nodes(ts, labels=labels)
    # ARGweaver
    ts = tskit.load("examples/Kreitman_SNP_argweaver.trees")
    labels = {n.id: n.metadata["name"] if n.is_sample() else "" for n in ts.nodes()}
    tree_seqs["ARGweaver"] = argutils.viz.label_nodes(ts, labels=labels)

    # Tsinfer
    ts = tskit.load("examples/Kreitman_SNP_tsinfer.trees")
    labels = {}
    for n in ts.nodes():
        labels[n.id] = ""
        if n.individual != tskit.NULL:
            ind_metadata = ts.individual(n.individual).metadata
            try:
                labels[n.id] = ind_metadata["name"]
            except TypeError:
                labels[n.id] = json.loads(ind_metadata.decode())["name"]
    tree_seqs["Tsinfer"] = argutils.viz.label_nodes(ts, labels=labels)

    # Relate JBOT (normally commented out)
    ts = tskit.load("examples/Kreitman_SNP_relate_jbot.trees")
    tables = ts.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for i, n in enumerate(tables.nodes):
        tables.nodes[i] = n.replace(metadata={})
    ts = tables.tree_sequence()
    # labels not stored by default in the Relate ts metadata, so use the previous ones
    labels = {n.id: labels[n.id] if n.is_sample() else "" for n in ts.nodes()}
    # tree_seqs["Relate JBOT"] = argutils.viz.label_nodes(ts, labels=labels)

    # Relate non JBOT
    ts = tskit.load("examples/Kreitman_SNP_relate_merged.trees")
    tables = ts.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for i, n in enumerate(tables.nodes):
        tables.nodes[i] = n.replace(metadata={})
    ts = tables.tree_sequence()
    # labels not stored by default in the Relate ts metadata, so use the previous ones
    labels = {n.id: labels[n.id] if n.is_sample() else "" for n in ts.nodes()}
    tree_seqs["Relate"] = argutils.viz.label_nodes(ts, labels=labels)

    widths = np.ones(len(tree_seqs))
    heights = [1, 0.2]
    fig, axes = plt.subplots(
        2, len(tree_seqs), figsize=(10, 6),
        gridspec_kw=dict(width_ratios=widths, height_ratios=heights),
        # autoscale_on=False
    )
    tree_seq_positions = []
    max_rank_by_ts = {}
    for lab, ax, ax_edges, name in zip("ABCD", axes[0], axes[1], tree_seqs.keys()):
        ts = tree_seqs[name]
        subtitle = f"{ts.num_trees} trees"
        params = dict(
            # some of these can get overwritten
            node_size=30,
            rotated_sample_labels=True,
            reverse_x_axis = False,
            draw_edge_widths=True,
            node_arity_colors=True,
            max_edge_width=2,
            tweak_x={},
            #use_ranked_times = False,  # By default use Y axis layout from graphviz dot
        )
        path = pathlib.Path(f"illustrations/assets/{name}.json")
        if path.is_file():
            # We have some positions from the tskit_arg_visualiser:
            # see https://github.com/kitchensjn/tskit_arg_visualizer/blob/main/plotting.md
            tskit_arg_visualizer_json = json.loads(path.read_text())
            params["pos"] = {}
            for n in tskit_arg_visualizer_json["arg"]["nodes"]:
                 params["pos"][n["id"]] = np.array([n["x"], -n["y"]])

        if name == "Tsinfer":
            params["tweak_x"] = {
            #    21: 12, 18: -12, 22: 15, 19: 5, 12: -3, 13: 2, 25: 5,
            #    23: 0, 11: 12, 23: 52, 5: 110, 25: 30, 20: 5, 24: 10, 17: -5
            }
        if name == "ARGweaver":
            #subtitle = "(typical MCMC result) " + subtitle
            params["tweak_x"] = {
                #54: 10, 48: 7, 49: 10, 26: 5, 30: 10, 32: -5, 31: 12, 15: -7, 40: 2,
                #44: -5, 39: -2, 45: -3, 52: -2, 51: -5, 42: 5,
            }
        if name == "KwARG":
            params["tweak_x"] = {
            #    18: 5, 17: 7, 16: 5, 13: -2, 20: -7, 19: -8, 11: -5, 12: -5, 35: -3,
            #    22: -5, 23: -5, 33: 10, 27: 5, 32: 8, 29: -2, 28: 2, 26: 5, 25: 3,
            }
        if name == "Relate JBOT":
            params["tweak_x"] = {
                40: 25, 35: 27, 38: 48, 31: 15, 34: 23, 37: 39, 36: 33, 33: 15, 32: 12,
                30: 35, 26: 5, 29: 25, 28: 25, 25: -5, 23: -10, 22: -15, 21: -2, 27: 25, 24: 20,
                20: -10, 15: -42, 11: -5, 16: -10, 13: -40, 18: -10, 19: -10,
            }

        if name == "Relate":
            params["reverse_x_axis"]=True
            params["tweak_x"] = {
                #20: -25, 30: 10, 19: -20, 24: 20, 15: 7, 13: 5, 12: 5, 28: 15, 27: -30,
                #6: -40, 0: 10, 1: 10, 2: 10, 3: 10, 14: 13, 11: 12, 25: 10,
                #28: 5, 26: 16, 17: -15, 16: 5, 21: -2, 23: 2, 29: 5
            }

        edge_colors_by_node, edge_colors_by_id = get_edge_colors(ts)
        pos, G = argutils.viz.draw(ts, ax, edge_colors=edge_colors_by_node, **params)
        ## Uncomment below to see equivalent simplified versions instead
        # ax.clear()
        # ts, node_map = ts.simplify(map_nodes=True)
        # pos = {node_map[k]:v for k, v in pos.items()}
        # edge_colors_by_node, edge_colors_by_id = get_edge_colors(ts)
        # params["tweak_x"] = None
        # params["reverse_x_axis"] = None
        # argutils.viz.draw(ts, ax, edge_colors=edge_colors_by_node, pos=pos, **params)
        ax.text(270, -15, lab, fontsize=20, family="serif")

        ax_edges.set_title(f"{name} ({subtitle})", y=1.05)

        # Add breakpoints and stacked edges below tree sequences, w same colormap as above
        ax_edges.axis("off")
        left_coord = 0.05
        width = 0.9

        edges = {}
        times = ts.tables.nodes.time
        for ((i, tree), (interval, edges_out, edges_in)) in zip(enumerate(
                ts.trees()), ts.edge_diffs()):

            relative_span = ((tree.interval.right - tree.interval.left) /
                             ts.get_sequence_length())

            for edge in edges_in:
                edges[edge.id] = (left_coord, -1, times[edge.child])
            for edge in edges_out:
                edges[edge.id] = (edges[edge.id][0], left_coord, edges[edge.id][2])

            ax_edges.axvline(left_coord, -0.2, 1.2, linestyle="--", linewidth=0.8)
            left_coord += relative_span * width
            ax_edges.axvline(left_coord, -0.2, 1.2, linestyle="--", linewidth=0.8)
        # Colormap normalization to the number of edges in each tree sequence (as above)
        norm = mpl.colors.Normalize(vmin=0, vmax=max_rank_by_ts[name])

        # Sort edges by age of child (as done above)
        edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1][2]))
        for index, (edge_id, edge) in enumerate(edges_sorted.items()):
            left = edge[0]
            right = edge[1]
            if right == -1:
                right = left_coord
            rect = patches.Rectangle(
                (left, (index / ts.num_edges)),
                (right - left),
                (1 / ts.num_edges), linewidth=1,
                facecolor=edge_colors_by_id[edge_id])
            ax_edges.add_patch(rect)

    graph_io = io.StringIO()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0)
    plt.savefig(graph_io, format="svg")
    graph_svg = graph_io.getvalue()
    plt.close()

    
    svg = [
        '<svg width="960" height="580" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
    ]
    svg.append('<g transform="">' + graph_svg[graph_svg.find("<svg"):] + "</g>")
    svg.append('<g transform="translate(150, 30) scale(0.5)">' + legend_svg() + "</g>")
    svg.append('</svg>')
    top_svg = (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    )
    top_svg += "\n".join(svg)
    with open(f"illustrations/inference.svg", "wt") as f:
        f.write(top_svg)

@click.command()
def cell_lines():
    with open(f"illustrations/cell-lines.svg", "wt") as f:
        f.write(pathlib.Path("illustrations/assets/cell-lines-full.svg").read_text())


cli.add_command(arg_in_pedigree)
cli.add_command(ancestry_resolution)
cli.add_command(simplification)
cli.add_command(cell_lines)
cli.add_command(inference)

if __name__ == "__main__":
    cli()