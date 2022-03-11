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
import numpy as np
import scipy.stats
import tskit
import tsinfer

import argutils


@click.group()
def cli():
    pass


@click.command()
def ancestry_resolution():
    """
    Ancestry resolution on the WH99 graph.
    """

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
            if edge.left == 4 and edge.right == 6:  # Hack for this ts
                return False
            return True

        for halign, func1 in {
            "center": edge_not_above_recombinant,
            "right": left_recombinant_edge,
            "left": right_recombinant_edge,
        }.items():
            for full_edge, func2 in {
                True: lambda e: e.left == 0 and e.right == 7,
                False: lambda e: not (e.left == 0 and e.right == 7),
            }.items():
                nx.draw_networkx_edge_labels(
                    G,
                    pos=pos,
                    ax=ax,
                    rotate=False,
                    font_weight="normal" if full_edge else "bold",
                    alpha=0.5 if full_edge else None,
                    font_size=16,
                    edge_labels={
                        (e.child, e.parent): f"({e.left:.0f},{e.right:.0f}]"
                        for e in ts.edges()
                        if func1(e) and func2(e)
                    },
                    horizontalalignment=halign,
                    bbox=dict(
                        boxstyle="round,pad=0.05",
                        ec=(1.0, 1.0, 1.0),
                        fc=(1.0, 1.0, 1.0),
                    ),
                )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
    fig.tight_layout()

    ax1.set_title("(a)", fontsize=32, family="serif")
    ts = argutils.viz.label_nodes(argutils.wh99_example())
    pos, G = argutils.viz.draw(
        ts,
        ax1,
        use_ranked_times=True,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=100,
        max_edge_width=2,
        font_size=14,
        tweak_x={
            0: 10,
            3: 29,
            4: 2.5,
            7: 20,
            11: 9,
            12: -17,
            20: -22.5,
            19: 3.5,
            21: -11,
            17: -25,
            13: -16,
            8: 2,
            9: -24,
            5: -16,
            6: -41,
            15: 0.6,
            16: -25.6,
            2: -10,
        },
        tweak_y={22: 1, 17: 0.8},
    )
    add_edge_labels(ax1, ts, G, pos)

    ax2.set_title("(b)", fontsize=32, family="serif")
    ts2 = argutils.simplify_keeping_all_nodes(ts)
    pos, G = argutils.viz.draw(
        ts2,
        ax2,
        pos=pos,
        node_color=mpl.colors.to_hex(plt.cm.tab20(1)),
        node_size=200,
        font_size=14,
        # arrows=True,
        draw_edge_widths=True,
    )
    add_edge_labels(ax2, ts2, G, pos)

    # From https://networkx.org/documentation/stable/auto_examples/drawing/plot_custom_node_icons.html
    icons = {
        "genome_empty": "illustrations/assets/genome_empty.png",
        "genome_empty_hamburger": "illustrations/assets/genome_empty_hamburger.png",
        "genome_full": "illustrations/assets/genome_full.png",
    }
    for letter in list(string.ascii_uppercase[3 : G.number_of_nodes()]):
        icons["genome_" + letter] = "illustrations/assets/genome_" + letter + ".png"
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
        a.set_title(
            string.ascii_uppercase[n],
            y=0,
            verticalalignment="bottom",
            loc="center",
            fontsize="xx-large",
        )
        a.axis("off")

    # Panel (b)
    tr_figure = ax2.transData.transform
    tr_axes = fig.transFigure.inverted().transform
    icon_size = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.0003
    icon_center = icon_size / 2.0
    for n in [0, 1, 2]:
        G.nodes[n]["image"] = images["genome_full"]
    for n in range(3, G.number_of_nodes()):
        G.nodes[n]["image"] = images["genome_" + string.ascii_uppercase[n]]
    G.nodes[9]["image"] = images["genome_empty_hamburger"]
    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        if n in [2, 4, 6, 7, 13, 21]:
            n_loc = "right"
        else:
            n_loc = "center"
        a.set_title(
            string.ascii_uppercase[n],
            verticalalignment="top",
            loc=n_loc,
            fontsize="x-large",
        )
        a.axis("off")

    graph_io = io.StringIO()
    plt.savefig(graph_io, format="svg", bbox_inches="tight")
    graph_svg = graph_io.getvalue()

    svg = [
        # Could caoncatenate more SVG stuff here in <g> tags, e.g.
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


@click.command()
def simplification():
    """
    Sequentially simplifying a WF simulation.
    """
    seed = 372
    t_x = {
        9: 3,
        6: -15,
        18: 15,
        19: -15,
        17: 5,
        13: 30,
        14: -20,
        10: -10,
        15: 5,
        16: -5,
        7: -5,
        11: 5,
        8: 8,
    }

    ts = argutils.sim_wright_fisher(2, 10, 100, recomb_proba=0.1, seed=seed)
    tables = ts.dump_tables()
    rank_times = scipy.stats.rankdata(tables.nodes.time, method="dense")
    # tweak rank times here for nicer viz
    rank_times = np.where(rank_times == 1, 0, rank_times)
    tables.nodes.time = rank_times
    node_order = np.arange(tables.nodes.num_rows)
    # NB this isn't quite right
    node_order[[0, 1, 2, 3]] = [2, 1, 3, 0]
    tables.subset(node_order)
    tables.sort()
    ts=tables.tree_sequence()

    labels = {i: string.ascii_uppercase[i] for i in range(len(string.ascii_uppercase))}
    # relabel the nodes to get samples reading A B C D
    labels.update({0: "B", 2: "A", 3: "C", 1: "D", 4: "F", 5: "E", 13: "O", 14: "N", 7: "I", 8: "H", 10: "L", 11: "K"})
    #labels = {i: i for i in range(26)}
    ts1 = argutils.viz.label_nodes(ts, labels)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(3, 12))
    col = mpl.colors.to_hex(plt.cm.tab20(1))
    pos, G = argutils.viz.draw(
        ts1,
        ax1,
        draw_edge_widths=True,
        use_ranked_times=False,
        node_color=col,
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
        node_color=col,
    )
    ts3, node_map = argutils.simplify_keeping_unary_in_coal(ts1, map_nodes=True)
    argutils.viz.draw(
        ts3,
        ax3,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_color=col,
    )
    ts4, node_map = ts1.simplify(map_nodes=True)
    argutils.viz.draw(
        ts4,
        ax4,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_color=col,
    )

    graph1_io = io.StringIO()
    plt.savefig(graph1_io, format="svg", bbox_inches="tight")
    plt.close()
    graph1_svg = graph1_io.getvalue()
    graph1_svg = graph1_svg[graph1_svg.find("<svg") :]

    svg = [
        '<svg width="900" height="900" xmlns="http://www.w3.org/2000/svg" '
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
                '.x-axis .tick .lab {font-weight: regular; font-size: 12; visibility: hidden} '
                '.x-axis .tick:first-child .lab, .x-axis .tick:last-child .lab {visibility: visible}'
            ),
        )
        svg.append(
            f'<text font-size="2em" font-family="serif" transform="translate(0, {200 * i + 30})">' +
            f'({string.ascii_lowercase[i]})</text>'
        )
        svg.append(f'<g transform="translate(250 {205 * i}) scale(0.83)">' + tree_svg + "</g>")
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
        edge_labels={(e.child, e.parent): f"({e.left:.0f},{e.right:.0f}]" for e in ts.edges()}
        full_edges = {k: v for k, v in edge_labels.items() if not argutils.is_recombinant(ts.node(k[0]).flags)}
        nx.draw_networkx_edge_labels(
            **params, font_weight="normal", alpha=0.5, edge_labels=full_edges, label_pos=0.6)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.A and k[1]==n.E}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.35)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.A and k[1]==n.F}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.3)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.C and k[1]==n.E}
        nx.draw_networkx_edge_labels(
            **params, font_weight="bold", edge_labels=lab, label_pos=0.7)
        lab = {k: v for k, v in edge_labels.items() if k[0]==n.C and k[1]==n.F}
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
                label = string.ascii_uppercase[tables.nodes.num_rows]
                metadata = {
                    "gender": "male" if individual == 0 else "female",
                    "name": label,
                    "genome": "paternal" if genome == 0 else "maternal",
                }
                setattr(n, label, tables.nodes.num_rows)
                flags = 0
                if gen == 0:
                    flags |= tskit.NODE_IS_SAMPLE
                if label == "A" or label == "C":
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
    tables.edges.add_row(child=n.A, parent=n.E, left=0, right=bp[0])
    tables.edges.add_row(child=n.A, parent=n.F, left=bp[0], right=l)
    tables.edges.add_row(child=n.B, parent=n.G, left=0, right=l)

    tables.individuals[1] = tables.individuals[1].replace(parents=[2, 3])
    tables.edges.add_row(child=n.C, parent=n.F, left=0, right=bp[1])
    tables.edges.add_row(child=n.C, parent=n.E, left=bp[1], right=l)
    tables.edges.add_row(child=n.D, parent=n.H, left=0, right=l)

    tables.individuals[2] = tables.individuals[2].replace(parents=[4, 5])
    tables.edges.add_row(child=n.E, parent=n.I, left=0, right=l)
    tables.edges.add_row(child=n.F, parent=n.K, left=0, right=l)

    tables.individuals[3] = tables.individuals[3].replace(parents=[4, 5])
    tables.edges.add_row(child=n.G, parent=n.I, left=0, right=l)
    tables.edges.add_row(child=n.H, parent=n.K, left=0, right=l)

    tables.individuals[4] = tables.individuals[4].replace(parents=[6, 7])
    tables.edges.add_row(child=n.I, parent=n.N, left=0, right=l)

    tables.individuals[5] = tables.individuals[5].replace(parents=[6, 7])
    tables.edges.add_row(child=n.K, parent=n.N, left=0, right=l)

    tables.sort()
    ts = tables.tree_sequence()

    ts_used = ts  # argutils.remove_unused_nodes(ts)  # use this to remove unused nodes

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 5))
    col = mpl.colors.to_hex(plt.cm.tab20(1))
    pos, G = argutils.viz.draw(
        ts_used, ax,
        reverse_x_axis=True,
        draw_edge_widths=False,
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
        size=(500, 500), node_labels={n.id: n.metadata["name"] for n in ts_simp.nodes()}
    )

    svg = [
        '<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        "<style>.tree-sequence text {font-family: sans-serif}</style>"
        '<text font-size="2em" font-family="serif" transform="translate(80, 30)">'
        "(a)</text>",
        '<text font-size="2em" font-family="serif" transform="translate(380, 30)">'
        "(b)</text>",
        '<text font-size="2em" font-family="serif" transform="translate(780, 30)">'
        "(c)</text>",
        '<g transform="translate(10, 60)">',
    ]
    svg.append('<g transform="scale(0.36)">' + pedigree_svg + "</g>")
    svg.append('<g transform="translate(240 -10) scale(0.83)">' + pedigree_ARG + "</g>")
    svg.append('<g transform="translate(580) scale(0.83)">' + pedigree_ts + "</g>")
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
    tree_seqs = {}
    # KwARG
    with open("examples/kreitman.kwarg") as f:
        ts = argutils.convert_kwarg(f, 11, 43)
        # ts = argutils.simplify_keeping_unary_in_coal(ts)  # in case we want to compare with tsinfer
        ts = argutils.viz.label_nodes(ts)
        tree_seqs["KwARG"] = ts

    # Tsinfer
    with open("examples/kreitman_snp.txt") as file:
        data = np.array([[int(d) for d in line.strip()] for line in file])
    with tsinfer.SampleData(sequence_length=data.shape[1]) as sample_data:
        for pos, sites in enumerate(data.T):
            sample_data.add_site(pos, sites)
    ts = tsinfer.infer(sample_data)
    # For the moment, tsinfer uses byte metadata, so we need to convert it to json
    tables = ts.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for n in ts.nodes():
        tables.nodes[n.id] = tables.nodes[n.id].replace(
            metadata = json.loads(n.metadata.decode() or "{}"))
    # temporary hack: remove the ultimate ancestor if it exists
    oldest_node = np.argmax(tables.nodes.time)
    if np.sum(tables.edges.parent==oldest_node) == 1:
        # only a single edge connects to the root. This is a unary "ultimate ancestor"
        # and can be removed (it will be removed in later tsinfer versions anyway)
        use = np.arange(tables.nodes.num_rows)
        use = use[use != oldest_node]
        tables.subset(use)
    ts = tables.tree_sequence()
    tree_seqs["Tsinfer"] = argutils.viz.label_nodes(ts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    col = mpl.colors.to_hex(plt.cm.tab20(1))
    for ax, (name, ts) in zip(axes, tree_seqs.items()):
        pos, G = argutils.viz.draw(
            ts, ax,
            nonsample_node_shrink=5,
            use_ranked_times=True if name == "Tsinfer" else None,
            draw_edge_widths=True,
            node_color=col,
            max_edge_width=2)
        ax.set_title(name)

    graph_io = io.StringIO()
    plt.savefig(graph_io, format="svg", bbox_inches="tight")
    graph_svg = graph_io.getvalue()
    plt.close()

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
    with open(f"illustrations/inference.svg", "wt") as f:
        f.write(top_svg)
     

cli.add_command(arg_in_pedigree)
cli.add_command(ancestry_resolution)
cli.add_command(simplification)
cli.add_command(inference)

if __name__ == "__main__":
    cli()
