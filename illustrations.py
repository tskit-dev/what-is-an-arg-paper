import click
import argutils
import matplotlib as mpl
import matplotlib.pyplot as plt

import io
import networkx as nx
import PIL
import string


@click.group()
def cli():
    pass


@click.command()
def arg_edge_annotations():
    """
    Ancestry propagation via WH99 graph.
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

    ax1.set_title("(a) eARG with implicit\nencoding (Wiuf & Hein)", fontsize="xx-large")
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

    ax2.set_title(
        "(b) explicit encoding\n(i.e. non-ancestral removed)", fontsize="xx-large"
    )
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
        "genome_empty": "illustrations/node_icons/genome_empty.png",
        "genome_empty_hamburger": "illustrations/node_icons/genome_empty_hamburger.png",
        "genome_full": "illustrations/node_icons/genome_full.png",
    }
    for letter in list(string.ascii_uppercase[3 : G.number_of_nodes()]):
        icons["genome_" + letter] = "illustrations/node_icons/genome_" + letter + ".png"
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
    outfile = "ARG_edge_annotations"
    with open(f"illustrations/{outfile}.svg", "wt") as f:
        f.write(top_svg)


@click.command()
def arg_node_simplification():
    """
    Sequentially simplifying a WF simulation.
    """
    seed = 372
    t_x = {
        9: 3,
        6: -8,
        18: 50,
        19: -15,
        17: 5,
        13: 10,
        14: 2,
        10: -10,
        15: 5,
        16: -5,
        7: -5,
        11: 5,
        8: 8,
    }

    ts = argutils.sim_wright_fisher(2, 10, 100, recomb_proba=0.1, seed=seed)
    labels = {i: string.ascii_uppercase[i] for i in range(len(string.ascii_uppercase))}
    # relabel the nodes to get samples reading A B C D
    labels.update({2: "B", 3: "C", 1: "D"})
    ts = argutils.viz.label_nodes(ts, labels)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))
    ax1.set_title("(a) little ARG")
    ax2.set_title("(b) remove pass through")
    ax3.set_title("(c) simplify (keep unary in coal)")
    ax4.set_title("(d) fully simplified")
    col = mpl.colors.to_hex(plt.cm.tab20(1))
    pos, G = argutils.viz.draw(
        ts,
        ax1,
        draw_edge_widths=True,
        use_ranked_times=True,
        node_color=col,
        tweak_x=t_x,
    )
    ts2, node_map = argutils.simplify_remove_pass_through(
        ts, repeat=True, map_nodes=True
    )
    argutils.viz.draw(
        ts2,
        ax2,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_color=col,
    )
    ts3, node_map = argutils.simplify_keeping_unary_in_coal(ts, map_nodes=True)
    argutils.viz.draw(
        ts3,
        ax3,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_color=col,
    )
    ts4, node_map = ts.simplify(map_nodes=True)
    argutils.viz.draw(
        ts4,
        ax4,
        draw_edge_widths=True,
        pos={node_map[i]: p for i, p in pos.items()},
        node_color=col,
    )

    graph1_io = io.StringIO()
    plt.savefig(graph1_io, format="svg", bbox_inches="tight")
    graph1_svg = graph1_io.getvalue()
    graph1_svg = graph1_svg[graph1_svg.find("<svg") :]

    tree_svg = ts.draw_svg(
        size=(970, 250),
        time_scale="rank",
        node_labels=labels,
        style=".x-axis .tick .lab {font-weight: regular; font-size: 12}",
    )

    # figlabelstyle = 'font-family="serif" font-size="30px"'

    svg = [
        '<svg width="900" height="700" xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">',
        "<style>.tree-sequence text {font-family: sans-serif}</style>"
        '<g transform="translate(10, 50)">',
    ]
    svg.append('<g transform="scale(0.87)">' + graph1_svg + "</g>")
    svg.append('<g transform="translate(0 400) scale(0.83)">' + tree_svg + "</g>")
    svg.append("</g></svg>")

    outfile = "ARG_recomb_node_deletion"

    top_svg = (
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
        '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
    )
    top_svg += "\n".join(svg)
    with open(f"illustrations/{outfile}.svg", "wt") as f:
        f.write(top_svg)


cli.add_command(arg_edge_annotations)
cli.add_command(arg_node_simplification)

if __name__ == "__main__":
    cli()