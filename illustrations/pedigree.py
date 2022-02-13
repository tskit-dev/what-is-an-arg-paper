import json
from pathlib import Path
import string
import sys
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt

import tskit

# useful modules in the top level dir
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import argutils

n = SimpleNamespace() # convenience labels
l=100
tables = tskit.TableCollection(sequence_length=l)
tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
for gen in range(4):
    for individual in range(2):
        i = tables.individuals.add_row()
        for genome in range(2):
            label = string.ascii_uppercase[tables.nodes.num_rows]
            metadata = {
                'gender': "male" if individual==0 else "female",
                'name': label,
                'genome': 'paternal' if genome==0 else "maternal",
            }
            setattr(n, label, tables.nodes.num_rows)
            tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE if gen==0 else 0,
                time=gen,
                metadata=metadata,
                individual=i,
            )

bp = [25, 60]
tables.edges.clear()
tables.individuals[0] = tables.individuals[0].replace(parents=[2,3])
tables.edges.add_row(child=n.A, parent=n.E, left=0, right=bp[0])
tables.edges.add_row(child=n.A, parent=n.F, left=bp[0], right=l)
tables.edges.add_row(child=n.B, parent=n.G, left=0, right=l)

tables.individuals[1] = tables.individuals[1].replace(parents=[2,3])
tables.edges.add_row(child=n.C, parent=n.F, left=0, right=bp[1])
tables.edges.add_row(child=n.C, parent=n.E, left=bp[1], right=l)
tables.edges.add_row(child=n.D, parent=n.H, left=0, right=l)

tables.individuals[2] = tables.individuals[2].replace(parents=[4,5])
tables.edges.add_row(child=n.E, parent=n.I, left=0, right=l)
tables.edges.add_row(child=n.F, parent=n.K, left=0, right=l)

tables.individuals[3] = tables.individuals[3].replace(parents=[4,5])
tables.edges.add_row(child=n.G, parent=n.I, left=0, right=l)
tables.edges.add_row(child=n.H, parent=n.K, left=0, right=l)

tables.individuals[4] = tables.individuals[4].replace(parents=[6,7])
tables.edges.add_row(child=n.I, parent=n.N, left=0, right=l)

tables.individuals[5] = tables.individuals[5].replace(parents=[6,7])
tables.edges.add_row(child=n.K, parent=n.N, left=0, right=l)

tables.sort()
ts = tables.tree_sequence()

ts_simp = ts.simplify(keep_unary=True)
fig, ax = plt.subplots(1, 1, figsize=(2, 4))
pos = argutils.viz.draw(
    ts_simp,
    ax,
    node_color=mpl.colors.to_hex(plt.cm.tab20(1))
)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(Path(__file__).parent / "pedigree_ARG.pdf")
with open(Path(__file__).parent / "pedigree_trees.svg", "wt") as file:
    file.write(ts_simp.draw_svg(
        size=(400, 400),
        node_labels = {n.id: n.metadata["name"] for n in ts_simp.nodes()})
    )


