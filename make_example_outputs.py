"""
Run tools and convert output to tree sequences
"""

import json
import os
import subprocess

import tsinfer
import tskit
import argutils
import numpy as np
import click

@click.group()
def cli():
    pass


@click.command()
def run_tsinfer():
    sample_data = tsinfer.load("examples/Kreitman_SNP.samples")
    ts = tsinfer.infer(sample_data)
    # For the moment, tsinfer uses byte metadata, so we need to convert it to json
    # so that we can label the nodes easily
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
    ts.dump("examples/Kreitman_SNP_tsinfer.trees")


@click.command()
def run_kwarg():
    sample_data = tsinfer.load("examples/Kreitman_SNP.samples") # to get the sample names
    # TODO - run kwarg on Kreitman_SNP.matrix here to create examples/kreitman.kwarg
    with open("examples/kreitman.kwarg") as f:
        ts = argutils.convert_kwarg(
            f,
            11,
            43,
            sample_names={
                s.id: sample_data.individual(s.individual).metadata["name"]
                for s in sample_data.samples()
            },
        )
        # Label the samples
        ts.dump("examples/Kreitman_SNP_kwarg.trees")

@click.command()
def run_argweaver():
    os.makedirs("examples/argweaver_output", exist_ok=True)
    subprocess.run([
        "tools/argweaver/bin/arg-sample",
        "--sites", "examples/Kreitman_SNP.sites",
        "--output", "examples/argweaver_output/arg-sample",
        "--overwrite",
        "--smc-prime",
        "--popsize", "1e6",
        "--mutrate", "5.49e-09",  # From stdpopsim
        "--recombrate", "8.4e-09",  # From stdpopsim
        "--randseed", "111",
        "--iters", "3",
        "--sample-step", "10000",
        "--no-compress-output",
    ])
    subprocess.run([
        "python2",
        "tools/argweaver/bin/smc2arg",
        "examples/argweaver_output/arg-sample.0.smc",
        "examples/argweaver_output/arg-sample.0.arg",
    ])
    with open("examples/argweaver_output/arg-sample.0.arg") as f:
        ts = argutils.convert_argweaver(f)
        ts.dump("examples/Kreitman_SNP_argweaver.trees")


cli.add_command(run_tsinfer)
cli.add_command(run_kwarg)
cli.add_command(run_argweaver)

if __name__ == "__main__":
    cli()
