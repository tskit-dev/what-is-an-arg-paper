"""
Run tools and convert output to tree sequences
"""

import json
import os
import subprocess

import tsinfer
import tsdate
import tskit
import argutils
import numpy as np
import click

mu = 5.49e-09  # From stdpopsim
rho = 8.4e-09  # From stdpopsim
Ne = 1e6  # Random guess

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
    ts = tsdate.date(ts, mutation_rate = mu, Ne = Ne)
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
        "--popsize", str(Ne),
        "--mutrate", str(mu),
        "--recombrate", str(rho),
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


@click.command()
def run_relate():
    sample_data = tsinfer.load("examples/Kreitman_SNP.samples")  # just for the seq len
    dir = "examples/Relate_output/"
    outfiles = "Kreitman_SNP"
    os.makedirs(dir, exist_ok=True)
    map_name = "Kreitman_SNP.map"
    with open(f"examples/{map_name}", "wt") as file:
        cM_per_MB = rho * 1e8
        print("pos", "COMBINED_rate", "Genetic_Map", sep=" ", file=file)
        print(0, f"{cM_per_MB:.5f}", 0, sep=" ", file=file)
        print(
            int(sample_data.sequence_length),
            f"{cM_per_MB:.5f}",
            sample_data.sequence_length / 1e6 * cM_per_MB,
            sep=" ",
            file=file)
    subprocess.run(
        [
            "../../tools/relate/bin/Relate",
            "--mode", "All",
            "-m", str(mu),
            "-N", str(Ne),
            "--haps", "../Kreitman_SNP.haps",
            "--sample", "../Kreitman_SNP.sample",
            "--map", f"../{map_name}",
            "--seed",  "111",
            "-o", outfiles,
        ],
        cwd=dir,
    )

    # Convert to JBOT tree sequence format
    subprocess.run([
        "tools/relate/bin/RelateFileFormats",
        "--mode", "ConvertToTreeSequence",
        "-i", f"{dir}{outfiles}",
        "-o", f"examples/Kreitman_SNP_relate_jbot"
    ])
    
    # Convert to time-uncalibrated tree sequence format
    #with open("examples/argweaver_output/arg-sample.0.arg") as f:
    #ts = argutils.convert_relate_without_times(outfiles
    
    
cli.add_command(run_tsinfer)
cli.add_command(run_kwarg)
cli.add_command(run_argweaver)
cli.add_command(run_relate)

if __name__ == "__main__":
    cli()
