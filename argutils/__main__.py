import click
import argutils


@click.group()
def cli():
    pass


@click.command()
@click.argument("num_samples", type=int)
@click.argument("output", type=click.File("wb"))
@click.option("--rho", "-r", type=float, default=0)
@click.option("--sequence-length", "-L", type=int, default=10)
@click.option("--seed", "-s", type=int, default=42)
def simulate(num_samples, sequence_length, rho, seed, output):
    """
    Simulate an ARG under the coalescent with recombination and write
    out the explicitly resolved ancestry result to file.
    """
    ts = argutils.sim_arg(num_samples, rho, sequence_length, seed=seed)
    ts.dump(output)


@click.command()
def draw():
    click.echo("IMPLEMENT ME")


cli.add_command(simulate)
cli.add_command(draw)

if __name__ == "__main__":
    cli()
