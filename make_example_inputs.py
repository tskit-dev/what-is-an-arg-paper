import click
import tsinfer
import numpy as np

@click.group()
def cli():
    with tsinfer.SampleData(sequence_length=2731) as sd:
        # TODO - add provenance, citing the Kreitman paper and the chr (2L) and alignment
        # used and saying that this only includes the 43 SNPs and not the 6 indels
        # sd.add_provenance("2022-03-11T16:00:00+0000", )
        Fl = sd.add_population(metadata={"name": "West Palm Beach, Florida", "year": 1979})
        Wa = sd.add_population(metadata={"name": "Seattle, Washington State", "year": 1979})
        Af = sd.add_population(metadata={"name": "Burundi, Africa", "year": 1977})
        Fr = sd.add_population(metadata={"name": "Bully, France", "year": 1977})
        Ja = sd.add_population(metadata={"name": "Ishigaki, Japan", "year": 1978})
        sd.add_individual(ploidy=1, metadata={"name": "Wa-S"}, population=Wa)
        sd.add_individual(ploidy=1, metadata={"name": "Fl-1S"}, population=Fl)
        sd.add_individual(ploidy=1, metadata={"name": "Af-S"}, population=Af)
        sd.add_individual(ploidy=1, metadata={"name": "Fr-S"}, population=Fr)
        sd.add_individual(ploidy=1, metadata={"name": "Fl-2s"}, population=Fl)
        sd.add_individual(ploidy=1, metadata={"name": "Ja-S"}, population=Ja)
        sd.add_individual(ploidy=1, metadata={"name": "Fl-F"}, population=Fl)
        sd.add_individual(ploidy=1, metadata={"name": "Fr-F"}, population=Fr)
        sd.add_individual(ploidy=1, metadata={"name": "Wa-F"}, population=Wa)
        sd.add_individual(ploidy=1, metadata={"name": "Af-F"}, population=Af)
        sd.add_individual(ploidy=1, metadata={"name": "Ja-F"}, population=Ja)
        sd.add_site(61,   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], ["C", "T"])
        sd.add_site(62,   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], ["C", "G"])
        sd.add_site(63,   [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], ["G", "C"])
        sd.add_site(168,  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], ["C", "A"])
        sd.add_site(176,  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1], ["A", "G"])
        sd.add_site(206,  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["A", "G"])
        sd.add_site(231,  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["T", "G"])
        sd.add_site(236,  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["A", "G"])
        sd.add_site(238,  [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0], ["A", "T"])
        sd.add_site(350,  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["T", "G"])
        sd.add_site(356,  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], ["G", "T"])
        sd.add_site(367,  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0], ["G", "C"])
        sd.add_site(575,  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], ["C", "G"])
        sd.add_site(649,  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["G", "T"])
        sd.add_site(776,  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], ["C", "A"])
        sd.add_site(879,  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], ["T", "G"])
        sd.add_site(959,  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], ["A", "G"])
        sd.add_site(988,  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(1131, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["T", "C"])
        sd.add_site(1292, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["T", "C"])
        sd.add_site(1298, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["C", "A"])
        sd.add_site(1346, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["A", "C"])
        sd.add_site(1417, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["C", "G"])
        sd.add_site(1425, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["A", "G"])
        sd.add_site(1451, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ["A", "G"])
        sd.add_site(1463, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["T", "A"])
        sd.add_site(1468, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["A", "T"])
        sd.add_site(1488, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["A", "C"])
        sd.add_site(1494, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(1506, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], ["C", "G"])
        sd.add_site(1515, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(1553, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], ["A", "C"])
        sd.add_site(1581, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(1590, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], ["T", "C"])
        sd.add_site(1620, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], ["A", "C"])
        sd.add_site(1659, [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0], ["G", "A"])
        sd.add_site(1756, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], ["A", "C"])
        sd.add_site(1804, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], ["C", "G"])
        sd.add_site(1971, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ["A", "T"])
        sd.add_site(1997, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ["G", "A"])
        sd.add_site(2009, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(2202, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], ["C", "T"])
        sd.add_site(2419, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ["T", "A"])
    global sample_data
    sample_data = sd


@click.command()
def tsinfer_input():
    """
    Create a ".sample" file in the format required by tsinfer
    (see https://tsinfer.readthedocs.io/en/latest/tutorial.html)
    """
    saved_samples = sample_data.copy(path="examples/Kreitman_SNP.samples")
    saved_samples.finalise()

@click.command()
def kwarg_input():
    """
    Create a ".matrix" file in the format required by kwarg
    """
    # a simple 0/1 sites-by-samples matrix
    with open("examples/Kreitman_SNP.matrix", "wt") as file:
        for row in sample_data.sites_genotypes[:].T:
            print("".join(str(g) for g in row), file=file)


@click.command()
def argweaver_input():
    """
    Create a ".sites" file in the format required by ARGweaver
    """
    with open("examples/Kreitman_SNP.sites", "wt") as file:
        print(
            "NAMES",
            "\t".join(i.metadata["name"] for i in sample_data.individuals()),
            sep="\t",
            file=file,
        )
        print("REGION",
            "\t".join(["2L", str(1), str(int(sample_data.sequence_length) + 1)]),
            sep="\t",
            file=file,
        )
        for variant in sample_data.variants():
            print(
                int(variant.site.position),
                "".join(np.array(variant.alleles)[variant.genotypes]),
                sep="\t",
                file=file,
            )

@click.command()
def relate_input():
    """
    Create a ".haps" file and a ".sample" file in the format required by relate
    (see https://myersgroup.github.io/relate/input_data.html)
    """
    with open("examples/Kreitman_SNP.haps", "wt") as file:
        for v in sample_data.variants():
            assert len(v.alleles) == 2
            print(
                "2",
                f"SNP{v.site.id}",
                int(v.site.position),
                v.alleles[0],
                v.alleles[1],
                " ".join([str(g) for g in v.genotypes]),
                sep=" ",
                file=file,
            )
    with open("examples/Kreitman_SNP.sample", "wt") as file:
        print("ID_1 ID_2 missing", file=file)
        print("0    0    0", file=file)
        for i in sample_data.individuals():
            assert len(i.samples) == 1
            assert i.id == i.samples[0]
            print(f'{i.metadata["name"].replace("-", "")} NA 0', file=file)

cli.add_command(tsinfer_input)
cli.add_command(kwarg_input)
cli.add_command(argweaver_input)
cli.add_command(relate_input)

if __name__ == "__main__":
    cli()
