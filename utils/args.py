"""
Utilities for generating and converting ARGs in various formats.
"""
import random
import math
import dataclasses

import intervaltree
import tskit

NODE_IS_RECOMB = 1 << 1


@dataclasses.dataclass
class Lineage:
    node: int
    ancestry: intervaltree.IntervalTree()

    @property
    def num_recombination_links(self):
        return self.ancestry.span() - 1

    @property
    def left(self):
        """
        Returns the leftmost position of ancestral material.
        """
        return self.ancestry.begin()

    @property
    def right(self):
        """
        Returns the rightmost position of ancestral material.
        """
        return self.ancestry.end()


def arg_sim(n, rho, L, seed=None):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    for _ in range(n):
        node = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        lineages.append(Lineage(
            node, intervaltree.IntervalTree.from_tuples([(0, L)])))

    t = 0
    while len(lineages) > 0:
        # print(lineages)
        lineage_links = [lineage.num_recombination_links for lineage in lineages]
        total_links = sum(lineage_links)
        re_rate = sum(lineage_links) * rho
        t_re = math.inf if re_rate == 0 else rng.expovariate(re_rate)
        k = len(lineages)
        print("t = ", t, "k=", k, "lineage_links", total_links)
        ca_rate = k * (k - 1) / 2
        t_ca = rng.expovariate(ca_rate)
        t_inc = min(t_re, t_ca)
        t += t_inc
        if t_inc == t_re:
            # Choose a lineage to recombine with probability equal to the
            # number of recombination links it subtends.
            index = rng.choices(range(k), weights=lineage_links)[0]
            lineage = lineages.pop(index)

            # Choose a breakpoint uniformly on that lineage
            breakpoint = rng.randrange(lineage.left + 1, lineage.right)
            assert lineage.left < breakpoint < lineage.right
            print("RE", breakpoint, lineage)
            node = tables.nodes.add_row(
                flags=NODE_IS_RECOMB, time=t, metadata={"breakpoint": breakpoint}
            )

            # Note: update for Griffiths ARG
            # for interval in lineage.ancestry:
            #     tables.edges.add_row(interval.begin, interval.end, node, lineage.node)

            left = lineage.ancestry
            right = left.copy()
            right.chop(0, breakpoint)
            left.chop(breakpoint, L)
            # left = intervaltree.IntervalTree.from_tuples(
            #     (interval.begin, interval.end) for interval in left)
            # right = intervaltree.IntervalTree.from_tuples(
            #     (interval.begin, interval.end) for interval in right)
            lineages.extend([Lineage(node, left), Lineage(node, right)])


        else:
            print("CA EVENT")
            a_index = rng.randrange(k)
            b_index = rng.randrange(k - 1)
            a = lineages.pop(a_index)
            b = lineages.pop(b_index)

            print("\ta = ", a)
            print("\tb = ", b)
            c = a.ancestry.union(b.ancestry)
            # c.split_overlaps()
            c.merge_overlaps()
            # print("\tintersection = ", a.ancestry.intersection(b.ancestry))
            print("\tc = ", c)

            flags = 0
            node = tables.nodes.add_row(flags=flags, time=t, metadata={})
            # for lineage in [a, b]:
            #     for interval in lineage:
            #         tables.edges.add_row(interval.begin, interval.end, node, interval.data)


            c = intervaltree.IntervalTree.from_tuples(
                (interval.begin, interval.end) for interval in c)
            lineages.append(Lineage(node, c))

        print()

arg_sim(5, 1, 10, seed=234)
