"""
Utilities for generating and converting ARGs in various formats.
"""
import random
import math
import dataclasses
from typing import List
from typing import Any

import tskit

NODE_IS_RECOMB = 1 << 1
NODE_IS_NONCOAL_CA = 1 << 2  # TODO better name


@dataclasses.dataclass
class AncestryInterval:
    left: int
    right: int
    ancestral_to: int


@dataclasses.dataclass
class Lineage:
    node: int
    ancestry: List[AncestryInterval]

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        return s[:-2] + "]"

    @property
    def num_recombination_links(self):
        return self.right - self.left - 1

    @property
    def left(self):
        """
        Returns the leftmost position of ancestral material.
        """
        return self.ancestry[0].left

    @property
    def right(self):
        """
        Returns the rightmost position of ancestral material.
        """
        return self.ancestry[-1].right

    def split(self, breakpoint):
        """
        Splits the ancestral material for this lineage at the specified
        breakpoint, and returns a second lineage with the ancestral
        material to the right.
        """
        left_ancestry = []
        right_ancestry = []
        for interval in self.ancestry:
            if interval.right <= breakpoint:
                left_ancestry.append(interval)
            elif interval.left >= breakpoint:
                right_ancestry.append(interval)
            else:
                assert interval.left < breakpoint < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=breakpoint))
                right_ancestry.append(dataclasses.replace(interval, left=breakpoint))
        self.ancestry = left_ancestry
        return Lineage(self.node, right_ancestry)


@dataclasses.dataclass
class MappingSegment:
    left: int
    right: int
    value: Any = None


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(MappingSegment(math.inf, 0))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


def merge_ancestry(lineages):
    segments = []
    for lineage in lineages:
        for interval in lineage.ancestry:
            segments.append(
                MappingSegment(interval.left, interval.right, (lineage, interval))
            )

    for left, right, U in overlapping_segments(segments):
        ancestral_to = sum(u.value[1].ancestral_to for u in U)
        interval = AncestryInterval(left, right, ancestral_to)
        yield interval, [u.value[0] for u in U]


class ArgSimulator:
    def __init__(self, num_samples, rho, L, seed):
        self.num_samples = num_samples
        self.L = L
        self.rho = rho
        self.rng = random.Random(seed)
        self.tables = tskit.TableCollection(L)
        self.tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        self.lineages = []
        for _ in range(num_samples):
            node = self.tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
            self.lineages.append(Lineage(node, [AncestryInterval(0, L, 1)]))

    def run(self):
        t = 0
        while len(self.lineages) > 0:
            # print(f"t = {t:.2f} k = {len(lineages)}")
            # for lineage in lineages:
            #     print(f"\t{lineage}")
            lineage_links = [lineage.num_recombination_links for lineage in self.lineages]
            total_links = sum(lineage_links)
            re_rate = total_links * self.rho
            t_re = math.inf if re_rate == 0 else self.rng.expovariate(re_rate)
            k = len(self.lineages)
            ca_rate = k * (k - 1) / 2
            t_ca = self.rng.expovariate(ca_rate)
            t_inc = min(t_re, t_ca)
            t += t_inc
            if t_inc == t_re:
                lineage = self.rng.choices(self.lineages, weights=lineage_links)[0]
                breakpoint = self.rng.randrange(lineage.left + 1, lineage.right)
                assert lineage.left < breakpoint < lineage.right
                node = self.tables.nodes.add_row(
                    flags=NODE_IS_RECOMB, time=t, metadata={"breakpoint": breakpoint}
                )
                right = lineage.split(breakpoint)
                self.lineages.append(right)
                for lineage in [lineage, right]:
                    for interval in lineage.ancestry:
                        self.tables.edges.add_row(
                            interval.left, interval.right, node, lineage.node
                        )
                    lineage.node = node
            else:
                a = self.lineages.pop(self.rng.randrange(len(self.lineages)))
                b = self.lineages.pop(self.rng.randrange(len(self.lineages)))
                # print(f"\ta = {a}")
                # print(f"\tb = {b}")
                c = Lineage(len(self.tables.nodes), [])
                flags = NODE_IS_NONCOAL_CA
                for interval, intersecting_lineages in merge_ancestry([a, b]):
                    if len(intersecting_lineages) > 1:
                        flags = 0  # This is a coalescence, treat this as ordinary tree node
                    if interval.ancestral_to < self.num_samples:
                        c.ancestry.append(interval)
                    for lineage in intersecting_lineages:
                        self.tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )
                self.tables.nodes.add_row(flags=flags, time=t, metadata={})
                # print(f"\tc = {c}")
                if len(c.ancestry) > 0:
                    self.lineages.append(c)

        self.tables.sort()
        return self.tables.tree_sequence()

# NOTE! This hasn't been statistically tested and is probably not correct.
def arg_sim(n, rho, L, seed=None):
    sim = ArgSimulator(n, rho, L, seed)
    return sim.run()


ts = arg_sim(5, 0.2, 10, seed=234)
print(ts.tables)

node_labels = {}
for node in ts.nodes():
    label = str(node.id)
    if node.flags == NODE_IS_RECOMB:
        label = f"R{node.id}"
    elif node.flags == NODE_IS_NONCOAL_CA:
        label = f"N{node.id}"
    node_labels[node.id] = label
print(ts.draw_text(node_labels=node_labels))
