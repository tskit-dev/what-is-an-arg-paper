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


# AncestryInterval is the equivalent of msprime's Segment class. The
# important different here is that we don't associated nodes with
# individual intervals here: because this is an ARG, nodes that
# we pass through are recorded.
#
# (The ancestral_to field is also different here, but that's because
# I realised that the way we're tracking extant ancestral material
# in msprime is unnecessarily complicated, and we can actually
# track it locally. There is potentially quite a large performance
# increase available in msprime from this.)


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: int
    right: int
    ancestral_to: int

    @property
    def span(self):
        return self.right - self.left


@dataclasses.dataclass
class Lineage:
    """
    A single lineage that is present during the simulation of the coalescent
    with recombination. The node field represents the last (as we go backwards
    in time) genome in which an ARG event occured. That is, we can imagine
    a lineage representing the passage of the ancestral material through
    a sequence of ancestral genomes in which it is not modified.
    """

    node: int
    ancestry: List[AncestryInterval]

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

    @property
    def num_recombination_links(self):
        """
        The number of positions along this lineage's genome at which a recombination
        event can occur.
        """
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


# The details of the machinery in the next two functions aren't important.
# It could be done more cleanly and efficiently. The basic idea is that
# we're providing a simple way to find the overlaps in the ancestral
# material of two or more lineages, abstracting the complex interval
# logic out of the main simulation.
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
    """
    Return an iterator over the ancestral material for the specified lineages.
    For each distinct interval at which ancestral material exists, we return
    the AncestryInterval and the corresponding list of lineages.
    """
    # See note above on the implementation - this could be done more cleanly.
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


@dataclasses.dataclass
class Node:
    time: float
    flags: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)


def fully_coalesced(lineages, n):
    """
    Returns True if all segments are ancestral to n samples in all
    lineages.
    """
    for lineage in lineages:
        for segment in lineage.ancestry:
            if segment.ancestral_to < n:
                return False
    return True


def sim_coalescent(n, rho, L, seed=None, resolved=True):
    """
    Simulate an ancestry-resolved ARG under the coalescent with recombination
    and return the tskit TreeSequence object.

    NOTE! This hasn't been statistically tested and is probably not correct.
    """
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    nodes = []
    for _ in range(n):
        lineages.append(Lineage(len(nodes), [AncestryInterval(0, L, 1)]))
        nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    t = 0

    # NOTE: the stopping condition here is more complicated because
    # we have to still keep track of segments after the MRCA has been
    # reached. We don't simulate back to the GMRCA as this may take a
    # very long time.

    # while len(lineages) > 0:
    while not fully_coalesced(lineages, n):
        # print(f"t = {t:.2f} k = {len(lineages)}")
        # for lineage in lineages:
        #     print(f"\t{lineage}")
        lineage_links = [lineage.num_recombination_links for lineage in lineages]
        total_links = sum(lineage_links)
        re_rate = total_links * rho
        t_re = math.inf if re_rate == 0 else rng.expovariate(re_rate)
        k = len(lineages)
        ca_rate = k * (k - 1) / 2
        t_ca = rng.expovariate(ca_rate)
        t_inc = min(t_re, t_ca)
        t += t_inc

        if t_inc == t_re:
            left_lineage = rng.choices(lineages, weights=lineage_links)[0]
            breakpoint = rng.randrange(left_lineage.left + 1, left_lineage.right)
            assert left_lineage.left < breakpoint < left_lineage.right
            right_lineage = left_lineage.split(breakpoint)
            lineages.append(right_lineage)
            child = left_lineage.node
            assert nodes[child].flags & NODE_IS_RECOMB == 0
            nodes[child].flags |= NODE_IS_RECOMB
            assert "breakpoint" not in nodes[child].metadata
            nodes[child].metadata["breakpoint"] = breakpoint
            for lineage in left_lineage, right_lineage:
                lineage.node = len(nodes)
                nodes.append(Node(time=t))
                if resolved:
                    for interval in lineage.ancestry:
                        tables.edges.add_row(
                            interval.left, interval.right, lineage.node, child
                        )
            if not resolved:
                tables.edges.add_row(0, breakpoint, left_lineage.node, child)
                tables.edges.add_row(breakpoint, L, right_lineage.node, child)
        else:
            a = lineages.pop(rng.randrange(len(lineages)))
            b = lineages.pop(rng.randrange(len(lineages)))
            c = Lineage(len(nodes), [])
            for interval, intersecting_lineages in merge_ancestry([a, b]):
                # if interval.ancestral_to < n:
                c.ancestry.append(interval)
                if resolved:
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )
            if not resolved:
                tables.edges.add_row(0, L, c.node, a.node)
                tables.edges.add_row(0, L, c.node, b.node)

            nodes.append(Node(time=t))
            # if len(c.ancestry) > 0:
            lineages.append(c)

    for node in nodes:
        tables.nodes.add_row(flags=node.flags, time=node.time, metadata=node.metadata)
    tables.sort()
    # TODO not sure if this is the right thing to do, but it makes it easier
    # to compare with examples.
    tables.edges.squash()
    return tables.tree_sequence()


@dataclasses.dataclass
class Individual:
    id: int = -1
    lineages: List[Lineage] = dataclasses.field(default_factory=list)
    collected_lineages: List[List[Lineage]] = dataclasses.field(
        default_factory=lambda: [[], []]
    )


def iter_lineages(individuals):
    for individual in individuals:
        for lineage in individual.lineages:
            yield lineage


def sim_wright_fisher(n, N, L, seed=None):
    """
    NOTE! This hasn't been statistically tested and is probably not correct.

    We don't keep track of the pedigree because this is inconsistent
    with the practise of dropping "pass through" nodes in which
    nothing happens.
    """
    # We don't offer a "resolved" option here, but it should be
    # easy enough to implement the resolved=False version. It should be
    # a case of skipping the second loop in which we go through the
    # collected ancestry, and instead writing out the edges directly when
    # parents are chosen
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()

    ancestors = []
    node_flags = []
    for _ in range(n):
        ind = Individual(tables.individuals.add_row(), [])
        for _ in range(2):
            node = tables.nodes.add_row(time=0, individual=ind.id)
            ind.lineages.append(Lineage(node, [AncestryInterval(0, L, 1)]))
            node_flags.append(tskit.NODE_IS_SAMPLE)
        ancestors.append(ind)

    t = 0
    while not fully_coalesced(iter_lineages(ancestors), 2 * n):
        t += 1
        # print("===")
        # print("T = ", t, "|A|=", len(ancestors))
        # for ancestor in ancestors:
        #     print(ancestor)
        # Group the ancestral lineages among the parent individuals chosen
        # among the previous generation. We map the individual's index
        # in the population to the lineages inherited from is maternal
        # and paternal genomes
        parents = {}
        for ancestor in ancestors:
            for lineage in ancestor.lineages:
                parent_index = rng.randrange(N)
                if parent_index not in parents:
                    parents[parent_index] = Individual()
                parent = parents[parent_index]
                # Randomise the order we add the lineages
                rng.shuffle(parent.collected_lineages)
                j = 0
                breakpoint = rng.randrange(1, L)
                # print("breakpoint", breakpoint)
                if lineage.left < breakpoint < lineage.right:
                    # print("effective recombination!", lineage.node)
                    right_lineage = lineage.split(breakpoint)
                    parent.collected_lineages[j].append(right_lineage)
                    j += 1
                    node_flags[lineage.node] |= NODE_IS_RECOMB
                parent.collected_lineages[j].append(lineage)

        # All the ancestral material has been distributed to the parental
        # lineages.
        ancestors.clear()

        # print("|P| = ", len(parents), "RE_children = ", recombinant_nodes)
        for parent in parents.values():
            for lineages in parent.collected_lineages:
                # These are all the lineages that have been collected
                # together for this lineage on this individual. If there
                # is at least one piece of ancestral material going through,
                # we create a node for it.
                # print("\tlineages = ")
                # for lin in lineages:
                #     print("\t\t", lin)

                merged_lineage = None
                if len(lineages) == 1 and node_flags[lineage.node] == 0:
                    merged_lineage = lineages[0]
                elif len(lineages) > 0:
                    if parent.id == -1:
                        parent.id = tables.individuals.add_row()
                    node = len(tables.nodes)
                    merged_lineage = Lineage(node, [])
                    node_flags.append(0)
                    for interval, intersecting_lineages in merge_ancestry(lineages):
                        # if interval.ancestral_to < 2 * n:  # n is *diploid* sample size
                        merged_lineage.ancestry.append(interval)
                        for child_lineage in intersecting_lineages:
                            tables.edges.add_row(
                                interval.left, interval.right, node, child_lineage.node
                            )
                    tables.nodes.add_row(time=t, individual=parent.id, metadata={})
                if merged_lineage is not None:
                    parent.lineages.append(merged_lineage)
            if len(parent.lineages) > 0:
                ancestors.append(parent)

        assert len(ancestors) <= N

    tables.nodes.flags = node_flags

    tables.sort()
    return tables.tree_sequence()


def wh99_example():
    """
    The example ARG from figure 1 of Wiuf and Hein 99, Recombination as a Point Process
    along Sequences. Each event is given a time increment of 1.
    """
    L = 7
    tables = tskit.TableCollection(L)
    t = 0
    for _ in range(3):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=t)

    def re(child, x):
        nonlocal t
        t += 1
        left_parent = tables.nodes.add_row(time=t)
        right_parent = tables.nodes.add_row(time=t)
        tables.edges.add_row(0, x, left_parent, child)
        tables.edges.add_row(x, L, right_parent, child)

    def ca(*args):
        nonlocal t
        t += 1
        parent = tables.nodes.add_row(time=t)
        for child in args:
            tables.edges.add_row(0, L, parent, child)

    re(1, x=4)
    re(2, x=2)
    ca(0, 3)
    re(5, x=5)
    ca(4, 8)
    re(10, x=3)
    ca(12, 9)
    ca(7, 11)
    re(13, x=6)
    ca(16, 6)
    ca(14, 15)
    re(18, x=1)
    ca(20, 17)
    ca(19, 21)

    tables.sort()
    return tables.tree_sequence()
