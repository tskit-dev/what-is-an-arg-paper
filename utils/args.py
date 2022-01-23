"""
Utilities for generating and converting ARGs in various formats.
"""
import random
import math
import collections
import dataclasses
from typing import List
from typing import Any

import tskit
import numpy as np

NODE_IS_RECOMB = 1 << 1
NODE_IS_NONCOAL_CA = 1 << 2  # TODO better name

def draw_arg(ts):

    node_labels = {}
    for node in ts.nodes():
        label = str(node.id)
        if node.flags == NODE_IS_RECOMB:
            label = f"R{node.id}"
        elif node.flags == NODE_IS_NONCOAL_CA:
            label = f"N{node.id}"
        node_labels[node.id] = label
    print(ts.draw_text(node_labels=node_labels))

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
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

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


# NOTE! This hasn't been statistically tested and is probably not correct.
def arg_sim(n, rho, L, seed=None):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    for _ in range(n):
        node = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        lineages.append(Lineage(node, [AncestryInterval(0, L, 1)]))

    t = 0
    while len(lineages) > 0:
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
            lineage = rng.choices(lineages, weights=lineage_links)[0]
            breakpoint = rng.randrange(lineage.left + 1, lineage.right)
            assert lineage.left < breakpoint < lineage.right
            node = tables.nodes.add_row(
                flags=NODE_IS_RECOMB, time=t, metadata={"breakpoint": breakpoint}
            )
            right = lineage.split(breakpoint)
            lineages.append(right)
            for lineage in [lineage, right]:
                for interval in lineage.ancestry:
                    tables.edges.add_row(
                        interval.left, interval.right, node, lineage.node
                    )
                lineage.node = node
        else:
            a = lineages.pop(rng.randrange(len(lineages)))
            b = lineages.pop(rng.randrange(len(lineages)))
            # print(f"\ta = {a}")
            # print(f"\tb = {b}")
            c = Lineage(len(tables.nodes), [])
            flags = NODE_IS_NONCOAL_CA
            for interval, intersecting_lineages in merge_ancestry([a, b]):
                if len(intersecting_lineages) > 1:
                    flags = 0  # This is a coalescence, treat this as ordinary tree node
                if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                for lineage in intersecting_lineages:
                    tables.edges.add_row(
                        interval.left, interval.right, c.node, lineage.node
                    )
            tables.nodes.add_row(flags=flags, time=t, metadata={})
            # print(f"\tc = {c}")
            if len(c.ancestry) > 0:
                lineages.append(c)

    tables.sort()
    return tables.tree_sequence()


# NOTE! This hasn't been statistically tested and is probably not correct.
def node_arg_sim(n, rho, L, seed=None):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()

    lineages = []
    for _ in range(n):
        node = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        lineages.append(Lineage(node, [AncestryInterval(0, L, 1)]))
    t = 0
    while len(lineages) > 0:
        print(f"t = {t:.2f} k = {len(lineages)}")
        for lineage in lineages:
            print(f"\t{lineage}")
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
            lineage = rng.choices(lineages, weights=lineage_links)[0]
            breakpoint = rng.randrange(lineage.left + 1, lineage.right)
            assert lineage.left < breakpoint < lineage.right
            node = tables.nodes.add_row(
                flags=NODE_IS_RECOMB, time=t, metadata={"breakpoint": breakpoint}
            )
            tables.edges.add_row(-math.inf, math.inf, node, lineage.node)
            lineage.node = node
            right = lineage.split(breakpoint)
            lineages.append(right)
        else:
            a = lineages.pop(rng.randrange(len(lineages)))
            b = lineages.pop(rng.randrange(len(lineages)))
            c = Lineage(len(tables.nodes), [])
            flags = NODE_IS_NONCOAL_CA
            for interval, intersecting_lineages in merge_ancestry([a, b]):
                if len(intersecting_lineages) > 1:
                    flags = 0  # This is a coalescence, treat this as ordinary tree node
                if interval.ancestral_to < n:
                    c.ancestry.append(interval)
            tables.nodes.add_row(flags=flags, time=t, metadata={})
            tables.edges.add_row(-math.inf, math.inf, c.node, a.node)
            tables.edges.add_row(-math.inf, math.inf, c.node, b.node)
            # print(f"\tc = {c}")
            if len(c.ancestry) > 0:
                lineages.append(c)
    # tables.sort()
    # return tables.tree_sequence()
    return tables


def convert_arg(tables):
    """
    Converts the specified non-ancestry tracking ARG to a tskit ARG.
    """
    out = tables.copy()
    out.edges.clear()
    nodes = sorted(tables.nodes, key=lambda x: x.time)

    parent = collections.defaultdict(list)
    children = collections.defaultdict(list)
    for edge in tables.edges:
        parent[edge.child].append(edge.parent)
        children[edge.parent].append(edge.child)

    lineages = []
    node_id = 0
    n = 0
    while node_id < len(nodes) and (nodes[node_id].flags & tskit.NODE_IS_SAMPLE != 0):
        node = nodes[node_id]
        print("sample:", node)
        assert nodes[node_id].time == 0
        lineages.append(Lineage(node_id, [AncestryInterval(0, tables.sequence_length, 1)]))
        node_id += 1
        n += 1

    for node_id in range(node_id, len(nodes)):
        node = nodes[node_id]
        parent = node_id
        # print("VISIT", node_id, node.time)
        # for lineage in lineages:
        #     print(f"\t{lineage}")
        if (node.flags & NODE_IS_RECOMB) != 0:
            # print("RE EVENT")
            child = children[parent][0]
            assert len(children[parent]) == 1
            # print(f"parent = {parent} child = {child}")
            breakpoint = node.metadata["breakpoint"]
            for lineage in lineages:
                if lineage.node == child:
                    break
            right = lineage.split(breakpoint)
            lineages.append(right)
            for lineage in [lineage, right]:
                for interval in lineage.ancestry:
                    out.edges.add_row(
                        interval.left, interval.right, parent, child,
                    )
                lineage.node = parent
        else:
            # print("COAL")
            # print(children[parent])
            assert len(children[parent]) == 2
            children_lineages = []
            for child in children[parent]:
                for j in range(len(lineages)):
                    if lineages[j].node == child:
                        children_lineages.append(lineages.pop(j))
                        break
            a, b  = children_lineages
            c = Lineage(parent, [])
            flags = NODE_IS_NONCOAL_CA
            for interval, intersecting_lineages in merge_ancestry([a, b]):
                if len(intersecting_lineages) > 1:
                    flags = 0  # This is a coalescence, treat this as ordinary tree node
                if interval.ancestral_to < n:
                    c.ancestry.append(interval)
                for lineage in intersecting_lineages:
                    out.edges.add_row(
                        interval.left, interval.right, c.node, lineage.node
                    )
            # print(node.flags, flags)
            # assert node.flags == flags
            # tables.nodes.add_row(flags=flags, time=t, metadata={})
            # print(f"\ta = {a}")
            # print(f"\tb = {b}")
            # print(f"\tc = {c}")
            if len(c.ancestry) > 0:
                lineages.append(c)
        print(f"t = {node.time:.2f}")
        for lineage in lineages:
            print(f"\t{lineage}")
    print(out)
    out.sort()
    return out.tree_sequence()

n = 3
rho = 0.3
L = 10
seed = 234
ts = arg_sim(n, rho, L, seed=seed)
# tables = node_arg_sim(n, rho, L, seed=seed)
# print(tables)
# ts2 = convert_arg(tables)

draw_arg(ts)
draw_arg(ts2)
