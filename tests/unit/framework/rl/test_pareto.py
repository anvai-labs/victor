# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for GEPA v2 Pareto frontier."""

import pytest

from victor.framework.rl.pareto import ParetoEntry, ParetoFrontier


class TestParetoEntry:
    def test_basic_creation(self):
        e = ParetoEntry(text_hash="abc", text="prompt text", generation=1)
        assert e.coverage_count == 0
        assert e.char_length == 0
        assert e.instance_scores == {}


class TestParetoDominance:
    def test_a_dominates_b(self):
        frontier = ParetoFrontier()
        # A is strictly better on all instances
        a = ParetoEntry("a", "text_a", 1, {"i1": 0.9, "i2": 0.8})
        b = ParetoEntry("b", "text_b", 1, {"i1": 0.7, "i2": 0.6})
        assert frontier._dominates(a, b)
        assert not frontier._dominates(b, a)

    def test_no_dominance_when_each_better_on_one(self):
        frontier = ParetoFrontier()
        a = ParetoEntry("a", "text_a", 1, {"i1": 0.9, "i2": 0.3})
        b = ParetoEntry("b", "text_b", 1, {"i1": 0.5, "i2": 0.8})
        assert not frontier._dominates(a, b)
        assert not frontier._dominates(b, a)

    def test_no_dominance_when_equal(self):
        frontier = ParetoFrontier()
        a = ParetoEntry("a", "text_a", 1, {"i1": 0.5})
        b = ParetoEntry("b", "text_b", 1, {"i1": 0.5})
        # Equal everywhere, not strictly better on any — no dominance
        assert not frontier._dominates(a, b)

    def test_no_dominance_with_disjoint_instances(self):
        frontier = ParetoFrontier()
        a = ParetoEntry("a", "text_a", 1, {"i1": 0.9})
        b = ParetoEntry("b", "text_b", 1, {"i2": 0.8})
        assert not frontier._dominates(a, b)


class TestParetoFrontier:
    def test_add_single_candidate(self):
        f = ParetoFrontier()
        added = f.add_candidate("h1", "text1", 1, {"i1": 0.5})
        assert added
        assert f.size == 1

    def test_prune_dominated_candidate(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.5, "i2": 0.5})
        f.add_candidate("h2", "text2", 2, {"i1": 0.9, "i2": 0.9})
        # h1 should be pruned — dominated by h2
        assert f.size == 1
        assert f.get_frontier()[0].text_hash == "h2"

    def test_non_dominated_both_survive(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.9, "i2": 0.3})
        f.add_candidate("h2", "text2", 2, {"i1": 0.3, "i2": 0.9})
        # Neither dominates the other — both survive
        assert f.size == 2

    def test_coverage_proportional_selection(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.9, "i2": 0.1})
        f.add_candidate("h2", "text2", 2, {"i1": 0.1, "i2": 0.9})
        # h1 is best on i1, h2 is best on i2 — each has coverage 1
        f._recompute_coverage()
        coverages = {c.text_hash: c.coverage_count for c in f.get_frontier()}
        assert coverages["h1"] == 1
        assert coverages["h2"] == 1

    def test_select_parent_returns_candidate(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.5})
        parent = f.select_parent()
        assert parent is not None
        assert parent.text_hash == "h1"

    def test_select_parent_empty_returns_none(self):
        f = ParetoFrontier()
        assert f.select_parent() is None

    def test_capacity_limit(self):
        f = ParetoFrontier(max_candidates=3)
        # Add 5 candidates each best on a unique instance
        for i in range(5):
            scores = {f"i{j}": (0.9 if j == i else 0.1) for j in range(5)}
            f.add_candidate(f"h{i}", f"text{i}", i, scores)
        assert f.size <= 3

    def test_update_instance_score(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.5})
        f.update_instance_score("h1", "i2", 0.8)
        entry = f.get_frontier()[0]
        assert entry.instance_scores["i2"] == 0.8

    def test_get_best_overall(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.3, "i2": 0.3})
        f.add_candidate("h2", "text2", 2, {"i1": 0.9, "i2": 0.1})
        best = f.get_best_overall()
        # h2: avg 0.5, h1: avg 0.3 — h2 is better overall
        assert best.text_hash == "h2"

    def test_reject_fully_dominated_on_add(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.9, "i2": 0.9})
        # h2 is dominated on both instances
        added = f.add_candidate("h2", "text2", 2, {"i1": 0.1, "i2": 0.1})
        assert not added
        assert f.size == 1

    def test_duplicate_hash_updates_scores(self):
        f = ParetoFrontier()
        f.add_candidate("h1", "text1", 1, {"i1": 0.5})
        f.add_candidate("h1", "text1", 1, {"i2": 0.8})
        assert f.size == 1
        entry = f.get_frontier()[0]
        assert "i1" in entry.instance_scores
        assert "i2" in entry.instance_scores
