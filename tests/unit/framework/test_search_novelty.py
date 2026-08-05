# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SearchNoveltyTracker — diminishing-returns detection on successive searches."""

from victor.framework.search_novelty import NoveltyConfig, SearchNoveltyTracker


def _search_result(paths_symbols):
    """Build a code_search tool-result dict carrying result_items."""
    return [
        {
            "tool_name": "code_search",
            "success": True,
            "result_items": [{"path": p, "qualified_name": s} for p, s in paths_symbols],
        }
    ]


def _same_five():
    return _search_result([(f"a{i}.py", f"f{i}") for i in range(5)])


def test_saturating_loop_force_completes_in_a_few_turns():
    t = SearchNoveltyTracker()
    # Warm-up turns (min_search_turns=2) don't count, then the same hits repeat.
    assert not t.record_turn(_same_five()).should_force_complete  # turn 1 (warm-up)
    assert not t.record_turn(_same_five()).should_force_complete  # turn 2 (warm-up)
    r3 = t.record_turn(_same_five())  # turn 3: ratio 0 -> low #1
    r4 = t.record_turn(_same_five())  # turn 4: low #2 -> nudge
    r5 = t.record_turn(_same_five())  # turn 5: low #3 -> force complete
    assert r3.novelty_ratio == 0.0
    assert r4.should_nudge and not r4.should_force_complete
    assert r5.should_force_complete


def test_distinct_files_never_force_completes():
    """A genuine 20-distinct-file task keeps surfacing new hits -> never saturates."""
    t = SearchNoveltyTracker()
    forced = False
    for i in range(20):
        # Each turn: 4 brand-new files.
        res = t.record_turn(_search_result([(f"f{i}_{j}.py", f"sym{i}_{j}") for j in range(4)]))
        forced = forced or res.should_force_complete
        assert res.novelty_ratio == 1.0
    assert not forced


def test_non_search_turns_are_neutral():
    t = SearchNoveltyTracker()
    # An edit/read turn (no result_items) doesn't disturb the counter.
    res = t.record_turn([{"tool_name": "edit", "success": True}])
    assert not res.had_search
    assert res.consecutive_low_novelty == 0
    assert t.record_turn(None).had_search is False


def test_narrow_single_file_research_saturates():
    """File-centric: repeatedly searching the SAME file (different queries) saturates.

    This is the U1 pattern — many narrow single-file queries re-covering a handful of files.
    """
    t = SearchNoveltyTracker()
    forced = False
    for _ in range(6):
        res = t.record_turn(_search_result([("only.py", "one")]))
        forced = forced or res.should_force_complete
    assert forced


def test_eight_files_over_many_searches_saturates():
    """The exact U1 shape: ~8 distinct files probed by many narrow searches -> saturates."""
    files = [f"victor/agent/f{i}.py" for i in range(8)]
    t = SearchNoveltyTracker()
    forced = False
    # First 8 searches each surface one new file (high novelty), then re-cover them.
    for i in range(20):
        f = files[i % 8]
        res = t.record_turn(_search_result([(f, f"sym{i}")]))
        forced = forced or res.should_force_complete
    assert forced


def test_config_tunes_force_complete_timing():
    t = SearchNoveltyTracker(NoveltyConfig(consecutive_low_novelty_limit=2, min_search_turns=1))
    t.record_turn(_same_five())  # turn 1 (warm-up, min_search_turns=1)
    t.record_turn(_same_five())  # turn 2: low #1
    assert t.record_turn(_same_five()).should_force_complete  # turn 3: low #2 -> force (limit=2)


def test_reset_clears_state():
    t = SearchNoveltyTracker()
    for _ in range(5):
        t.record_turn(_same_five())
    t.reset()
    # Fresh: the same hits are novel again.
    assert t.record_turn(_same_five()).novelty_ratio == 1.0
