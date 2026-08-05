# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Behavioral tests for research workflow escape hatches.

The repair-loop hatches (research_gap_repair_decision,
research_memory_reuse_decision) are covered in test_research_workflows.py;
this module covers the remaining conditions, the transforms, and the
registries.
"""

from victor_research.escape_hatches import (
    CONDITIONS,
    TRANSFORMS,
    competitive_threat_level,
    fact_verdict,
    format_bibliography,
    literature_relevance,
    merge_search_results,
    should_search_more,
    source_coverage_check,
    source_credibility_check,
)


class TestSourceCoverageCheck:
    def test_sufficient_coverage(self):
        ctx = {"sources": ["a", "b", "c", "d", "e"], "coverage_score": 0.9}
        assert source_coverage_check(ctx) == "sufficient"

    def test_marginal_on_decent_score(self):
        ctx = {"sources": ["a", "b"], "coverage_score": 0.65}
        assert source_coverage_check(ctx) == "marginal"

    def test_marginal_on_source_count_alone(self):
        # 4 sources >= 5 * 0.7 even with low coverage score
        ctx = {"sources": ["a", "b", "c", "d"], "coverage_score": 0.1}
        assert source_coverage_check(ctx) == "marginal"

    def test_needs_more_when_sparse(self):
        ctx = {"sources": ["a"], "coverage_score": 0.2}
        assert source_coverage_check(ctx) == "needs_more"

    def test_empty_context_needs_more(self):
        assert source_coverage_check({}) == "needs_more"


class TestShouldSearchMore:
    def test_high_coverage_proceeds(self):
        assert should_search_more({"coverage_score": 0.9}) == "proceed"

    def test_max_iterations_forces_proceed(self):
        ctx = {"coverage_score": 0.3, "search_iterations": 3, "max_iterations": 3}
        assert should_search_more(ctx) == "proceed"

    def test_significant_gaps_search_more(self):
        ctx = {
            "coverage_score": 0.4,
            "search_iterations": 1,
            "max_iterations": 3,
            "gaps": ["pricing", "benchmarks", "adoption"],
        }
        assert should_search_more(ctx) == "search_more"

    def test_few_gaps_proceed(self):
        ctx = {
            "coverage_score": 0.7,
            "search_iterations": 1,
            "max_iterations": 3,
            "gaps": ["pricing"],
        }
        assert should_search_more(ctx) == "proceed"


class TestSourceCredibilityCheck:
    def test_no_sources_low_credibility(self):
        assert source_credibility_check({}) == "low_credibility"

    def test_high_credibility(self):
        ctx = {
            "validated_sources": [
                {"credibility": 0.9},
                {"credibility": 0.85},
            ]
        }
        assert source_credibility_check(ctx) == "high_credibility"

    def test_acceptable_mid_range(self):
        ctx = {"validated_sources": [{"credibility": 0.6}, {"credibility": 0.5}]}
        assert source_credibility_check(ctx) == "acceptable"

    def test_low_credibility_average(self):
        ctx = {"validated_sources": [{"credibility": 0.2}, {"credibility": 0.3}]}
        assert source_credibility_check(ctx) == "low_credibility"

    def test_non_dict_sources_default_acceptable(self):
        ctx = {"validated_sources": ["not-a-dict"]}
        assert source_credibility_check(ctx) == "acceptable"


class TestFactVerdict:
    def test_no_evidence_unverifiable(self):
        assert fact_verdict({}) == "unverifiable"

    def test_low_confidence_unverifiable(self):
        ctx = {"supporting_evidence": ["a", "b"], "confidence": 0.1}
        assert fact_verdict(ctx) == "unverifiable"

    def test_unanimous_support_is_true(self):
        ctx = {
            "supporting_evidence": ["a", "b", "c"],
            "refuting_evidence": [],
            "confidence": 0.9,
        }
        assert fact_verdict(ctx) == "true"

    def test_mostly_true(self):
        ctx = {
            "supporting_evidence": ["a", "b", "c", "d"],
            "refuting_evidence": ["e"],
            "confidence": 0.9,
        }
        assert fact_verdict(ctx) == "mostly_true"

    def test_mixed_evidence(self):
        ctx = {
            "supporting_evidence": ["a"],
            "refuting_evidence": ["b"],
            "confidence": 0.8,
        }
        assert fact_verdict(ctx) == "mixed"

    def test_unanimous_refutation_is_false(self):
        ctx = {
            "supporting_evidence": [],
            "refuting_evidence": ["a", "b", "c"],
            "confidence": 0.9,
        }
        assert fact_verdict(ctx) == "false"


class TestLiteratureRelevance:
    def test_high_relevance_score(self):
        assert literature_relevance({"paper": {"relevance_score": 0.9}}) == "highly_relevant"

    def test_well_cited_paper_is_highly_relevant(self):
        ctx = {"paper": {"relevance_score": 0.3, "citation_count": 100}}
        assert literature_relevance(ctx) == "highly_relevant"

    def test_relevant_band(self):
        assert literature_relevance({"paper": {"relevance_score": 0.65}}) == "relevant"

    def test_marginal_band(self):
        assert literature_relevance({"paper": {"relevance_score": 0.45}}) == "marginal"

    def test_irrelevant(self):
        assert literature_relevance({"paper": {"relevance_score": 0.1}}) == "irrelevant"


class TestCompetitiveThreatLevel:
    def test_direct_competitor_with_share_is_high(self):
        ctx = {"competitor": {"is_direct_competitor": True, "market_share": 0.3}}
        assert competitive_threat_level(ctx) == "high"

    def test_overlap_and_growth_is_high(self):
        ctx = {"market_overlap": 0.8, "growth_rate": 0.3}
        assert competitive_threat_level(ctx) == "high"

    def test_fast_growth_is_emerging(self):
        ctx = {"market_overlap": 0.1, "growth_rate": 0.6}
        assert competitive_threat_level(ctx) == "emerging"

    def test_moderate_overlap_is_medium(self):
        ctx = {"market_overlap": 0.5, "growth_rate": 0.0}
        assert competitive_threat_level(ctx) == "medium"

    def test_no_signals_is_low(self):
        assert competitive_threat_level({}) == "low"


class TestMergeSearchResults:
    def test_merges_and_tags_source_types(self):
        ctx = {
            "web_results": [{"url": "https://a.com", "title": "A"}],
            "academic_results": [{"url": "https://b.edu", "title": "B"}],
            "code_results": [{"url": "https://c.dev", "title": "C"}],
        }
        merged = merge_search_results(ctx)

        assert merged["source_count"] == 3
        assert merged["by_type"] == {"web": 1, "academic": 1, "code": 1}

    def test_deduplicates_by_url_first_wins(self):
        ctx = {
            "web_results": [{"url": "https://same.com"}],
            "academic_results": [{"url": "https://same.com"}],
        }
        merged = merge_search_results(ctx)

        assert merged["source_count"] == 1
        assert merged["sources"][0]["source_type"] == "web"

    def test_skips_results_without_urls(self):
        ctx = {"web_results": [{"title": "no url"}, {"url": "https://a.com"}]}
        merged = merge_search_results(ctx)

        assert merged["source_count"] == 1


class TestFormatBibliography:
    def test_formats_entries_with_defaults(self):
        ctx = {"validated_sources": [{"url": "https://a.com"}]}
        bib = format_bibliography(ctx)

        assert bib["count"] == 1
        assert bib["style"] == "apa"
        entry = bib["entries"][0]
        assert entry["title"] == "Unknown"
        assert entry["year"] == "n.d."
        assert entry["source_type"] == "web"

    def test_respects_citation_style(self):
        ctx = {"validated_sources": [], "citation_style": "mla"}
        assert format_bibliography(ctx)["style"] == "mla"

    def test_ignores_non_dict_sources(self):
        ctx = {"validated_sources": ["raw-string", {"title": "Real", "url": "u"}]}
        bib = format_bibliography(ctx)

        assert bib["count"] == 1
        assert bib["entries"][0]["title"] == "Real"


class TestRegistries:
    def test_conditions_registry_complete(self):
        assert set(CONDITIONS) == {
            "source_coverage_check",
            "should_search_more",
            "research_gap_repair_decision",
            "research_memory_reuse_decision",
            "source_credibility_check",
            "fact_verdict",
            "literature_relevance",
            "competitive_threat_level",
        }
        assert all(callable(fn) for fn in CONDITIONS.values())

    def test_transforms_registry_complete(self):
        assert set(TRANSFORMS) == {
            "merge_search_results",
            "format_bibliography",
        }
        assert all(callable(fn) for fn in TRANSFORMS.values())
