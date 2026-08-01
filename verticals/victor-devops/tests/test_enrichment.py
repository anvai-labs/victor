# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the DevOps context enrichment strategy."""

from pathlib import Path

from victor_contracts.enrichment_runtime import EnrichmentContext, EnrichmentType

from victor_devops.enrichment import DevOpsEnrichmentStrategy, _detect_infra_context


class TestInfraContextDetection:
    def test_detects_docker_from_file_mentions(self):
        context = _detect_infra_context(["Dockerfile", "app/main.py"], "build the image")

        assert "Dockerfile" in context["docker"]

    def test_detects_terraform_from_prompt(self):
        context = _detect_infra_context([], "run terraform plan for staging")

        assert "_prompt_mention" in context["terraform"]

    def test_empty_input_detects_nothing(self):
        context = _detect_infra_context([], "hello world")

        assert all(not files for files in context.values())

    def test_all_categories_initialized(self):
        context = _detect_infra_context([], "")

        assert set(context) == {"docker", "kubernetes", "terraform", "ci_cd", "ansible", "helm"}


class TestEnrichmentStrategy:
    async def test_docker_prompt_produces_docker_enrichment(self):
        strategy = DevOpsEnrichmentStrategy()
        ctx = EnrichmentContext(file_mentions=["Dockerfile"], task_type="infrastructure")

        enrichments = await strategy.get_enrichments("optimize my Dockerfile", ctx)

        sources = [e.source for e in enrichments]
        assert "devops_docker" in sources
        docker = enrichments[sources.index("devops_docker")]
        assert docker.type == EnrichmentType.PROJECT_CONTEXT
        assert "multi-stage builds" in docker.content

    async def test_unrelated_prompt_produces_no_enrichments(self):
        strategy = DevOpsEnrichmentStrategy()
        ctx = EnrichmentContext(file_mentions=[], task_type="general")

        enrichments = await strategy.get_enrichments("write a poem", ctx)

        assert enrichments == []

    async def test_multiple_domains_produce_multiple_enrichments(self):
        strategy = DevOpsEnrichmentStrategy()
        ctx = EnrichmentContext(
            file_mentions=["Dockerfile", "deployment.yaml"],
            task_type="infrastructure",
        )

        enrichments = await strategy.get_enrichments(
            "containerize the app and deploy to kubernetes", ctx
        )

        sources = {e.source for e in enrichments}
        assert "devops_docker" in sources

    def test_set_project_root(self, tmp_path: Path):
        strategy = DevOpsEnrichmentStrategy()

        strategy.set_project_root(tmp_path)

        assert strategy._project_root == tmp_path
